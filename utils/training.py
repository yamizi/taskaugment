import torch
import numpy as np
import os
from utils.multitask_losses import compute_mtloss, AverageMeter
from utils.losses import MSELoss
from utils.attacks.pgd import MTPGD


epochs_checkpoints = [60,90,95]
def adjust_learning_rate(initial_lr, optimizer, epoch,n_epochs):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    lr = initial_lr * (1 - epoch / n_epochs) ** 0.9

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def adv_val_model(network,test_loader,params, use_cuda=True, experiment=None,criteria=None, epoch=0):
    network.eval()

    tasks_labels = params.get("tasks")
    aux_task = tasks_labels[1] if len(tasks_labels) > 1 else None

    correct = 0
    adv_correct = 0
    nb_elements = 0
    loss_mse = 0
    loss_total = 0
    adv_loss_total = 0
    adv_loss_mse = 0
    adv_loss_total = 0

    attack_tasks = params.get("attack_target", ["class_object"])
    max_eps = int(params.get("max_eps", 8))
    step_eps = int(params.get("step_eps", 2))
    steps = int(params.get("steps", 4))

    atk = MTPGD(network, attack_tasks, criteria, eps=max_eps / 255, alpha=step_eps / 255, steps=steps)

    for i, (x_original, y) in enumerate(test_loader):
        print(f"eval batch {i}")
        if use_cuda:
            data = x_original.to("cuda:0")
            target = [y_.to("cuda:0").float() for (k, y_) in y.items()]

        adv_train = atk(x_original, y)
        with torch.no_grad():
            adv_output = network(adv_train)
            output = network(data)
            nb_elements += output["rep"].size(0)

            loss, _, _ = compute_mtloss(criteria, output, dict(zip(y.keys(), target)),
                                                             equally=True, avg_losses=None)

            adv_loss, _, _ = compute_mtloss(criteria, adv_output, dict(zip(y.keys(), target)),
                                        equally=True, avg_losses=None)

            pred = output["class_object"].data.max(1, keepdim=True)[1]
            correct += pred.eq(y["class_object"].data.max(1, keepdim=True)[1].to("cuda:0")).sum()
            loss_total += loss * output["rep"].size(0)

            adv_pred = adv_output["class_object"].data.max(1, keepdim=True)[1]
            adv_correct += adv_pred.eq(y["class_object"].data.max(1, keepdim=True)[1].to("cuda:0")).sum()
            adv_loss_total += adv_loss * output["rep"].size(0)

            if aux_task:
                aux = output[aux_task]
                aux_truth = y[aux_task].to("cuda:0")
                loss_ = MSELoss(reduction="sum")
                loss_mse += loss_(aux, aux_truth).item()

                adv_aux = adv_output[aux_task]
                adv_loss_mse += loss_(adv_aux, aux_truth).item()

    acc = 100. * correct / nb_elements
    mse_aux = loss_mse / nb_elements
    val_loss = loss_total.item() / nb_elements

    adv_acc = 100. * adv_correct / nb_elements
    adv_mse_aux = adv_loss_mse / nb_elements
    adv_val_loss = adv_loss_total.item() / nb_elements

    if experiment is not None:
        experiment.log_metric("val_acc", acc, step=epoch)
        experiment.log_metric("val_loss", val_loss, step=epoch)
        experiment.log_metric("val_{}".format(aux_task), mse_aux, step=epoch)

        experiment.log_metric("adv_acc", adv_acc, step=epoch)
        experiment.log_metric("adv_loss", adv_val_loss, step=epoch)
        experiment.log_metric("adv_{}".format(aux_task), adv_mse_aux, step=epoch)

    return acc, mse_aux, val_loss, adv_acc, adv_mse_aux, adv_val_loss

def adv_train_model(network,train_loader,test_loader,params={},n_epochs=25,criteria=None,
                  log_interval=100,use_cuda=True, experiment=None, loss_fn=torch.nn.BCEWithLogitsLoss()):

    tasks_labels = params.get("tasks")
    aux_task = tasks_labels[1] if len(tasks_labels)>1 else None

    strategy = params.get("strategy","adv_only")
    uniqueid = params.get("uniqueid",strategy)

    os.makedirs(os.path.join('.','models',uniqueid),exist_ok=True)

    attack_tasks = params.get("attack_target",["class_object"])
    max_eps = int(params.get("max_eps", 8))
    step_eps = int(params.get("max_eps",2))
    steps = int(params.get("steps",4))

    lr = params.get("lr", 0.01)
    momentum = params.get("momentum", 0.9)
    start_epoch = 1
    params["epochs"] = n_epochs
    params["lr_change"] = 0
    params["lr"] = lr

    if params.get("optim") == 'sgd':
        print("Using SGD")
        optimizer = torch.optim.SGD(network.parameters(), lr, momentum=momentum)
    else:
        print("Using Adam")
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    checkpoint_file = params.get("checkpoint_file")

    if checkpoint_file is not None:
        checkpoint_path = os.path.join('.','models',checkpoint_file+".pth")
        checkpoint_file = torch.load(checkpoint_path)
        network.load_state_dict(checkpoint_file['model'])
        optimizer.load_state_dict(checkpoint_file['optimizer'])
        start_epoch = int(checkpoint_file['epoch'])
        n_epochs = n_epochs+start_epoch

        print(f"loading checkpoint from {checkpoint_path}; start epoch {start_epoch}; total epoch: {n_epochs} ")

    atk = MTPGD(network, attack_tasks, criteria, eps=max_eps / 255, alpha=step_eps / 255, steps=steps)

    def train(epoch, avg_losses):
        strategy = params.get("strategy", "adv_only")
        train_corrects = 0
        train_total = 0
        train_aux = 0

        #lr = adjust_learning_rate(params, optimizer, epoch)
        for batch_idx, (x_original, y) in enumerate(train_loader):

            if use_cuda:
                x_train = x_original.to("cuda:0")
                y_train = [y_.to("cuda:0").float() for (k,y_) in y.items()]

            clean_output = network(x_train)
            nb_elements = clean_output["rep"].size(0)

            optimizer.zero_grad()

            if (strategy == "clean/adv"):
                if(np.random.random()<0.5):
                    strategy = "adv_only"
                else:
                    strategy = "clean_only"

            if(strategy=="adv_only"):
                adv_train = atk(x_train, y)
                adv_output = network(adv_train)
                optimizer.zero_grad()

                _, loss_adv, _ = compute_mtloss(criteria, adv_output, dict(zip(y.keys(), y_train)), equally=True,
                                                loss_dict={}, avg_losses=None)
                loss = list(loss_adv.values())

            elif(strategy=="clean_only"):

                optimizer.zero_grad()
                _, loss_clean, _ = compute_mtloss(criteria,clean_output, dict(zip(y.keys(),y_train)), equally=True, loss_dict={},avg_losses=None)

                loss = list(loss_clean.values())

            elif(strategy=="clean+adv"):

                adv_train = atk(x_train, y)
                adv_output = network(adv_train)
                optimizer.zero_grad()

                _, loss_clean, _ = compute_mtloss(criteria,clean_output, dict(zip(y.keys(),y_train)), equally=True, loss_dict={},avg_losses=None)

                _, loss_adv, _ = compute_mtloss(criteria, adv_output, dict(zip(y.keys(), y_train)), equally=True,
                                                              loss_dict={}, avg_losses=None)

                loss = list(loss_clean.values()) +list(loss_adv.values())

            loss = torch.mean(torch.stack(loss))
            loss.backward()
            optimizer.step()

            ls = loss.item()
            pred = clean_output["class_object"].data.max(1, keepdim=True)[1]

            correct = pred.eq(y["class_object"].data.max(1, keepdim=True)[1].to("cuda:0")).sum().item()
            train_corrects += correct
            train_total += nb_elements

            if aux_task:
                aux = clean_output[aux_task]
                aux_truth = y[aux_task]
                loss_ = MSELoss(reduction="sum")
                loss_mse = loss_(aux, aux_truth.to("cuda:0"))
                train_aux += loss_mse.item()


            if batch_idx % log_interval == 0:

                acc = 100. * correct / nb_elements
                acc_total = 100. * train_corrects / train_total
                mse_aux = train_aux / train_total

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Batch Acc: {:.3f} Total Acc: {:.3f}'.format(
                    epoch, batch_idx * len(x_train), len(train_loader.sampler),
                           100. * batch_idx / len(train_loader), ls, acc, acc_total))
                train_losses.append(ls)

                model_path = os.path.join('.','models',uniqueid,
                                          'train_{}'.format(params.get("model")))

                torch.save(network.state_dict(),model_path+'_model.pth' )
                torch.save(optimizer.state_dict(), model_path + '_optimizer.pth')

                if experiment is not None:
                    experiment.log_metric("loss",ls,epoch=batch_idx,step=epoch)
                    experiment.log_metric("train_acc", acc, epoch=batch_idx, step=epoch)
                    experiment.log_metric("total_acc", acc_total, epoch=batch_idx, step=epoch)
                    experiment.log_metric("mse_{}".format(aux_task), mse_aux, epoch=batch_idx, step=epoch)


    def validate(epoch, avg_losses, best_loss):

        network.eval()
        loss = 0
        correct = 0
        nb_elements = 0
        loss_mse = 0
        loss_total = 0

        with torch.no_grad():
            for x_original, y in test_loader:

                if use_cuda:
                    data = x_original.to("cuda:0")
                    target = [y_.to("cuda:0").float() for (k, y_) in y.items()]

                output = network(data)
                nb_elements +=output["rep"].size(0)
                loss_dict = {}
                if criteria is None:
                    loss = loss_fn(output, target)
                else:
                    loss, loss_dict, avg_losses = compute_mtloss(criteria, output, dict(zip(y.keys(), target)),
                                                                 equally=True, loss_dict=loss_dict,
                                                                 avg_losses=avg_losses)

                pred = output["class_object"].data.max(1, keepdim=True)[1]
                correct += pred.eq(y["class_object"].data.max(1, keepdim=True)[1].to("cuda:0")).sum()
                loss_total += loss*output["rep"].size(0)

                if aux_task:
                    aux = output[aux_task]
                    aux_truth = y[aux_task].to("cuda:0")
                    loss_ = MSELoss(reduction="sum")
                    loss_mse += loss_(aux, aux_truth).item()

        acc = 100. * correct / nb_elements
        mse_aux = loss_mse/ nb_elements
        val_loss = loss_total.item()/nb_elements

        if val_loss<best_loss:
            best_loss = val_loss
            checkpoint = {
                'epoch':     epoch,
                'model': network.state_dict(),
                'optimizer': optimizer.state_dict(),
                'tasks':list(output.keys()),
                'loss':best_loss,
                'lr_sched': optimizer.param_groups[0].get("lr")}

            model_path = os.path.join('.','models',uniqueid,'best_{}.pth'.format(params.get("model")))
            torch.save(checkpoint, model_path)


        if epoch%100==0:
            checkpoint = {
                'epoch': epoch,
                'model': network.state_dict(),
                'optimizer': optimizer.state_dict(),
                'tasks': list(output.keys()),
                'loss': best_loss,
                'lr_sched': optimizer.param_groups[0].get("lr")}

            model_path = os.path.join('.','models',uniqueid,'checkpoint_{}_{}.pth'.format(params.get("model"),epoch))
            torch.save(checkpoint, model_path)

        if experiment is not None:
            experiment.log_metric("val_acc", acc, step=epoch)
            experiment.log_metric("val_loss", val_loss, step=epoch)
            experiment.log_metric("val_{}".format(aux_task), mse_aux, step=epoch)
            experiment.log_metric("lr", optimizer.param_groups[0].get("lr"), step=epoch)

            for k, ls in avg_losses.items():
                experiment.log_metric("val_loss_{}".format(k), ls.avg, step=epoch)

        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, nb_elements, acc))

        return val_loss



    train_losses = []
    network.train()

    avg_losses = {}
    avg_losses_val = {}
    for c_name, criterion_fun in criteria.items():
        avg_losses[c_name] = AverageMeter()
        avg_losses_val[c_name] = AverageMeter()

    best_loss = np.inf
    for epoch in range(start_epoch, n_epochs + 1):
        train(epoch, avg_losses)

        if epoch*n_epochs//100 in epochs_checkpoints:
            lr = adjust_learning_rate(lr, optimizer, epoch, n_epochs)

        best_loss = validate(epoch, avg_losses_val,best_loss)

    model_path = os.path.join('.','models',uniqueid, 'final_{}.pth'.format(params.get("model")))
    torch.save(network, model_path)

    return network
