import sys
sys.path.append(".")
from utils import init_comet
import torch
import json
from os.path import join
from utils.xrayvision import init_seed, init_dataset, init_model
from utils.cka import CKA
from experiments import get_argparser

parser = get_argparser("cka_3")
cfg = parser.parse_args()
print(cfg)


if __name__ == '__main__':
    init_seed(cfg)
    cfg.dataset="chex"

    model_dir = join(cfg.output_dir,cfg.dataset,"best")
    experiment = init_comet(args=vars(cfg), project_name=cfg.name, workspace=cfg.workspace)

    # Setting the dataset
    train_dataset, test_dataset = init_dataset(cfg)
    data_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.batch_size,
                                              shuffle=cfg.shuffle,
                                              num_workers=cfg.threads,
                                              pin_memory=cfg.cuda)


    # create models
    model2 = init_model(cfg, test_dataset)
    model2_label = cfg.labelfilter
    weights_file = join(model_dir, f"{model2_label}.pt")
    model2.load_state_dict(torch.load(weights_file).state_dict())

    layer_names_2 = [module_name for (module_name, module) in model2.named_modules() if
                     not isinstance(module, torch.nn.Sequential) and "conv" in module_name]
    layer_names_2 = [e for (i, e) in enumerate(layer_names_2) if i % cfg.skip_layers == 0]

    models = ["Atelectasis","Atelectasis-Edema-Effusion-Cardiomegaly-Consolidation-Pneumothorax-Pneumonia"]
    for model in models:
        cfg.labelfilter = model
        _, test_dataset2 = init_dataset(cfg)
        model1 = init_model(cfg, test_dataset2)
        weights_file = join(model_dir, f"{model}.pt")
        model1.load_state_dict(torch.load(weights_file).state_dict())

        layer_names_1 = [module_name for (module_name, module) in model1.named_modules() if
                         not isinstance(module, torch.nn.Sequential) and "conv" in module_name]
        layer_names_1 = [e for (i, e) in enumerate(layer_names_1) if i % cfg.skip_layers == 0]

        with torch.no_grad():
            cka = CKA(model1, model2,
                      model1_name=model,
                      model2_name=model2_label,
                      model1_layers=layer_names_1,
                      model2_layers=layer_names_2,
                      device="cuda" if cfg.cuda else "cpu")

            cka.compare(data_loader, max_batches=cfg.skip_batch)
            results = cka.export()
        results["CKA"] = results.get("CKA").cpu().numpy().tolist()
        results["Cx"] = cfg.labelfilter
        results["Cx'"] = model2_label
        results["Ex"] = "best"
        results["Ex'"] = "best"
        results["Dx"] = "chex"
        results["Dx'"] = "chex"

        experiment.log_asset_data(json.dumps(results), f"file{model}.json")