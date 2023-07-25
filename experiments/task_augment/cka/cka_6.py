import sys
sys.path.append(".")
from utils import init_comet
import torch
import json
from os.path import join
from utils.xrayvision import init_seed, init_dataset, init_model
from utils.cka import CKA
from experiments import get_argparser

parser = get_argparser("cka_6")
cfg = parser.parse_args()
print(cfg)


if __name__ == '__main__':
    init_seed(cfg)
    cfg.dataset="chex"

    model_dir = join(cfg.output_dir,cfg.dataset,"best")
    experiment = init_comet(args=vars(cfg), project_name=cfg.name, workspace=cfg.workspace)

    # Setting the dataset
    train_dataset, test_dataset = init_dataset(cfg)

    # create models
    init_seed(cfg)
    model1 = init_model(cfg, test_dataset)
    init_seed(cfg)
    model2 = init_model(cfg, test_dataset)


    chex_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.batch_size,
                                              shuffle=cfg.shuffle,
                                              num_workers=cfg.threads,
                                              pin_memory=cfg.cuda)

    weights_file = join(model_dir, f"{cfg.labelfilter}.pt")
    model1.load_state_dict(torch.load(weights_file).state_dict())

    layer_names_1 = [module_name for (module_name, module) in model1.named_modules() if
                     not isinstance(module, torch.nn.Sequential) and "conv" in module_name]
    layer_names_1 = [e for (i, e) in enumerate(layer_names_1) if i % cfg.skip_layers == 0]


    cfg.dataset = "nih"
    # Setting the dataset
    _, nih_dataset = init_dataset(cfg)
    nih_loader = torch.utils.data.DataLoader(nih_dataset,
                                              batch_size=cfg.batch_size,
                                              shuffle=cfg.shuffle,
                                              num_workers=cfg.threads,
                                              pin_memory=cfg.cuda)


    with torch.no_grad():
        cka = CKA(model1, model1,
                  model1_name=cfg.labelfilter,
                  model2_name=cfg.labelfilter,
                  model1_layers=layer_names_1,
                  model2_layers=layer_names_1,
                  device="cuda" if cfg.cuda else "cpu")

        cka.compare(chex_loader, nih_loader, max_batches=cfg.skip_batch)
        results = cka.export()
    #cka.plot_results()
    results["CKA"] = results.get("CKA").cpu().numpy().tolist()
    results["Cx"] = cfg.labelfilter
    results["Cx'"] = cfg.labelfilter
    results["Ex"] = "best"
    results["Ex'"] = "best"
    results["Dx"] = "chex"
    results["Dx'"] = "chex"

    experiment.log_asset_data(json.dumps(results), "file.json")