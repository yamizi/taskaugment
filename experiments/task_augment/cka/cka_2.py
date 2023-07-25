import sys
sys.path.append(".")
from utils import init_comet
import torch
import json
from os.path import join
from utils.xrayvision import init_seed, init_dataset, init_model
from utils.cka import CKA
from experiments import get_argparser

parser = get_argparser("cka_2")
cfg = parser.parse_args()
print(cfg)


if __name__ == '__main__':
    init_seed(cfg)
    cfg.dataset="chex"

    model1_dir = join(cfg.output_dir,"chex","best")
    model2_dir = join(cfg.output_dir, "nih", "best")

    experiment = init_comet(args=vars(cfg), project_name=cfg.name, workspace=cfg.workspace)

    # Setting the dataset
    train_dataset, test_dataset = init_dataset(cfg)

    # create models
    model1 = init_model(cfg, test_dataset)
    model2 = init_model(cfg, test_dataset)


    data_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.batch_size,
                                              shuffle=cfg.shuffle,
                                              num_workers=cfg.threads,
                                              pin_memory=cfg.cuda)

    weights1_file = join(model1_dir, f"{cfg.labelfilter}.pt")
    weights2_file = join(model2_dir, f"{cfg.labelfilter}.pt")

    print("loading model from ", weights1_file)
    print("loading model from ", weights2_file)
    model1.load_state_dict(torch.load(weights1_file).state_dict())
    model2.load_state_dict(torch.load(weights2_file).state_dict())

    layer_names_1 = [module_name for (module_name, module) in model1.named_modules() if
                     not isinstance(module, torch.nn.Sequential) and "conv" in module_name]
    layer_names_1 = [e for (i, e) in enumerate(layer_names_1) if i % cfg.skip_layers == 0]

    layer_names_2 = [module_name for (module_name, module) in model2.named_modules() if
                     not isinstance(module, torch.nn.Sequential) and "conv" in module_name]
    layer_names_2 = [e for (i, e) in enumerate(layer_names_2) if i % cfg.skip_layers == 0]

    with torch.no_grad():
        cka = CKA(model1, model2,
                  model1_name="nih "+cfg.labelfilter,
                  model2_name="chex "+cfg.labelfilter,
                  model1_layers=layer_names_1,
                  model2_layers=layer_names_2,
                  device="cuda" if cfg.cuda else "cpu")

        cka.compare(data_loader, max_batches=cfg.skip_batch)
        results = cka.export()
    #cka.plot_results()
    results["CKA"] = results.get("CKA").cpu().numpy().tolist()
    results["Cx"] = cfg.labelfilter
    results["Cx'"] = cfg.labelfilter
    results["Ex"] = "best"
    results["Ex'"] = "best"
    results["Dx"] = "chex"
    results["Dx'"] = "nih"

    experiment.log_asset_data(json.dumps(results), "file.json")