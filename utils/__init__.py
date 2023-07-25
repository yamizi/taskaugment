from app_config import COMET_APIKEY
from comet_ml import Experiment
import time

def init_comet(args, project_name="stegano-draft", workspace="yamizi"):
    if project_name=="" or project_name==None:
        return None

    timestamp = time.time()
    args["timestamp"] = timestamp
    workspace = args.get("workspace", workspace)
    xp = args.get("xp", args.get("algorithm", ""))
    xp_param = args.get("xp_param", args.get("max_eps", ""))
    experiment_name = "{}_{}_{}_{}".format(xp, args.get("dataset",""), xp_param, timestamp)
    experiment = Experiment(api_key=COMET_APIKEY,
                            project_name=project_name,
                            workspace=workspace,
                            auto_param_logging=False, auto_metric_logging=False,
                            parse_args=False, display_summary=False, disabled=False)

    experiment.set_name(experiment_name)
    experiment.log_parameters(args)

    return experiment