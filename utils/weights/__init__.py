"""
Weighting strategies from https://github.com/median-research-group/LibMTL
MIT Licence
"""

from utils.weights.weighting import EW, CAGrad, IMTL, GradVac, PCGrad, MGDA
from utils.weights.abstract_arch import AbsArchitecture

def init_strategy(cfg, model, dataset, device ):
    strategy = cfg.weight_strategy
    weight_args = {}

    if strategy=="cag":
        strat =  CAGrad
        weight_args["calpha"] = 0.5
        weight_args["rescale "] = 1

    elif strategy=="mgda":
        strat =  MGDA
        weight_args["mgda_gn"] = "none"

    elif strategy=="imtl":
        strat = IMTL

    elif strategy=="gv":
        strat = GradVac
        weight_args["beta"] = 0.5

    elif strategy=="pcg":
        strat =  PCGrad

    else:
        strat =  EW

    class MTLmodel(AbsArchitecture, strat):
        def __init__(self, model, device, kwargs={}):
            tasks= model.tasks
            if int(cfg.force_cosine)>0:
                tasks.append("cosine")

            super(MTLmodel, self).__init__(model.tasks, encoder_class=None, decoders=model.task_to_decoder, rep_grad=False, multi_input=False, device=device, **kwargs)
            self.encoder = model.encoder
            self.init_param()

    strat = MTLmodel(model=model.model,device=device) if hasattr(model,"model") else MTLmodel(model=model,device=device)
    return strat, weight_args

