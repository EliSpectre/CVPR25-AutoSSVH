from .AutoSSVH_optim import AutoSSVH_opt_schedule

def get_opt_schedule(cfg, model):
    if cfg.model_name in ["AutoSSVH"]:
        return AutoSSVH_opt_schedule(cfg, model)
