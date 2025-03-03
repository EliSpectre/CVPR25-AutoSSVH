from .AutoSSVH_inference import AutoSSVH_inference

def get_inference(cfg, data, model,device):
    if cfg.model_name in ["AutoSSVH"]:
        return AutoSSVH_inference(cfg, data, model,device)