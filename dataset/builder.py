from .AutoSSVH_dataset import get_AutoSSVH_train_loader, get_AutoSSVH_eval_loader

def get_train_data(cfg):
    if cfg.model_name in ["AutoSSVH"]:
        return get_AutoSSVH_train_loader(cfg, shuffle=True)

def get_eval_data(cfg):
    if cfg.model_name in ["AutoSSVH"]:
        return get_AutoSSVH_eval_loader(cfg)