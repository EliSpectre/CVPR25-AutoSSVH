from .AutoSSVH import AutoSSVH


def get_model(cfg):
    if cfg.model_name == 'AutoSSVH':
        model = AutoSSVH(cfg)
    return model
    