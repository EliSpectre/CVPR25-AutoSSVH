from .AutoSSVH_loss import AutoSSVH_criterion

def get_loss(cfg, data, model, epoch, i, total_len, logger,cluster_result,criterion,device):
    if cfg.model_name in ["AutoSSVH"]:
        return AutoSSVH_criterion(cfg, data, model, epoch, i, total_len, logger,cluster_result,criterion,device)

