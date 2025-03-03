import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import sample
import sys
sys.path.append('..')
from torch.autograd import Variable


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def dcl(out_1, out_2, batch_size, temperature=0.5, tau_plus=0.1,device=None):
    out_1 = F.normalize(out_1, dim=1)
    out_2 = F.normalize(out_2, dim=1)

    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = get_negative_mask(batch_size).to(device)
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    if True:
        N = batch_size * 2 - 2
        Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
        Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
    else:
        Ng = neg.sum(dim=-1)

    loss = (- torch.log(pos / (pos + Ng) )).mean()
    return loss
    
def Loss_CVH(all_hashcode_norm,index,criterion,cluster_result,batchsize,cfg,device):
        proto_labels = []
        proto_logits = []
        for n, (im2cluster,prototypes,density) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'],cluster_result['density'])):
            # get positive prototypes
            pos_proto_id = im2cluster[index]
            pos_prototypes = prototypes[pos_proto_id]    
            
            # sample negative prototypes
            all_proto_id = [i for i in range(im2cluster.max()+1)]       
            neg_proto_id = list(set(all_proto_id)-set(pos_proto_id.tolist()))
            sample_size = min(batchsize,len(neg_proto_id))
            #neg_proto_id = sample(neg_proto_id,batchsize) #sample r negative prototypes 
            neg_proto_id = sample(neg_proto_id,sample_size) 
            neg_prototypes = prototypes[neg_proto_id]    

            proto_selected = torch.cat([pos_prototypes,neg_prototypes],dim=0)
            # compute prototypical logits
            logits_proto = torch.mm(all_hashcode_norm,proto_selected.t())

            # targets for prototype assignment
            labels_proto = torch.linspace(0, all_hashcode_norm.size(0)-1, steps=all_hashcode_norm.size(0)).long().to(device)
            
            # scaling temperatures for the selected prototypes
            temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).to(device)],dim=0)]  
            logits_proto /= temp_proto
            
            proto_labels.append(labels_proto)
            proto_logits.append(logits_proto)
        loss_proto = 0.

        for proto_out,proto_target in zip(proto_logits, proto_labels):
            loss_proto += criterion(proto_out, proto_target)  

        # average loss across all sets of prototypes
        loss_proto /= len(cfg.num_cluster) 
        return loss_proto




def AutoSSVH_criterion(cfg, data, model, epoch, i, total_len, logger,cluster_result=None,criterion=None,device=None):
    data = {key: value.to(device) for key, value in data.items()}
    batchsize = data["visual_word"].size(0)
    device = data["visual_word"].device
    index = data["index"].squeeze()

    bool_masked_pos_1 = data["mask"][:,0,:].to(device, non_blocking=True).flatten(1).to(torch.bool)
    bool_masked_pos_2 = data["mask"][:,1,:].to(device, non_blocking=True).flatten(1).to(torch.bool)

        

    frame_1, hash_code_1,_ = model.forward(data["visual_word"], bool_masked_pos_1)
    frame_2, hash_code_2,_ = model.forward(data["visual_word"], bool_masked_pos_2)

    hash_code_1 = torch.mean(hash_code_1, 1)
    hash_code_2 = torch.mean(hash_code_2, 1)

    # recon_loss
    labels_1 = data["visual_word"][bool_masked_pos_1].reshape(batchsize, -1, cfg.feature_size)
    labels_2 = data["visual_word"][bool_masked_pos_2].reshape(batchsize, -1, cfg.feature_size)
    recon_loss = F.mse_loss(frame_1, labels_1) + F.mse_loss(frame_2, labels_2)

    # contra_loss(loss_vc)
    loss_vc = dcl(hash_code_1, hash_code_2, batchsize, temperature=cfg.temperature, tau_plus=cfg.tau_plus,device=device)
    
    # loss_cvh
    loss_cvh = 0.0
    if cfg.CVH and cluster_result is not None:
        hash_code_norm_1 = F.normalize(hash_code_1, dim=1)
        hash_code_norm_2 = F.normalize(hash_code_2, dim=1)
        loss_cvh = Loss_CVH(hash_code_norm_1, index, criterion, cluster_result, batchsize, cfg, device)+Loss_CVH(hash_code_norm_2, index, criterion, cluster_result, batchsize, cfg, device)
        loss_cvh = loss_cvh/2

    loss = recon_loss + cfg.a * loss_vc + cfg.b * loss_cvh
    if i % 10 == 0 or batchsize < cfg.batch_size:  
        logger.info('Epoch:[%d/%d] Step:[%d/%d] reconstruction_loss: %.2f loss_vc: %.2f loss_cvh: %.2f' \
            % (epoch+1, cfg.num_epochs, i, total_len,\
            recon_loss.data.cpu().numpy(), loss_vc.data.cpu().numpy(),\
            loss_cvh))

    return loss

