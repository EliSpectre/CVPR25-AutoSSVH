import numpy as np
import scipy.io as sio 
import scipy.sparse as sp
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F

def run_kmeans(x, args,gpu_id,device):
    """
    Args:
        x: data to be clustered
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}
    
    for seed, num_cluster in enumerate(args.num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        
        clus = faiss.Clustering(d, k)

        # 关闭输出日志
        clus.verbose = False  # 关闭详细输出日志

        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 200
        clus.min_points_per_centroid = 2

        res = faiss.StandardGpuResources()
        cfg_cluster = faiss.GpuIndexFlatConfig()
        cfg_cluster.useFloat16 = False
        cfg_cluster.device = gpu_id  
        index = faiss.GpuIndexFlatL2(res, d, cfg_cluster)  #存疑，这里是利用l2距离查询

        clus.train(x, index)   

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
        
        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        density = args.kmeans_temperature*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(device)
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    

        im2cluster = torch.LongTensor(im2cluster).to(device)               
        density = torch.Tensor(density).to(device)
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


def mAP(cateTrainTest, IX, num_return_NN=None):
    numTrain, numTest = IX.shape

    num_return_NN = numTrain if not num_return_NN else num_return_NN

    apall = np.zeros((numTest, 1))
    yescnt_all = np.zeros((numTest, 1))
    for qid in range(numTest):
        query = IX[:, qid]
        x, p = 0, 0

        for rid in range(num_return_NN):
            if cateTrainTest[query[rid], qid]:
                x += 1
                p += x/(rid*1.0 + 1.0)
        yescnt_all[qid] = x
        if not p: apall[qid] = 0.0
        else: apall[qid] = p/(num_return_NN*1.0)

    return np.mean(apall),apall,yescnt_all  


def topK(cateTrainTest, HammingRank, k=500):
    numTest = cateTrainTest.shape[1]

    precision = np.zeros((numTest, 1))
    recall = np.zeros((numTest, 1))

    topk = HammingRank[:k, :]

    for qid in range(numTest):
        retrieved = topk[:, qid]
        rel = cateTrainTest[retrieved, qid]
        retrieved_relevant_num = np.sum(rel)
        real_relevant_num = np.sum(cateTrainTest[:, qid])

        precision[qid] = retrieved_relevant_num/(k*1.0)
        recall[qid] = retrieved_relevant_num/(real_relevant_num*1.0)

    return precision.mean(), recall.mean()