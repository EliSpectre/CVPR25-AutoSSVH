# model
model_name = 'AutoSSVH'
use_checkpoint = None
feature_size = 4096
hidden_size = 256
max_frames = 25
nbits = 64
AutoSSVH_type = 'small'

# dataset
dataset = 'fcv'
workers = 1
batch_size = 512
mask_prob = 0.75

# train
seed = 1
num_epochs = 805
a = 1.0
temperature = 0.5
tau_plus = 0.1
train_num_sample = 45585

# Component Voting Hash Learning(CVH)
CVH=True
num_cluster = [250,500,1000]#[250,400,600]#
warmup_epoch = 100  #40 60- 80 
kmeans_temperature = 0.2
b = 0.01
data_drop_rate = 0.

# test
test_batch_size = 128
test_num_sample = 45600

# optimizer
optimizer_name = 'Adam'
schedule = 'StepLR'
lr = 1e-4
min_lr = 1e-5
lr_decay_rate = 20
lr_decay_gamma = 0.9
weight_decay = 0.0

# path
data_root = f"data/{dataset}/"
home_root = './'

# path:train
train_feat_path = [data_root + 'fcv_train_feats.h5']

# path:test
test_feat_path = [data_root + 'fcv_test_feats.h5'] # database+query
label_path = [data_root + 'fcv_test_labels.mat']

# path:save
save_dir = home_root + "checkpoint/" + dataset
file_path = f"{save_dir}/{model_name}_{nbits}bit"
log_path = f"{home_root}logs/{dataset}S5VH_{nbits}bit"+"cluster"+"wohashvoting"