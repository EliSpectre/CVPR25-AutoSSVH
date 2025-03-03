# model
model_name = "AutoSSVH"
use_checkpoint = None
feature_size = 2048
hidden_size = 256
max_frames = 30
nbits = 64
AutoSSVH_type = 'small'

# dataset
dataset = 'activitynet'
workers = 1
batch_size = 128
mask_prob = 0.5

# train
seed = 1
num_epochs = 800
a = 0.2
temperature = 0.5
tau_plus = 0.05
train_num_sample = 9722

# Component Voting Hash Learning(CVH)
CVH=True
num_cluster = [250,400,600]#[250 500 1000]
warmup_epoch = 100  #40 60- 80 
kmeans_temperature = 0.2
b = 0.01
data_drop_rate = 0.2 

# test
test_batch_size = 128
test_num_sample = 3758
query_num_sample = 1000

# optimizer
optimizer_name = 'Adam'
schedule = 'StepLR'
lr = 1e-4
min_lr = 1e-6
lr_decay_rate = 20
lr_decay_gamma = 0.9
weight_decay = 0.0

# path
data_root = f"data/{dataset}/"
home_root = './'

# path:train
train_feat_path = [data_root + 'train_feats.h5']

# path:test
test_feat_path = [data_root + 'test_feats.h5'] # database
label_path = [data_root + 're_label.mat']
query_feat_path = [data_root + 'query_feats.h5'] # query
query_label_path = [data_root + 'q_label.mat']

# path:save
save_dir = home_root + "checkpoint/" + dataset
file_path = f"{save_dir}/{model_name}_{nbits}bit"
log_path = f"{home_root}logs/{dataset}_{nbits}bit"

