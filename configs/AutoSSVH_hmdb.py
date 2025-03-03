# model
model_name = 'AutoSSVH'
use_checkpoint = None
feature_size = 4096
hidden_size = 256
max_frames = 25
nbits = 16
AutoSSVH_type = 'small'

# dataset
dataset = 'hmdb'
workers = 1
batch_size = 128
mask_prob = 0.7

# train
seed = 1
num_epochs = 350
a = 1
temperature = 0.5
tau_plus = 0.1
train_num_sample = 3570

# Component Voting Hash Learning(CVH)
CVH= True
train_CVH = True
num_cluster = [250,400,600]
warmup_epoch = 50
kmeans_temperature = 0.2
b = 0.2
data_drop_rate = 0.

# test
test_batch_size = 128
test_num_sample = 3570 # test database
query_num_sample = 1530 # test query

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
train_feat_path = [data_root + 'hmdb_train_feats.h5']

# path:test
test_feat_path = [data_root + 'hmdb_train_feats.h5'] # database
label_path = [data_root + 'hmdb_train_labels.mat']
query_feat_path = [data_root + 'hmdb_test_feats.h5'] # query
query_label_path = [data_root + 'hmdb_test_labels.mat']

# path:save
save_dir = home_root + "checkpoint/" + dataset
file_path = f"{save_dir}/{model_name}_{nbits}bit"
log_path = f"{home_root}logs/{dataset}_{nbits}bit"


