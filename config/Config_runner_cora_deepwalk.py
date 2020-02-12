
import os
import time

f_debug = True

# parameters for method name
method_name = 'GANRL/'
initial_method = 'deepwalk/'
f_initial_random = False

f_method_name = method_name.split('/')[0]

f_graph_set = ['G', 'AG', 'HG']
f_graph = f_graph_set[0]
f_initial_graph = f_graph_set[0]

# parameters for node type
node_type_4 = 'node'
node_type_5 = 'attnode'

f_neighborhood_set_total = ['rw', 'metarw', 'one_hop', 'same_att']
f_neighborhood_set = [{'nei_type': f_neighborhood_set_total[0], 'center_type': 'all', 'node_type': 'all'},  # node_type = 'all' or list of node type
                      ]  # center_type = 'all' or list of center node type

f_GAN_set = ['mix', 'sep']
f_GAN = f_GAN_set[0]

# parameters for data path
data_path = 'data/cora_labeled/'

G_path = data_path+'graph.edgelist'
A_path = data_path+'feature.attmat'
label_path = data_path+'group.txt'

# parameters for simulate random walks
num_walks = 40
walk_length = 40

# parameters for result path
tt = time.ctime().replace(' ','-')
result_path = 'results/'+ method_name + initial_method + data_path.split('/')[-2]+'/' + f_graph + '/'

interdata_path = 'interdata/' + data_path.split('/')[-2]+'/'

embedding_path = 'results/'+ initial_method + data_path.split('/')[-2]+'/' + f_initial_graph + '/'

# parameters for data type
weighted = False
directed = False

# parameters for neighborhood
num_walks_per_neighborhood = num_walks//4   # number of walks to construct a neighborhood
f_remove_neighbors_set = ['low_freq', 'intersection', None]
f_remove_neighbors = f_remove_neighbors_set[2]
# frequency_threshold = num_walks_per_neighborhood
frequency_threshold = 0
att_threshold = None
window_size = 10

max_memory_data_size = 1000000000

# meta path
metapath_nan = [node_type_4, node_type_5, node_type_4]
metapath_nnn = [node_type_4,node_type_4,node_type_4]
metapath = metapath_nan

# parameters for GANRL
embedding_size = 128  # Dimension of the embedding vector.
multi_processing = True
load_model = False  # whether loading existing model for initialization
save_steps = 10
n_epochs = 50  # number of outer loops
update_ratio = 1    # updating ratio when choose the trees
flag_uniform_unigram = False
flag_sample_multi_attempt = True    # used in sample function in GANRL, indicate whether to attempt other chances if sample a neighbor without neighbors
convergence_threshold = -5e-4
attempt_times = 20


# parameters for generator
learn_rate_gen = 1e-4
l2_loss_weight_gen = 0
loss_func_set = ['sig','ne','nce']
loss_func_gen = loss_func_set[2]   # possible loss function: nce loss, negative loss, sigmoid loss ******
num_sampled_gen = 10    # for negative sampling
n_epochs_gen = 4  # number of inner loops for the generator
gen_interval = 2  # sample new nodes for the generator for every gen_interval iterations
batch_size_gen = 128  # batch size for the generator
n_sample_gen = [40]  # number of samples for the generator
neighborhood_probability_ratio = 0
flag_center_nodes_based_on_frequency = False
flag_zero_bias_gen = False

flag_teacher_forcing = False
n_epochs_tea = 10
tea_interval = n_epochs_tea
batch_size_tea = 64
n_sample_tea = [20]

# parameters for discriminator
learn_rate_dis = 1e-4
l2_loss_weight_dis = 0
n_epochs_dis = 2  # number of inner loops for the discriminator    ******
dis_interval = 1  # sample new nodes for the discriminator for every dis_interval iterations    ******
batch_size_dis = 128  # batch size for the discriminator
n_sample_dis = [20]  # number of positive/negative samples for the discriminator
flag_zero_bias_dis = False

# parameters for evaluation
normalize = False

# parameters for classification
flag_classification = True
node_type_for_classification = node_type_4
train_ratio_set = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
train_ratio = 0.7
classifier_set = ['LR','LinearSVM']
classifier = classifier_set[0]
max_iter = 10000

# parameters for link prediction
flag_link_prediction = True
test_link_ratio_set = [0.2]
validation_link_ratio = 0
positive_ratio = 0.5
num_link_samples = 20000
k_of_precisionK_pred = [10,100,500]

flag_convergence_learning = False
batch_num_interval = 1e4


