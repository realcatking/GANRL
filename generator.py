'''
the generator model of the GAN framework
'''

import tensorflow as tf


class Generator(object):
    def __init__(self, Config, n_node, node_emd_init, unigrams):
        self.n_node = n_node
        self.node_emd_init = node_emd_init  # list of embeddings of each type node

        with tf.variable_scope('generator'):
            self.embedding_matrix = tf.get_variable(name="embedding",
                                                    shape=self.node_emd_init.shape,
                                                    initializer=tf.constant_initializer(self.node_emd_init),
                                                    trainable=True)
            if Config.flag_zero_bias_gen:
                self.bias_vector = tf.zeros([self.n_node])
            else:
                self.bias_vector = tf.Variable(tf.zeros([self.n_node]))

        self.node_id = tf.placeholder(tf.int32, shape=[None])
        self.node_neighbor_id = tf.placeholder(tf.int64, shape=[None])
        self.reward = tf.placeholder(tf.float32, shape=[None])

        self.root_set = tf.placeholder(tf.int32, shape=[None])
        self.root_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.root_set)  # batch_size * n_embed
        self.root_score = tf.matmul(self.root_embedding, self.embedding_matrix, transpose_b=True) + self.bias_vector
        self.root_score = tf.clip_by_value(self.root_score, clip_value_min=-20, clip_value_max=20)

        self.all_score = tf.matmul(self.embedding_matrix, self.embedding_matrix, transpose_b=True) + self.bias_vector
        self.node_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_id)  # batch_size * n_embed
        self.node_neighbor_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_neighbor_id)
        self.bias = tf.gather(self.bias_vector, self.node_neighbor_id)
        self.score = tf.reduce_sum(self.node_embedding * self.node_neighbor_embedding, axis=1) + self.bias
        # self.prob = tf.clip_by_value(tf.nn.sigmoid(self.score), 1e-5, 1)
        self.l2_loss = Config.l2_loss_weight_gen * (tf.nn.l2_loss(self.node_neighbor_embedding) + tf.nn.l2_loss(self.node_embedding) + tf.nn.l2_loss(
            self.bias))
        if Config.loss_func_gen == 'sig':
            self.loss_tmp = self.sig_loss(self.embedding_matrix, self.bias_vector, self.node_neighbor_id,
                                          self.node_embedding)
            self.loss = -tf.reduce_mean(self.loss_tmp * self.reward) + self.l2_loss
        else:
            def loss_default():
                raise Exception('unrecognized loss type, '
                                'the valid types are one of the following:\n'
                                'nce\n'
                                'ne\n'
                                'arti_ne\n'
                                )
            loss_dict = {'nce': tf.nn.nce_loss,
                         'ne': tf.nn.sampled_softmax_loss}
            vocabulary_size = self.n_node
            sampled_values = tf.nn.fixed_unigram_candidate_sampler(tf.reshape(self.node_neighbor_id,(-1,1)), 1, Config.num_sampled_gen, True, vocabulary_size,
                                                                   distortion=0.75, num_reserved_ids=0, num_shards=1,
                                                                   shard=0,
                                                                   unigrams=unigrams)
            self.loss_tmp = loss_dict.get(Config.loss_func_gen, loss_default)(
                    weights=self.embedding_matrix,
                    biases=self.bias_vector,
                    labels=tf.reshape(self.node_neighbor_id,(-1,1)),
                    inputs=self.node_embedding,
                    num_sampled=Config.num_sampled_gen,
                    num_classes=vocabulary_size,
                    sampled_values=sampled_values)

            self.loss = tf.reduce_mean(self.loss_tmp * self.reward) + self.l2_loss
        optimizer = tf.train.AdamOptimizer(Config.learn_rate_gen)
        self.g_updates = optimizer.minimize(self.loss)

        self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding_matrix), 1, keepdims=True))

        if Config.flag_teacher_forcing:
            if Config.loss_func_gen == 'sig':
                self.loss_teacherforcing = -tf.reduce_mean(self.loss_tmp) + self.l2_loss
            else:
                self.loss_teacherforcing = tf.reduce_mean(self.loss_tmp) + self.l2_loss
            self.teacherforcing_updates = tf.train.AdamOptimizer(Config.learn_rate_gen).minimize(self.loss_teacherforcing)

    def sig_loss(self,weight, bias, label,embedding):
        label_weight = tf.nn.embedding_lookup(weight, label)
        label_bias = tf.gather(bias, label)
        score = tf.reduce_sum(embedding * label_weight, axis=1) + label_bias
        loss = tf.log(tf.clip_by_value(tf.nn.sigmoid(score), 1e-5, 1))
        return loss


class Generator_hete(object):
    def __init__(self, Config, n_node, node_emd_init, unigrams_type):
        self.n_node = n_node
        self.node_emd_init = node_emd_init

        with tf.variable_scope('generator'):
            self.embedding_matrix = tf.get_variable(name="embedding",
                                                    shape=self.node_emd_init.shape,
                                                    initializer=tf.constant_initializer(self.node_emd_init),
                                                    trainable=True)
            if Config.flag_zero_bias_gen:
                self.bias_vector = tf.zeros([self.n_node])
            else:
                self.bias_vector = tf.Variable(tf.zeros([self.n_node]))

        self.node_id = tf.placeholder(tf.int32, shape=[None])
        self.node_neighbor_id = tf.placeholder(tf.int64, shape=[None])
        self.reward = tf.placeholder(tf.float32, shape=[None])
        self.type_word_input = tf.placeholder(tf.int64, shape=[None])

        self.root_set = tf.placeholder(tf.int32, shape=[None])
        self.root_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.root_set)  # batch_size * n_embed
        self.root_score = tf.matmul(self.root_embedding, self.embedding_matrix, transpose_b=True) + self.bias_vector
        self.root_score = tf.clip_by_value(self.root_score, clip_value_min=-20, clip_value_max=20)

        self.all_score = tf.matmul(self.embedding_matrix, self.embedding_matrix, transpose_b=True) + self.bias_vector
        self.node_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_id)  # batch_size * n_embed

        self.weights = tf.nn.embedding_lookup(self.embedding_matrix, self.type_word_input)
        self.biases = tf.nn.embedding_lookup(self.bias_vector, self.type_word_input)

        self.node_neighbor_embedding = tf.nn.embedding_lookup(self.weights, self.node_neighbor_id)
        self.bias = tf.gather(self.biases, self.node_neighbor_id)
        self.score = tf.reduce_sum(self.node_embedding * self.node_neighbor_embedding, axis=1) + self.bias
        self.l2_loss = Config.l2_loss_weight_gen * (tf.nn.l2_loss(self.node_neighbor_embedding) + tf.nn.l2_loss(self.node_embedding) + tf.nn.l2_loss(
            self.bias))

        def loss_default():
            raise Exception('unrecognized loss type, '
                            'the valid types are one of the following:\n'
                            'nce\n'
                            'ne\n'
                            'arti_ne\n'
                            )
        loss_type_dict = {'nce': tf.nn.nce_loss,
                     'ne': tf.nn.sampled_softmax_loss}
        vocabulary_size = self.n_node
        vocabulary_size_type = {}
        sampled_values_type = {}
        self.loss_tmp_type = {}
        self.loss_type = {}
        self.g_updates_type = {}
        for word_type in unigrams_type:
            vocabulary_size_type[word_type] = len(unigrams_type[word_type])
            sampled_values_type[word_type] = tf.nn.fixed_unigram_candidate_sampler(tf.reshape(self.node_neighbor_id,(-1,1)), 1, Config.num_sampled_gen, True, vocabulary_size_type[word_type],
                                                                   distortion=0.75, num_reserved_ids=0, num_shards=1,
                                                                   shard=0,
                                                                   unigrams=unigrams_type[word_type])
            self.loss_tmp_type[word_type] = loss_type_dict.get(Config.loss_func_gen, loss_default)(
                    weights=self.weights,
                    biases=self.biases,
                    labels=tf.reshape(self.node_neighbor_id,(-1,1)),
                    inputs=self.node_embedding,
                    num_sampled=Config.num_sampled_gen,
                    num_classes=vocabulary_size_type[word_type],
                    sampled_values=sampled_values_type[word_type])

            self.loss_type[word_type] = tf.reduce_mean(self.loss_tmp_type[word_type] * self.reward) + self.l2_loss
            self.g_updates_type[word_type] = tf.train.AdamOptimizer(Config.learn_rate_gen).minimize(self.loss_type[word_type])

        self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding_matrix), 1, keepdims=True))

        if Config.flag_teacher_forcing:
            self.loss_teacherforcing_type = {}
            self.teacherforcing_updates_type = {}
            for word_type in unigrams_type:
                self.loss_teacherforcing_type[word_type] = tf.reduce_mean(self.loss_tmp_type[word_type]) + self.l2_loss
                self.teacherforcing_updates_type[word_type] = tf.train.AdamOptimizer(Config.learn_rate_gen).minimize(self.loss_teacherforcing_type[word_type])

