'''
the discriminator model of the GAN framework
'''

import tensorflow as tf


class Discriminator(object):
    def __init__(self, Config, n_node, node_emd_init):
        self.n_node = n_node
        self.node_emd_init = node_emd_init

        with tf.variable_scope('discriminator'):
            self.embedding_matrix = tf.get_variable(name="embedding",
                                                    shape=self.node_emd_init.shape,
                                                    initializer=tf.constant_initializer(self.node_emd_init),
                                                    trainable=True)
            if Config.flag_zero_bias_dis:
                self.bias_vector = tf.zeros([self.n_node])
            else:
                self.bias_vector = tf.Variable(tf.zeros([self.n_node]))

        self.node_id = tf.placeholder(tf.int32, shape=[None])
        self.node_neighbor_id = tf.placeholder(tf.int32, shape=[None])
        self.label = tf.placeholder(tf.float32, shape=[None])

        self.node_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_id)
        self.node_neighbor_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_neighbor_id)
        self.bias = tf.gather(self.bias_vector, self.node_neighbor_id)
        self.score = tf.reduce_sum(tf.multiply(self.node_embedding, self.node_neighbor_embedding), axis=1) + self.bias

        self.l2_loss = Config.l2_loss_weight_dis * (
                tf.nn.l2_loss(self.node_neighbor_embedding) +
                tf.nn.l2_loss(self.node_embedding) +
                tf.nn.l2_loss(self.bias))
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.score)) + self.l2_loss
        optimizer = tf.train.AdamOptimizer(Config.learn_rate_dis)
        self.d_updates = optimizer.minimize(self.loss)
        self.score = tf.clip_by_value(self.score, clip_value_min=-20, clip_value_max=20)
        self.reward = tf.log(1 + tf.exp(self.score))

        self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding_matrix), 1, keepdims=True))