'''
the GANRL model
'''

import tensorflow as tf
import numpy as np
import os
import math
import tqdm
from tqdm import trange

import generator, discriminator
import Evaluation, Read_Write_Embeddings


class GANRL(object):
    def __init__(self, Config, G, embeddings, result_path, neighborhood, center_type_set, node_type_set, type_word_list = None, debug_data = [None,None], node_pair_prox_array = None, unigrams = None):
        print('start GANRL\n')
        print("reading graphs...")
        self.graph = G
        del G
        self.Config = Config
        self.n_node = len(self.graph)
        self.debug_path = result_path+'debug/'
        if not os.path.exists(self.debug_path):
            os.makedirs(self.debug_path)
        self.log_dir = result_path+'log/'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.link_samples = debug_data[0]
        self.label_list = debug_data[1]
        self.node_pair_prox_array = node_pair_prox_array
        self.batch_count = 0

        self.neighborhood = neighborhood
        self.num_neighborhood_type = len(neighborhood)
        self.neighborhood_index_set = []
        self.neighborhood_repeat_set = []
        for nei_i in range(len(neighborhood)):
            self.neighborhood_index_set.append(0)
            self.neighborhood_repeat_set.append(len(neighborhood[nei_i]))
        self.root_nodes = [i for i in range(self.n_node)]

        print("reading initial embeddings...")
        self.node_embed_init_d = embeddings
        self.node_embed_init_g = embeddings
        if unigrams == None or Config.flag_uniform_unigram:
            self.unigrams = [1]*self.n_node
        else:
            self.unigrams = unigrams

        self.type_word_list = type_word_list
        if type_word_list != None:
            self.total_type = type_word_list.keys()
            self.center_type_set = center_type_set
            self.node_type_set = node_type_set
            self.unigrams_type = {}
            self.type_word_array = {}
            for word_type in type_word_list:
                self.type_word_array[word_type] = np.array(type_word_list[word_type], dtype=np.int32)
                self.unigrams_type[word_type] = np.array(self.unigrams)[type_word_list[word_type]].tolist()

        print("building GAN model...")
        self.discriminator = None
        self.generator = None
        self.build_generator()
        self.build_discriminator()

        self.latest_checkpoint = tf.train.latest_checkpoint(self.log_dir)
        self.saver = tf.train.Saver()

        self.config = tf.ConfigProto(
            log_device_placement=False
        )
        self.config.gpu_options.allow_growth = True
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init_op)


    def build_generator(self):
        """initializing the generator"""

        with tf.variable_scope("generator"):
            if self.type_word_list!=None and self.Config.loss_func_gen != 'sig':
                self.generator = generator.Generator_hete(self.Config, n_node=self.n_node, node_emd_init=self.node_embed_init_g,
                                                          unigrams_type=self.unigrams_type)
            else:
                self.generator = generator.Generator(self.Config, n_node=self.n_node, node_emd_init=self.node_embed_init_g, unigrams=self.unigrams)

    def build_discriminator(self):
        """initializing the discriminator"""

        with tf.variable_scope("discriminator"):
            self.discriminator = discriminator.Discriminator(self.Config, n_node=self.n_node, node_emd_init=self.node_embed_init_d)

    def train(self):
        # restore the model from the latest checkpoint if exists
        checkpoint = tf.train.get_checkpoint_state(self.log_dir)
        if checkpoint and checkpoint.model_checkpoint_path and self.Config.load_model:
            print("loading the checkpoint: %s" % checkpoint.model_checkpoint_path)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

        results = self.evaluation(0)
        if results:
            result_old = results[0]
            result_max = 0
        count = 0

        for epoch in trange(self.Config.n_epochs, desc='global loop'):

            # D-steps
            center_nodes = []
            neighbor_nodes = []
            labels = []
            for d_epoch in trange(self.Config.n_epochs_dis, desc='loop for dis'):
                # generate new nodes for the discriminator for every dis_interval iterations
                if d_epoch % self.Config.dis_interval == 0:
                    if self.type_word_list==None:
                        center_nodes, neighbor_nodes, labels = self.prepare_data_for_d()
                    else:
                        center_nodes, neighbor_nodes, labels = self.prepare_data_for_d_hete()
                    print('the number of samples for discriminator is %d' % len(center_nodes))

                # training
                train_size = len(center_nodes)
                start_list = list(range(0, train_size, self.Config.batch_size_dis))
                np.random.shuffle(start_list)
                for start in start_list:
                    end = start + self.Config.batch_size_dis
                    self.sess.run(self.discriminator.d_updates,
                                  feed_dict={self.discriminator.node_id: np.array(center_nodes[start:end]),
                                             self.discriminator.node_neighbor_id: np.array(neighbor_nodes[start:end]),
                                             self.discriminator.label: np.array(labels[start:end])})
                if self.Config.f_debug:
                    d_loss,d_score,d_norm,d_l2_loss = self.sess.run([self.discriminator.loss, self.discriminator.score,self.discriminator.norm,self.discriminator.l2_loss],
                                           feed_dict={self.discriminator.node_id: np.array(center_nodes),
                                                      self.discriminator.node_neighbor_id: np.array(neighbor_nodes),
                                                      self.discriminator.label: np.array(labels)})
                    print('\ndis loss: {}, dis target loss: {}, dis l2 loss: {}, dis_norm_mean: {}, dis_score_max: {}, dis_score_min: {}\n'.format(d_loss,d_loss-d_l2_loss,d_l2_loss,np.mean(d_norm),d_score.max(),d_score.min()))

            # G-steps
            for g_epoch in trange(self.Config.n_epochs_gen, desc='loop for gen'):
                if g_epoch % self.Config.gen_interval == 0:
                    if self.type_word_list==None:
                        node_1, node_2, reward = self.prepare_data_for_g()
                        print('the number of samples for generator is %d\n' % len(node_1))
                        print('\nreward max: {}, reward min: {}, reward mean: {}\n'.format(reward.max(), reward.min(),np.mean(reward)))
                    else:
                        node_1, node_2, reward = self.prepare_data_for_g_hete()
                        if self.Config.loss_func_gen != 'sig':
                            new_node_2 = {}
                            for word_type in node_2:
                                new_node_2[word_type] = []
                                for n in node_2[word_type]:
                                    new_node_2[word_type].append(self.type_word_list[word_type].index(n))
                        for word_type in node_2:
                            print('the number of %s type samples for generator is %d\n' % (word_type, len(node_1[word_type])))

                # training
                if self.type_word_list==None:
                    train_size = len(node_1)
                    start_list = list(range(0, train_size, self.Config.batch_size_gen))
                    np.random.shuffle(start_list)
                    for start in start_list:
                        if self.Config.flag_convergence_learning:
                            modes = [self.generator, self.discriminator]
                            model_name = ["gen", "dis"]
                            results_cov = []
                            if self.batch_count % self.Config.batch_num_interval == 0:
                                for i in range(2):
                                    embedding_matrix = self.sess.run(modes[i].embedding_matrix)
                                    roc_set = Evaluation.compute_roc_of_proximity(self.node_pair_prox_array,
                                                                                  embedding_matrix)
                                    results_cov.append(
                                        'batch: ' + str(self.batch_count) + ': ' + model_name[i] + ": " + str(roc_set[0]) + " | " + str(
                                            roc_set[1]) + " | " + str(
                                            roc_set[2]) + "\n")
                                with open(self.debug_path + 'covergence' + '-' + self.Config.tt, mode="a+") as f:
                                    f.writelines(results_cov)
                            self.batch_count += 1

                        end = start + self.Config.batch_size_gen
                        self.sess.run(self.generator.g_updates,
                                      feed_dict={self.generator.node_id: np.array(node_1[start:end]),
                                                 self.generator.node_neighbor_id: np.array(node_2[start:end]),
                                                 self.generator.reward: np.array(reward[start:end])})
                    if self.Config.f_debug:
                        g_loss, g_score, g_norm, g_l2_loss = self.sess.run(
                            [self.generator.loss, self.generator.score, self.generator.norm, self.generator.l2_loss],
                            feed_dict={self.generator.node_id: np.array(node_1),
                                       self.generator.node_neighbor_id: np.array(node_2),
                                       self.generator.reward: np.array(reward)})
                        print(
                            '\ngen loss: {}, gen target loss: {}, gen l2 loss:{}, gen norm mean: {}, gen score max: {}, gen score min: {}\n'.format(
                                g_loss, g_loss - g_l2_loss, g_l2_loss, np.mean(g_norm), g_score.max(), g_score.min()))
                else:
                    if self.Config.loss_func_gen == 'sig':
                        for word_type in node_1:
                            train_size = len(node_1[word_type])
                            start_list = list(range(0, train_size, self.Config.batch_size_gen))
                            np.random.shuffle(start_list)
                            for start in start_list:
                                if self.Config.flag_convergence_learning:
                                    modes = [self.generator, self.discriminator]
                                    model_name = ["gen", "dis"]
                                    results_cov = []
                                    if self.batch_count % self.Config.batch_num_interval == 0:
                                        for i in range(2):
                                            embedding_matrix = self.sess.run(modes[i].embedding_matrix)
                                            roc_set = Evaluation.compute_roc_of_proximity(self.node_pair_prox_array,
                                                                                          embedding_matrix)
                                            results_cov.append(
                                                'batch: ' + str(self.batch_count) + ': ' + model_name[i] + ": " + str(
                                                    roc_set[0]) + " | " + str(
                                                    roc_set[1]) + " | " + str(
                                                    roc_set[2]) + "\n")
                                        with open(self.debug_path + 'covergence' + '-' + self.Config.tt,
                                                  mode="a+") as f:
                                            f.writelines(results_cov)
                                    self.batch_count += 1

                                end = start + self.Config.batch_size_gen
                                self.sess.run(self.generator.g_updates,
                                              feed_dict={self.generator.node_id: np.array(node_1[word_type][start:end]),
                                                         self.generator.node_neighbor_id: np.array(node_2[word_type][start:end]),
                                                         self.generator.reward: np.array(reward[word_type][start:end])})

                            if self.Config.f_debug:
                                g_loss,g_score,g_norm,g_l2_loss = self.sess.run([self.generator.loss,self.generator.score,self.generator.norm,self.generator.l2_loss],
                                                   feed_dict={self.generator.node_id: np.array(node_1[word_type]),
                                                              self.generator.node_neighbor_id: np.array(node_2[word_type]),
                                                              self.generator.reward: np.array(reward[word_type])})
                                print('\n{}: gen loss: {}, gen target loss: {}, gen l2 loss:{}, gen norm mean: {}, gen score max: {}, gen score min: {}\n'.format(word_type,g_loss,g_loss-g_l2_loss,g_l2_loss,np.mean(g_norm),g_score.max(),g_score.min()))
                    else:
                        for word_type in node_1:
                            train_size = len(node_1[word_type])
                            start_list = list(range(0, train_size, self.Config.batch_size_gen))
                            np.random.shuffle(start_list)
                            for start in start_list:
                                if self.Config.flag_convergence_learning:
                                    modes = [self.generator, self.discriminator]
                                    model_name = ["gen", "dis"]
                                    results_cov = []
                                    if self.batch_count % self.Config.batch_num_interval == 0:
                                        for i in range(2):
                                            embedding_matrix = self.sess.run(modes[i].embedding_matrix)
                                            roc_set = Evaluation.compute_roc_of_proximity(self.node_pair_prox_array,
                                                                                          embedding_matrix)
                                            results_cov.append(
                                                'batch: ' + str(self.batch_count) + ': ' + model_name[i] + ": " + str(
                                                    roc_set[0]) + " | " + str(
                                                    roc_set[1]) + " | " + str(
                                                    roc_set[2]) + "\n")
                                        with open(self.debug_path + 'covergence' + '-' + self.Config.tt,
                                                  mode="a+") as f:
                                            f.writelines(results_cov)
                                    self.batch_count += 1

                                end = start + self.Config.batch_size_gen
                                self.sess.run(self.generator.g_updates_type[word_type],
                                              feed_dict={self.generator.node_id: np.array(node_1[word_type][start:end]),
                                                         self.generator.node_neighbor_id: np.array(
                                                             new_node_2[word_type][start:end]),
                                                         self.generator.reward: np.array(reward[word_type][start:end]),
                                                         self.generator.type_word_input: self.type_word_array[word_type]})

                            if self.Config.f_debug:
                                g_loss, g_score, g_norm, g_l2_loss = self.sess.run(
                                    [self.generator.loss_type[word_type], self.generator.score, self.generator.norm,
                                     self.generator.l2_loss],
                                    feed_dict={self.generator.node_id: np.array(node_1[word_type]),
                                               self.generator.node_neighbor_id: np.array(new_node_2[word_type]),
                                               self.generator.reward: np.array(reward[word_type]),
                                               self.generator.type_word_input: self.type_word_array[word_type]})
                                print(
                                    '\n{}: gen loss: {}, gen target loss: {}, gen l2 loss:{}, gen norm mean: {}, gen score max: {}, gen score min: {}\n'.format(
                                        word_type, g_loss, g_loss - g_l2_loss, g_l2_loss, np.mean(g_norm), g_score.max(),
                                        g_score.min()))

            # Teacher forcing steps
            if self.Config.flag_teacher_forcing:
                for t_epoch in trange(self.Config.n_epochs_tea, desc='loop for tea'):
                    if t_epoch % self.Config.tea_interval == 0:
                        if self.type_word_list!=None:
                            center_nodes, neighbor_nodes = self.prepare_data_for_t_hete()
                            if self.Config.loss_func_gen != 'sig':
                                new_neighbor_nodes = {}
                                for word_type in neighbor_nodes:
                                    new_neighbor_nodes[word_type] = []
                                    for n in neighbor_nodes[word_type]:
                                        new_neighbor_nodes[word_type].append(self.type_word_list[word_type].index(n))
                        else:
                            center_nodes, neighbor_nodes = self.prepare_data_for_t()

                    # training
                    if self.type_word_list!=None:
                        if self.Config.loss_func_gen == 'sig':
                            for word_type in center_nodes:
                                train_size = len(center_nodes[word_type])
                                start_list = list(range(0, train_size, self.Config.batch_size_tea))
                                np.random.shuffle(start_list)
                                for start in start_list:
                                    end = start + self.Config.batch_size_tea
                                    self.sess.run(self.generator.teacherforcing_updates,
                                                  feed_dict={self.generator.node_id: np.array(center_nodes[word_type][start:end]),
                                                             self.generator.node_neighbor_id: np.array(
                                                                 neighbor_nodes[word_type][start:end])})
                                if self.Config.f_debug:
                                    t_loss, t_score, t_norm, t_l2_loss = self.sess.run(
                                        [self.generator.loss_teacherforcing, self.generator.score, self.generator.norm,
                                         self.generator.l2_loss],
                                        feed_dict={self.generator.node_id: np.array(center_nodes[word_type]),
                                                   self.generator.node_neighbor_id: np.array(neighbor_nodes[word_type])}
                                    )
                                    print(
                                        '\ntea loss: {}, tea target loss: {}, tea l2 loss:{}, tea norm mean: {}, tea score max: {}, tea score min: {}\n'.format(
                                            t_loss, t_loss - t_l2_loss, t_l2_loss, np.mean(t_norm), t_score.max(),
                                            t_score.min()))
                        else:
                            for word_type in center_nodes:
                                train_size = len(center_nodes[word_type])
                                start_list = list(range(0, train_size, self.Config.batch_size_tea))
                                np.random.shuffle(start_list)
                                for start in start_list:
                                    end = start + self.Config.batch_size_tea
                                    self.sess.run(self.generator.teacherforcing_updates_type[word_type],
                                                  feed_dict={self.generator.node_id: np.array(
                                                      center_nodes[word_type][start:end]),
                                                             self.generator.node_neighbor_id: np.array(
                                                                 new_neighbor_nodes[word_type][start:end]),
                                                  self.generator.type_word_input: self.type_word_array[word_type]})
                                if self.Config.f_debug:
                                    t_loss, t_score, t_norm, t_l2_loss = self.sess.run(
                                        [self.generator.loss_teacherforcing_type[word_type], self.generator.score, self.generator.norm,
                                         self.generator.l2_loss],
                                        feed_dict={self.generator.node_id: np.array(center_nodes[word_type]),
                                                   self.generator.node_neighbor_id: np.array(new_neighbor_nodes[word_type]),
                                                   self.generator.type_word_input: self.type_word_array[word_type]}
                                    )
                                    print(
                                        '\ntea loss: {}, tea target loss: {}, tea l2 loss:{}, tea norm mean: {}, tea score max: {}, tea score min: {}\n'.format(
                                            t_loss, t_loss - t_l2_loss, t_l2_loss, np.mean(t_norm), t_score.max(),
                                            t_score.min()))
                    else:
                        train_size = len(center_nodes)
                        start_list = list(range(0, train_size, self.Config.batch_size_tea))
                        np.random.shuffle(start_list)
                        for start in start_list:
                            end = start + self.Config.batch_size_tea
                            self.sess.run(self.generator.teacherforcing_updates,
                                          feed_dict={self.generator.node_id: np.array(center_nodes[start:end]),
                                                     self.generator.node_neighbor_id: np.array(neighbor_nodes[start:end])})
                        if self.Config.f_debug:
                            t_loss, t_score, t_norm, t_l2_loss = self.sess.run(
                                [self.generator.loss_teacherforcing, self.generator.score, self.generator.norm, self.generator.l2_loss],
                                feed_dict={self.generator.node_id: np.array(center_nodes),
                                           self.generator.node_neighbor_id: np.array(neighbor_nodes)}
                                           )
                            print(
                                '\ntea loss: {}, tea target loss: {}, tea l2 loss:{}, tea norm mean: {}, tea score max: {}, tea score min: {}\n'.format(
                                    t_loss, t_loss - t_l2_loss, t_l2_loss, np.mean(t_norm), t_score.max(), t_score.min()))

            results = self.evaluation(epoch+1)
            if results:
                if results[0]>=result_max:
                    result_max = results[0]
                    self.write_embeddings_to_file()
                    embedding_matrix_g = self.sess.run(self.generator.embedding_matrix)
                    embedding_matrix_d = self.sess.run(self.discriminator.embedding_matrix)
                    self.saver.save(self.sess, self.log_dir + "model.checkpoint", global_step=epoch)

                delta = results[0]-result_old
                result_old = results[0]
                if delta < self.Config.convergence_threshold or results[0]<result_max:
                    count+=1
                    if count >= self.Config.attempt_times:
                        break
                else:
                    count = 0
            for nei_i in range(self.num_neighborhood_type):
                self.neighborhood_index_set[nei_i] += 1
                if self.neighborhood_index_set[nei_i] == self.neighborhood_repeat_set[nei_i]:
                    self.neighborhood_index_set[nei_i] = 0
        tqdm.tqdm.write("training completes")
        tf.reset_default_graph()
        return embedding_matrix_g, embedding_matrix_d


    def write_embeddings_to_file(self):
        """write embeddings of the generator and the discriminator to files"""

        embedding_matrix = self.sess.run(self.generator.embedding_matrix)
        Read_Write_Embeddings.write_embeddings_to_file(embedding_matrix, self.debug_path + 'gen' + '-' + self.Config.tt)
        embedding_matrix = self.sess.run(self.discriminator.embedding_matrix)
        Read_Write_Embeddings.write_embeddings_to_file(embedding_matrix, self.debug_path + 'dis' + '-' + self.Config.tt)


    def evaluation(self, epoch):
        modes = [self.generator, self.discriminator]
        model_name = ["gen","dis"]
        results_lp = []
        results_cl = []
        results = []
        if not self.link_samples == None:
            for i in range(2):
                embedding_matrix = self.sess.run(modes[i].embedding_matrix)
                roc_lp,acc_lp,macro_f1_lp,micro_f1_lp = Evaluation.link_prediction_roc_acc_f1(self.link_samples, embedding_matrix)
                results_lp.append('epoch: ' + str(epoch) + ': ' + model_name[i] + ":" + str(roc_lp) + " | " + str(acc_lp) + " | " + str(macro_f1_lp) + " | " + str(micro_f1_lp) + "\n")
                results.append(roc_lp)
            with open(self.debug_path+'lp'+'-'+self.Config.tt, mode="a+") as f:
                f.writelines(results_lp)
        if not self.label_list == None:
            for i in range(2):
                embedding_matrix = self.sess.run(modes[i].embedding_matrix)
                acc_cl, macro_f1_cl, micro_f1_cl = Evaluation.classification(self.label_list, embedding_matrix, self.Config.train_ratio, self.Config.classifier, self.Config.max_iter)
                results_cl.append('epoch: ' + str(epoch) + ': ' + model_name[i] + ":" + str(acc_cl) + " | " + str(macro_f1_cl) + " | " + str(micro_f1_cl) + "\n")
                results.append(macro_f1_cl)
            with open(self.debug_path+'cl'+'-'+self.Config.tt, mode="a+") as f:
                f.writelines(results_cl)
        if self.link_samples == None and self.label_list == None:
            embedding_matrix = self.sess.run(self.generator.embedding_matrix)
            results = Evaluation.compute_roc_of_proximity(self.node_pair_prox_array, embedding_matrix)
            with open(self.debug_path + '1st' + '-' + self.Config.tt, mode="a+") as f:
                f.writelines('epoch: ' + str(epoch) + ': ' + str(results[0]) + "\n")
        return results


    def sample_softmax(self, root, sample_num, for_d, nei=None):
        score_distribution = self.sess.run(self.generator.root_score,
                                           feed_dict={self.generator.root_set: np.array([root])})[0,:]
        score_distribution = softmax(score_distribution)

        score_distribution[root] = 0
        if for_d:
            score_distribution[nei] = 0
            if np.sum(score_distribution) == 0:
                return None

        score_distribution = score_distribution / np.sum(score_distribution)
        if not for_d and nei and nei!=[root]:
            sum_nei = np.sum(score_distribution[nei])
            if sum_nei == 0:
                print(str(root)+'\n')
                for n in nei:
                    print(str(n)+'\n')
                return None
            if sum_nei<self.Config.neighborhood_probability_ratio:
                score_distribution[nei] = score_distribution[nei]*self.Config.neighborhood_probability_ratio*(1-sum_nei)/(sum_nei*(1-self.Config.neighborhood_probability_ratio))
                score_distribution = score_distribution*(1-self.Config.neighborhood_probability_ratio)/(1-sum_nei)
                if np.sum(score_distribution)!=1:
                    score_distribution = score_distribution/np.sum(score_distribution)
        samples = np.random.choice(self.n_node, size=sample_num, replace=True, p=score_distribution).tolist()
        return samples

    def prepare_data_for_d(self):
        """generate positive and negative samples for the discriminator"""

        center_nodes = []
        neighbor_nodes = []
        labels = []

        if self.Config.flag_center_nodes_based_on_frequency:
            root_nodes = np.random.choice(self.root_nodes, size=self.n_node, replace=True,
                                          p=np.array(self.unigrams) / np.sum(self.unigrams))
        else:
            root_nodes = self.root_nodes
        for nei_i in range(self.num_neighborhood_type):
            neighborhood_index = self.neighborhood_index_set[nei_i]
            for i in root_nodes:
                if np.random.rand() < self.Config.update_ratio:
                    count = list(self.neighborhood[nei_i][neighborhood_index][i].values())
                    nei = list(self.neighborhood[nei_i][neighborhood_index][i].keys())
                    if nei:
                        norm_count = count / np.sum(count)
                        pos = np.random.choice(nei, size=self.Config.n_sample_dis[nei_i], replace=True, p=norm_count).tolist()
                    else:
                        pos = []

                    neg = self.sample_softmax(i,self.Config.n_sample_dis[nei_i],True,nei)

                    if len(pos) != 0 and neg is not None:
                        # positive samples
                        center_nodes.extend([i] * len(pos))
                        neighbor_nodes.extend(pos)
                        labels.extend([1] * len(pos))

                        # negative samples
                        center_nodes.extend([i] * len(neg))
                        neighbor_nodes.extend(neg)
                        labels.extend([0] * len(neg))

        return center_nodes, neighbor_nodes, labels


    def prepare_data_for_g(self):
        """sample nodes for the generator"""
        node_1 = []
        node_2 = []
        # multiprocessing
        if self.Config.flag_center_nodes_based_on_frequency:
            root_nodes = np.random.choice(self.root_nodes, size=self.n_node, replace=True,
                                          p=np.array(self.unigrams) / np.sum(self.unigrams))
        else:
            root_nodes = self.root_nodes
        for nei_i in range(self.num_neighborhood_type):
            neighborhood_index = self.neighborhood_index_set[nei_i]
            for i in root_nodes:
                neighbors = list(self.neighborhood[nei_i][neighborhood_index][i].keys())
                generate_sample = self.sample_softmax(i, self.Config.n_sample_gen[nei_i], False, nei=neighbors)
                if generate_sample!=None:
                    node_1.extend([i]*self.Config.n_sample_gen[nei_i])
                    node_2.extend(generate_sample)
        reward = self.sess.run(self.discriminator.reward,
                               feed_dict={self.discriminator.node_id: np.array(node_1),
                                          self.discriminator.node_neighbor_id: np.array(node_2)})
        return node_1, node_2, reward


    def prepare_data_for_t(self):
        center_nodes = []
        neighbor_nodes = []
        if self.Config.flag_center_nodes_based_on_frequency:
            root_nodes = np.random.choice(self.root_nodes,size=self.n_node,replace=True,p=np.array(self.unigrams)/np.sum(self.unigrams))
        else:
            root_nodes = self.root_nodes
        for nei_i in range(self.num_neighborhood_type):
            neighborhood_index = self.neighborhood_index_set[nei_i]
            for i in root_nodes:
                if np.random.rand() < self.Config.update_ratio:
                    count = list(self.neighborhood[nei_i][neighborhood_index][i].values())
                    nei = list(self.neighborhood[nei_i][neighborhood_index][i].keys())

                    if nei:
                        norm_count = count/np.sum(count)
                        pos = np.random.choice(nei, size=self.Config.n_sample_tea[nei_i], replace=True, p=norm_count).tolist()
                    else:
                        pos = []
                    if len(pos) != 0:
                        # positive samples
                        center_nodes.extend([i] * len(pos))
                        neighbor_nodes.extend(pos)

        return center_nodes, neighbor_nodes


    def sample_softmax_hete(self, root, sample_num, for_d, word_type, nei=None):
        # score_distribution = self.all_score[root,:].copy()
        score_distribution = self.sess.run(self.generator.root_score,
                                           feed_dict={self.generator.root_set: np.array([root])})[0, :]
        score_distribution = softmax(score_distribution)
        score_distribution[root] = 0
        for label_type in self.type_word_list:
            if label_type == word_type:
                continue
            score_distribution[self.type_word_list[label_type]] = 0
        if for_d and nei!=None:
            score_distribution[nei] = 0
        if np.sum(score_distribution) == 0:
            return None
        score_distribution = score_distribution / np.sum(score_distribution)
        if not for_d and nei and nei!=[root]:
            sum_nei = np.sum(score_distribution[nei])
            if sum_nei == 0:
                print(str(root)+'\n')
                for n in nei:
                    print(str(n)+'\n')
                return None
            if sum_nei<self.Config.neighborhood_probability_ratio:
                score_distribution[nei] = score_distribution[nei]*self.Config.neighborhood_probability_ratio*(1-sum_nei)/(sum_nei*(1-self.Config.neighborhood_probability_ratio))
                score_distribution = score_distribution*(1-self.Config.neighborhood_probability_ratio)/(1-sum_nei)
                if np.sum(score_distribution)!=1:
                    score_distribution = score_distribution/np.sum(score_distribution)
        samples = np.random.choice(self.n_node, size=sample_num, replace=True, p=score_distribution).tolist()
        return samples


    def prepare_data_for_d_hete(self):
        """generate positive and negative samples for the discriminator"""

        center_nodes = []
        neighbor_nodes = []
        labels = []

        for nei_i in range(self.num_neighborhood_type):
            center_set = []
            if self.center_type_set[nei_i] == 'all':
                center_set = self.root_nodes
            else:
                for tp in self.center_type_set[nei_i]:
                    center_set.extend(self.type_word_list[tp])
            if self.Config.flag_center_nodes_based_on_frequency:
                root_nodes = np.random.choice(center_set, size=len(center_set), replace=True,
                                              p=np.array(self.unigrams) / np.sum(self.unigrams))
            else:
                root_nodes = center_set
            neighborhood_index = self.neighborhood_index_set[nei_i]
            num_type = len(self.neighborhood[nei_i][neighborhood_index])
            for i in root_nodes:
                if np.random.rand() < self.Config.update_ratio:
                    for word_type in self.neighborhood[nei_i][neighborhood_index]:
                        count = list(self.neighborhood[nei_i][neighborhood_index][word_type][i].values())
                        nei = list(self.neighborhood[nei_i][neighborhood_index][word_type][i].keys())

                        if nei:
                            norm_count = count/np.sum(count)
                            pos = np.random.choice(nei, size=math.ceil(self.Config.n_sample_dis[nei_i]/num_type), replace=True, p=norm_count).tolist()
                        else:
                            pos = []

                        neg = self.sample_softmax_hete(i,math.ceil(self.Config.n_sample_dis[nei_i]/num_type),True,word_type,nei)
                        if len(pos) != 0 and neg is not None:
                            # positive samples
                            center_nodes.extend([i] * len(pos))
                            neighbor_nodes.extend(pos)
                            labels.extend([1] * len(pos))

                            # negative samples
                            center_nodes.extend([i] * len(neg))
                            neighbor_nodes.extend(neg)
                            labels.extend([0] * len(neg))

        return center_nodes, neighbor_nodes, labels


    def prepare_data_for_g_hete(self):
        """sample nodes for the generator"""
        node_1 = {}
        node_2 = {}
        reward = {}
        for nei_i in range(self.num_neighborhood_type):
            if self.node_type_set[nei_i] == 'all':
                for word_type in self.type_word_list:
                    node_1[word_type] = []
                    node_2[word_type] = []
                    reward[word_type] = []
            else:
                for word_type in self.node_type_set[nei_i]:
                    node_1[word_type] = []
                    node_2[word_type] = []
                    reward[word_type] = []
        # multiprocessing

        for nei_i in range(self.num_neighborhood_type):
            center_set = []
            if self.center_type_set[nei_i] == 'all':
                center_set = self.root_nodes
            else:
                for tp in self.center_type_set[nei_i]:
                    center_set.extend(self.type_word_list[tp])
            if self.Config.flag_center_nodes_based_on_frequency:
                root_nodes = np.random.choice(center_set, size=len(center_set), replace=True,
                                              p=np.array(self.unigrams) / np.sum(self.unigrams))
            else:
                root_nodes = center_set
            neighborhood_index = self.neighborhood_index_set[nei_i]
            num_type = len(self.neighborhood[nei_i][neighborhood_index])
            for i in root_nodes:
                if np.random.rand() < self.Config.update_ratio:
                    for word_type in self.neighborhood[nei_i][neighborhood_index]:
                        neighbors = list(self.neighborhood[nei_i][neighborhood_index][word_type][i].keys())
                        generate_sample = self.sample_softmax_hete(i, math.ceil(self.Config.n_sample_gen[nei_i]/num_type), False, word_type, nei=neighbors)
                        if generate_sample != None:
                            node_1[word_type].extend([i]*len(generate_sample))
                            node_2[word_type].extend(generate_sample)
        for word_type in node_2:
            reward[word_type] = self.sess.run(self.discriminator.reward,
                                   feed_dict={self.discriminator.node_id: np.array(node_1[word_type]),
                                              self.discriminator.node_neighbor_id: np.array(node_2[word_type])})
        return node_1, node_2, reward


    def prepare_data_for_t_hete(self):
        center_nodes = {}
        neighbor_nodes = {}
        for nei_i in range(self.num_neighborhood_type):
            if self.node_type_set[nei_i] == 'all':
                for word_type in self.type_word_list:
                    center_nodes[word_type] = []
                    neighbor_nodes[word_type] = []
            else:
                for word_type in self.node_type_set[nei_i]:
                    center_nodes[word_type] = []
                    neighbor_nodes[word_type] = []
        for nei_i in range(self.num_neighborhood_type):
            center_set = []
            if self.center_type_set[nei_i] == 'all':
                center_set = self.root_nodes
            else:
                for tp in self.center_type_set[nei_i]:
                    center_set.extend(self.type_word_list[tp])
            if self.Config.flag_center_nodes_based_on_frequency:
                root_nodes = np.random.choice(center_set, size=len(center_set), replace=True,
                                              p=np.array(self.unigrams) / np.sum(self.unigrams))
            else:
                root_nodes = center_set
            neighborhood_index = self.neighborhood_index_set[nei_i]
            num_type = len(self.neighborhood[nei_i][neighborhood_index])
            for i in root_nodes:
                if np.random.rand() < self.Config.update_ratio:
                    for word_type in self.neighborhood[nei_i][neighborhood_index]:
                        count = list(self.neighborhood[nei_i][neighborhood_index][word_type][i].values())
                        nei = list(self.neighborhood[nei_i][neighborhood_index][word_type][i].keys())

                        if nei:
                            norm_count = count/np.sum(count)
                            pos = np.random.choice(nei, size=math.ceil(self.Config.n_sample_tea[nei_i]/num_type), replace=True, p=norm_count).tolist()
                        else:
                            pos = []
                        if len(pos) != 0:
                            # positive samples
                            center_nodes[word_type].extend([i] * len(pos))
                            neighbor_nodes[word_type].extend(pos)

        return center_nodes, neighbor_nodes


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for computation stability
    return e_x / e_x.sum()


