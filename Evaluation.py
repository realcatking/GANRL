'''
Evaluation method for network embedding
'''

import math
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing



def classification(label, embeddings, train_ratio = 0.75, classifierStr ='LinearSVM', max_iter = 1000, repeat_time = 10):
    data_for_class = []
    label_list = []
    for i in label:
        data_for_class.append(embeddings[i[0]])
        label_list.append(i[1])
    sum_acc = sum_ma_f1 = sum_mi_f1 = 0
    for i in range(repeat_time):
        train_data, test_data, train_label, test_label = train_test_split(data_for_class,label_list,train_size=train_ratio)
        acc, macro_f1, micro_f1 = _classifiction(train_data,test_data,train_label,test_label,classifierStr,max_iter)
        sum_acc +=acc
        sum_ma_f1 += macro_f1
        sum_mi_f1 += micro_f1
    mean_acc = sum_acc/repeat_time
    mean_macro_f1 = sum_ma_f1/repeat_time
    mean_micro_f1 = sum_mi_f1/repeat_time
    print('\nClassification Accuracy=%f, macro_f1=%f, micro_f1=%f\n' % (mean_acc, mean_macro_f1, mean_micro_f1))
    return mean_acc, mean_macro_f1, mean_micro_f1


def _classifiction(train_vec, test_vec, train_y, test_y, classifierStr='LinearSVM', max_iter = 1000, normalize=False):
    if classifierStr == 'LinearSVM':
        classifier = LinearSVC(max_iter=max_iter)
    elif classifierStr == 'LR':
        classifier = OneVsRestClassifier(LogisticRegression(max_iter=max_iter, solver='liblinear'))


    if normalize:
        print('Normalize data')
        allvec = list(train_vec)
        allvec.extend(test_vec)
        allvec_normalized = preprocessing.normalize(allvec, norm='l2', axis=1)
        train_vec = allvec_normalized[0:len(train_y)]
        test_vec = allvec_normalized[len(train_y):]


    classifier.fit(train_vec, train_y)
    y_pred = classifier.predict(test_vec)
    cm = confusion_matrix(test_y, y_pred)
    acc = accuracy_score(test_y, y_pred)

    macro_f1 = f1_score(test_y, y_pred, average='macro')
    micro_f1 = f1_score(test_y, y_pred, average='micro')

    per = len(train_y) * 1.0 /(len(test_y)+len(train_y))

    return acc, macro_f1, micro_f1


def cosine_similarity(a,b):
    if a.ndim == 1:
        return np.dot(a,b) / (norm(a) * norm(b))
    else:
        inner_sim = np.dot(a,b.T)
        diag_inverse_sqrt = np.diag(1/np.sqrt(np.diag(inner_sim)))
        return np.dot(np.dot(diag_inverse_sqrt,inner_sim),diag_inverse_sqrt)


def norm(a):
    return math.sqrt(np.dot(a,a))


def inner_similarity(a,b):
    return np.dot(a,b.T)


def link_prediction_roc_acc_f1(link_samples, embeddings):

    y_true = np.array([link_samples[i][2] for i in range(len(link_samples))])
    y_score = np.array([inner_similarity(embeddings[link_samples[i][0], :], embeddings[link_samples[i][1], :]) for i in
                 range(len(link_samples))])
    y_pred = np.zeros(len(y_score))
    median = np.median(y_score)
    index_pos = y_score >= median
    index_neg = y_score < median
    y_pred[index_pos] = 1
    y_pred[index_neg] = 0
    roc = roc_auc_score(y_true, y_score)
    accuracy = accuracy_score(y_true,y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    if roc < 0.5:
        roc = 1 - roc
    print('\nThe ROC, Accuracy, Macro-F1, Micro-F1 of the link prediction is {}, {}, {}, {}\n'.format(roc, accuracy, macro_f1, micro_f1))

    return roc, accuracy, macro_f1, micro_f1


def evaluation(embedding_matrix, result_path, tt, link_samples=None, label_list=None, train_ratio=None, classifier=None, max_iter=None):
    results_lp = []
    results_cl = []
    if not link_samples == None:
        roc_lp, acc_lp, macro_f1_lp, micro_f1_lp = link_prediction_roc_acc_f1(link_samples, embedding_matrix)
        results_lp.append(str(roc_lp) + " | " + str(acc_lp) + " | " + str(macro_f1_lp) + " | " + str(micro_f1_lp) + "\n")
        print('link prediction: ' + str(roc_lp) + " | " + str(acc_lp) + " | " + str(macro_f1_lp) + " | " + str(micro_f1_lp) + "\n")
        with open(result_path + 'lp' + '-' + tt, mode="a+") as f:
            f.writelines(results_lp)
    if not label_list == None:
        acc_cl, macro_f1_cl, micro_f1_cl = classification(label_list, embedding_matrix, train_ratio, classifier, max_iter)
        results_cl.append("train ratio: " + str(train_ratio) + " : " + str(acc_cl) + " | " + str(macro_f1_cl) + " | " + str(micro_f1_cl) + "\n")
        print('classification: ' + "train ratio: " + str(train_ratio) + " : " + str(acc_cl) + " | " + str(macro_f1_cl) + " | " + str(micro_f1_cl) + "\n")
        with open(result_path + 'cl' + '-' + str(train_ratio) + '-' + tt, mode="a+") as f:
            f.writelines(results_cl)


def evaluation_set(embedding_matrix, result_path, tt, link_samples=None, label_list=None, train_ratio_set=None, classifier=None, max_iter=None):
    results_lp = []
    results_cl = []
    if not link_samples == None:
        roc_lp, acc_lp, macro_f1_lp, micro_f1_lp = link_prediction_roc_acc_f1(link_samples, embedding_matrix)
        results_lp.append(str(roc_lp) + " | " + str(acc_lp) + " | " + str(macro_f1_lp) + " | " + str(micro_f1_lp) + "\n")
        print('link prediction: ' + str(roc_lp) + " | " + str(acc_lp) + " | " + str(macro_f1_lp) + " | " + str(micro_f1_lp) + "\n")
        with open(result_path + 'lp' + '-' + tt, mode="a+") as f:
            f.writelines(results_lp)
    elif not label_list == None:
        for train_ratio in train_ratio_set:
            acc_cl, macro_f1_cl, micro_f1_cl = classification(label_list, embedding_matrix, train_ratio, classifier, max_iter)
            results_cl.append("train ratio: " + str(train_ratio) + " : " + str(acc_cl) + " | " + str(macro_f1_cl) + " | " + str(micro_f1_cl) + "\n")
            print('classification: ' + "train ratio: " + str(train_ratio) + " : " + str(acc_cl) + " | " + str(macro_f1_cl) + " | " + str(micro_f1_cl) + "\n")
        with open(result_path + 'cl' + '-' + tt, mode="a+") as f:
            f.writelines(results_cl)


def compute_roc_of_proximity(proximity_label, embeddings):
    num_prox = proximity_label.shape[1]-2
    num_pairs = proximity_label.shape[0]
    y_score = np.array([inner_similarity(embeddings[proximity_label[i,0], :], embeddings[proximity_label[i,1], :]) for i in
                        range(num_pairs)])
    roc_set = []
    for i_p in range(num_prox):
        y_true = proximity_label[:,i_p+2]
        roc = roc_auc_score(y_true, y_score)
        if roc < 0.5:
            roc = 1 - roc
        roc_set.append(roc)

        print('\nThe ROC of {} is {}\n'.format(i_p, roc))

    return roc_set