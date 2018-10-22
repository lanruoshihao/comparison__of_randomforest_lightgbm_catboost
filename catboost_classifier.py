#!usr/bin/python
# -*- coding:utf-8 -*-

import os
import re
import numpy as np
import jieba
import operator
import codecs
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures


def read_file_name(file_name):
    # print("file_name:", file_name)
    open_file = codecs.open(file_name, encoding='gb18030', errors='ignore')
    f_content = open_file.read()
    return f_content


def read_all_file(dir_path):
    neg_dir = os.path.join(dir_path, 'neg')
    pos_dir = os.path.join(dir_path, 'pos')
    all_file = []
    label = []
    for pos, folder, files in os.walk(pos_dir):
        for file_name in files:
            file_name = os.path.join(pos_dir, file_name)
            f_content = read_file_name(file_name)
            all_file.append(f_content)
            label.append(1)

    for neg, folder, files in os.walk(neg_dir):
        for file_name in files:
            file_name = os.path.join(neg_dir, file_name)
            f_content = read_file_name(file_name)
            all_file.append(f_content)
            label.append(0)
    return all_file, label


def construct_word_list(file_content, fstop):
    pattern = re.compile(u"[\d\s+\.\!\/_,$%^*()?;；。：、:-【】+\"\']+|[+——！，;:。：？、~@#￥%……&* -]+|[()（）～]+")
    file_content = re.sub(pattern, '', file_content)
    word_cut = jieba.lcut(file_content)
    word_list = [word for word in word_cut if word not in fstop and len(word) >= 2]
    return word_list


def preprocess_data(file_list, label, fstop, flag=0):
    #标识flag=0表示的是训练集,flag=1表示的是测试集
    all_word_list = []
    pos_word = []
    neg_word = []
    for js in range(len(file_list)):
        word_list = construct_word_list(file_list[js], fstop)
        all_word_list.append(word_list)
        if label[js] == 1:
            pos_word.extend(word_list)
        elif label[js] == 0:
            neg_word.extend(word_list)

    if flag == 0:
        return all_word_list, label, pos_word, neg_word
    else:
        return all_word_list, label


def create_word_scores(pos_word, neg_word):
    '''
    计算每个特征的卡方统计量大小
    :param pos_word:
    :param neg_word:
    :return:
    '''
    # 读取正样本
    posWords = list(pos_word)

    # 读取负样本，其它类别
    negWords = list(neg_word)

    word_fd = FreqDist()  # 可统计所有词的词频
    cond_word_fd = ConditionalFreqDist()  # 可统计积极文本中的词频和消极文本中的词频
    for word in posWords:
        word_fd[word] += 1
        cond_word_fd[u'1'][word] += 1

    for word in negWords:
        word_fd[word] += 1
        cond_word_fd[u'0'][word] += 1

    pos_word_count = cond_word_fd[u'1'].N()  # 积极词的数量

    neg_word_count = cond_word_fd[u'0'].N()  # 消极词的数量

    total_word_count = pos_word_count + neg_word_count

    word_scores = {}

    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd[u'1'][word], (freq, pos_word_count),
                                                total_word_count)  # 计算积极词的卡方统计量，这里也可以计算互信息等其它统计量

        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd[u'0'][word], (freq, neg_word_count),
                                               total_word_count)  # 同理
        word_scores[word] = pos_score + neg_score  # 一个词的信息量等于积极卡方统计量加上消极卡方统计量
    return word_scores


def find_best_words(word_scores, number):
    '''
    将特征按照卡方统计量从大到小排序
    :param word_scores:
    :param number:
    :return:
    '''
    best_vals = sorted(word_scores.items(), key=operator.itemgetter(1), reverse=True)[
                :number]  # 把词按信息量倒序排序。number是特征的维度
    best_words = list([w for w, s in best_vals])
    return best_words


def text_word_frequency(all_word_list, data_dict, label):
    '''
    求出样本的特征的词频
    :param all_word_list:
    :return:
    '''
    all_word_freq = []
    for index in range(len(all_word_list)):
        word_frequency = []
        for word in data_dict:
            word_count = all_word_list[index].count(word)
            word_frequency.append(word_count)
        all_word_freq.append(word_frequency)
    return all_word_freq, label


def construct_test_list(test_file, test_label, fstop):
    '''
    切分测试文本，并提取出关键词
    :param test_file:
    :return:
    '''
    data_list = jieba.lcut(test_file)
    test_list = [word for word in data_list if word not in fstop and len(word) >= 2]
    return test_list, test_label


def construct_verify_list(verify_file, fstop):
    '''
    处理验证集,并提取出其中的关键
    :param verify_file:
    :param fstop:
    :return:
    '''
    f_content = read_file_name(verify_file)
    data_list = jieba.lcut(f_content)
    verify_list = [word for word in data_list if word not in fstop and len(word) >= 2]
    return verify_list


if __name__ == '__main__':
    dir_path = 'senti_analysis-master/data/ChnSentiCorp_htl_ba_2000'
    fstop_path = 'senti_analysis-master/data/stopWord.txt'
    fstop = codecs.open(fstop_path, encoding='utf-8').read()

    all_file, data_label = read_all_file(dir_path)

    accuray_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    number = 1600   #特征的维数
    skf = StratifiedKFold(n_splits=5)    #5折交叉验证
    for train_index, test_index in skf.split(all_file, data_label):
        train_data, test_data = np.array(all_file)[train_index], np.array(all_file)[test_index]
        train_label, test_label = np.array(data_label)[train_index], np.array(data_label)[test_index]

        train_word_list, train_label, pos_word, neg_word = preprocess_data(train_data, train_label, fstop)
        word_scores = create_word_scores(pos_word, neg_word)
        best_words = find_best_words(word_scores, number)

        train_word_freq, train_label = text_word_frequency(train_word_list,  best_words, train_label)

        # clf = CatBoostClassifier(n_estimators=39, iterations=5, num_boost_round=50, learning_rate=0.2, max_bin=200)
        clf = CatBoostClassifier(n_estimators=99, learning_rate=0.1, max_bin=200)
        clf.fit(pd.DataFrame(train_word_freq), train_label)

        test_word_list, test_label = preprocess_data(test_data, test_label, fstop, flag=1)
        test_word_freq, test_label = text_word_frequency(test_word_list, best_words, test_label)

        predict_label = clf.predict(test_word_freq)

        accuracy = metrics.accuracy_score(test_label, predict_label)
        precision = metrics.precision_score(test_label, predict_label)
        recall = metrics.recall_score(test_label, predict_label)
        f1_score = metrics.f1_score(test_label, predict_label)

        accuray_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)

    print u'平均分类准确率：%f'%np.mean(accuray_list)
    print u'平均分类精确率：%f' % np.mean(precision_list)
    print u'平均分类召回率：%f' % np.mean(recall_list)
    print u'平均分类f1_score：%f' % np.mean(f1_score_list)
    print accuray_list
    print precision_list
    print recall_list
    print f1_score_list

