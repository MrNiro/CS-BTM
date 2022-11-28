# -*- coding: utf-8 -*-
import numpy as np
import indexDocs
from scipy import spatial

from BitermEncoder import BitermBert
from doc import Doc
from sampler import *
import os
import math


class Model:
    def __init__(self, k, alpha, beta, n_iter, save_step, has_b=False):
        self.K = k                      # number of topics
        self.vocabulary_size = None     # vocabulary size

        self.alpha = alpha          # hyper-parameters of p(z)
        self.beta = beta            # hyper-parameters of p(w|z)

        self.pw_b = None            # the background word distribution
        self.nw_z = None            # n(w,z), size K*W 各单词被分配到主题z上的次数.
        self.nb_z = np.zeros(k)     # n(b|z), size K*1 各biterm被分配到主题z上的次数,在论文中是用来计算Nz的
        self.bs = list()            # list of all biterms

        self.pz = None              # the probability proportion of K topics
        self.pw_z = None            # the probability proportion of each word in each topic
        self.topic_words = list()   # words index for each topic, sorted by word frequency within each topic

        self.indexToWord = None
        self.indexToVector = None
        self.biterm_encoder = None
        self.sim_thresh = 0.9
        self.Sim = dict()

        # If true, the topic 0 is set to a background topic that equals to the empirical word distribution.
        # It can be used to filter out common words
        self.has_background = False

        self.n_iter = n_iter
        self.save_step = save_step
        self.has_background = has_b

    def train(self, doc_path, output_dir):
        """
        @description: 生成模型运行函数，狄利克雷-多项 共轭分布，Gibbs采样
        @param {type}
        @return:
        """
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            os.mkdir(output_dir + "/model")
        vocabulary_path = output_dir + "/vocabulary.txt"
        index_docs = self.load_docs(doc_path, vocabulary_path)
        self.model_init(index_docs)

        print("Begin iteration")
        model_dir = output_dir + "/model/k" + str(self.K) + "."
        for i in range(1, self.n_iter + 1):
            print("\riter " + str(i) + "/" + str(self.n_iter), end='...')
            for b in range(len(self.bs)):
                # 根据每个biterm更新文章中的参数
                # 计算核心代码，self.bs中保存的是词对的biterm
                self.update_biterm(self.bs[b])
            if i % self.save_step == 0:
                self.save_model(model_dir)

        self.save_model(model_dir)

    def infer(self, doc_path, model_dir, vocabulary_path):
        index_docs = self.load_docs(doc_path, vocabulary_path, if_load_voc=True)
        if self.pz is None and self.pw_z is None:
            self.load_model(model_dir)
        if len(self.bs) == 0:
            self.model_init(index_docs)

        self.indexToWord = sorted(index_docs.wordToIndex.keys(), key=lambda x: index_docs.wordToIndex[x])
        # topic_words_with_frequency = sorted(list(zip(indexToWord, self.pw_b)), key=lambda x: x[1])

        self.load_index_to_vector("../../data/word_embeddings.txt")

        # print("Start Encoding texts...")
        # for i, each in enumerate(self.indexToWord):
        #     print("\rEncoding %d / %d " % (i, len(index_docs.wordToIndex)), end="...")
        #     self.indexToVector.append(self.biterm_encoder.encode(each))
        # print("Encoding Done!")

        topic_dict = {}
        topic_dict_index = {}
        cur_id = 0
        for doc_idx, each in enumerate(index_docs.docIndex):
            pz_d = np.zeros(self.K)  # the probability proportion of the Doc in each Topic

            d = Doc(each)
            biterms = []
            d.gen_biterms(biterms, cur_id)
            pb_d = self.compute_pb_d(biterms)

            for idx, bi in enumerate(biterms):
                # calculate pz_d via probability proportion of each biterm
                pz_b = self.compute_pz_b(bi)

                for i in range(self.K):
                    # BTM论文中应为Sum(p(z|b) * p(b|d))，但由于p(b|d)近于均匀分布，因此这里不再进行计算
                    # 而CS-BTM利用Bert对p(b|d)的计算进行了改进
                    pz_d[i] += (pz_b[i] * pb_d[idx])

            pz_d = self.normalize_ndarray(pz_d)
            k = int(np.argmax(pz_d))            # 确定该doc所属的topic编号k

            sentence = list(map(lambda x: self.indexToWord[x], each))
            sentence = str(sentence).strip("[]").replace(",", "").replace("\'", "")

            if k in topic_dict_index:
                topic_dict[k].append(sentence)
                topic_dict_index[k].append(each)
            else:
                topic_dict[k] = [sentence]
                topic_dict_index[k] = [each]

            # print(pb_d)
            print("Doc: %d Topic: %d\t %s\n" % (doc_idx, k, sentence))

        self.generate_topic_words()
        print("\n\n======================== Topic Words ========================")
        for i in range(self.K):
            topic_words = list(map(lambda x: self.indexToWord[x], self.topic_words[i]))
            print("topic", i, topic_words[:5])
        # for t in range(5, 11):
        #     self.cal_coherence(t, topic_dict_index)

        return topic_dict

    def model_init(self, index_docs):
        """
        @description: 初始化模型的代码。
        @param :None
        @return: 初始化self.nv_z 和self.nwz，
        """
        self.pw_b = np.zeros(self.vocabulary_size)
        self.nw_z = np.zeros((self.K, self.vocabulary_size))

        for each in index_docs.docIndex:
            d = Doc(each)
            biterms = []
            d.gen_biterms(biterms, len(self.bs))
            # statistic the empirical word distribution
            for i in range(d.size()):
                w = d.get_w(i)
                self.pw_b[w] += 1  # 统计词频
            for b in biterms:
                # 此处直接添加当前句子产生的所有biterms
                # 导致最终会出现大量的重复biterms，这样没有问题的吗？
                # 噢没有问题！本来就应该遍历biterms的时候统计词频
                # 但在计算biterm相似度的时候，目前已b_id构建字典，仍会有大量的重复计算
                self.bs.append(b)  # self.bs中添加的是一个biterm类。类的内容是这段文本中所有可能的词的组合.

        # 做归一化处理,现在 pw_b中保存的是 词：词频率。
        self.pw_b = self.normalize_ndarray(self.pw_b)

        for biterm in self.bs:
            # k表示的是从0-K之间的随机数。用来将biterm随机分配给各个topic
            k = uni_sample(self.K)
            self.assign_biterm_topic(biterm, k)  # 入参是一个词对(biterm)和他对应的主题

    def load_model(self, model_dir):
        # load pz - the probability proportion of K topics
        pt = open(model_dir + "k" + str(self.K) + ".pz")
        if not pt:
            Exception("Model file not found!")

        for line in pt.readlines():
            info = map(lambda x: float(x), line.strip().split())
            self.pz = np.asarray(list(info))
        assert (abs(self.pz.sum() - 1) < 1e-4)

        # load pw_z - the probability proportion of each word in each topic
        pt_2 = open(model_dir + "k" + str(self.K) + ".pw_z")
        if not pt_2:
            Exception("Model file not found!")

        tmp = []
        for line in pt_2.readlines():
            info = map(lambda x: float(x), line.strip().split())
            tmp.append(list(info))
        self.pw_z = np.asarray(tmp)
        print("n(z)=%d, n(w)=%d\n" % (self.pw_z.shape[0], self.pw_z.shape[1]))
        assert (self.pw_z.shape[0] > 0 and abs(self.pw_z[0].sum() - 1) < 1e-4)

    def load_docs(self, doc_path, vocabulary_path, if_load_voc=False):
        """
        @description: 读取文档并做indexing，生成self.pw_b 和 self.bs
        """

        print("load docs: " + doc_path)

        index_docs = indexDocs.IndexDocs(if_load_voc=if_load_voc)
        self.vocabulary_size = index_docs.run_indexDocs(doc_path, vocabulary_path)

        return index_docs

    def load_index_to_vector(self, word_embeddings_path):
        embedding_file = open(word_embeddings_path)
        embeddings_dict = {}
        for line in embedding_file.readlines():
            info = line.strip().split(" ")
            word = info[0]
            embeddings = list(map(lambda x: float(x), info[1:]))
            embeddings = np.asarray(embeddings)
            embeddings_dict[word] = embeddings
        print("Embeddings loaded!")
        embedding_file.close()

        self.indexToVector = []
        self.biterm_encoder = BitermBert(model_name="sentence-transformers/all-MiniLM-L6-v2")
        for i, each in enumerate(self.indexToWord):
            if each in embeddings_dict:
                self.indexToVector.append(embeddings_dict[each])
            else:
                self.indexToVector.append(self.biterm_encoder.encode(each))
        print("Word index to Vector init done!")

    def generate_topic_words(self):
        for k in range(self.K):
            topic_words_index_with_frequency = sorted(list(zip(range(0, self.vocabulary_size), self.pw_z[k])),
                                                      key=lambda x: x[1], reverse=True)
            topic_words_index = list(map(lambda x: x[0], topic_words_index_with_frequency))
            self.topic_words.append(topic_words_index)

    def normalize_ndarray(self, array, smoother=0):
        t_sum = array.sum()

        array = (array + smoother) / (t_sum + self.K * smoother)
        return array

    def update_biterm(self, bi):
        self.reset_biterm_topic(bi)

        # compute p(z|b) in the paper
        pz_b = self.compute_pz_b(bi)

        # sample topic for biterm b
        k = mul_sample(pz_b)
        self.assign_biterm_topic(bi, k)  # 更新论文中的Nz,N_wiz,N_wjz.

    def reset_biterm_topic(self, bi):
        k = bi.z
        w1 = bi.wi
        w2 = bi.wj

        self.nb_z[k] -= 1
        self.nw_z[k][w1] -= 1
        self.nw_z[k][w2] -= 1
        assert (self.nb_z[k] > -10e-7 and self.nw_z[k][w1] > -10e-7 and self.nw_z[k][w2] > -10e-7)
        bi.reset_z()

    def assign_biterm_topic(self, bi, k):
        # bi是每一个词对，K是主题的个数。
        bi.set_z(k)
        w1 = bi.wi  # 词对中的一个词
        w2 = bi.wj  # 词对中的第二个词
        self.nb_z[k] += 1  # self.nb_z: 表示的是在那么多的词对中，每个主题出现的次数。
        self.nw_z[k][w1] += 1  # self.nwz[1][1] 表示的是在主题1中，1号单词出现的次数。
        self.nw_z[k][w2] += 1  # self.nwz[2][3] 表示的是在出题2中，2号单词出现的次数。

    def compute_pz_b(self, bi):
        pz_b = np.zeros(self.K)
        w1 = bi.wi  # 取到词对中的第一个词编号。
        w2 = bi.wj  # 取到词对中的第二个词编号。

        for k in range(self.K):
            if self.pz is None and self.pw_z is None:
                if self.has_background and k == 0:
                    pw1k = self.pw_b[w1]
                    pw2k = self.pw_b[w2]
                else:
                    pw1k = (self.nw_z[k][w1] + self.beta) / (2 * self.nb_z[k] + self.vocabulary_size * self.beta)
                    pw2k = (self.nw_z[k][w2] + self.beta) / (2 * self.nb_z[k] + 1 + self.vocabulary_size * self.beta)

                # len(self.bs)表示的是在文档中以后多少的词对
                pk = (self.nb_z[k] + self.alpha) / (len(self.bs) + self.K * self.alpha)
                pz_b[k] = pk * pw1k * pw2k
            else:
                pz_b[k] = self.pz[k] * self.pw_z[k][w1] * self.pw_z[k][w2]

        pz_b = self.normalize_ndarray(pz_b)
        return pz_b

    def compute_pb_d(self, biterms):
        bi_num = len(biterms)
        print("biterms num:", bi_num)

        lambda_b = [0 for _ in range(bi_num)]
        for i, bi in enumerate(biterms):
            # 按照论文，此处应该只和当前doc中的biterms比较
            bi_id = bi.b_id
            if bi_id not in self.Sim:
                self.Sim[bi_id] = {}
            for j, bi_2 in enumerate(biterms):
                # lines below are for local biterms
                if i == j:
                    continue
                bi_2_id = bi_2.b_id
                if bi_2_id not in self.Sim[bi_id]:
                    similarity = self.cal_biterm_sim(bi, bi_2)
                    self.Sim[bi_id][bi_2_id] = similarity
                else:
                    similarity = self.Sim[bi_id][bi_2_id]

                # if j < i:
                #     similarity = Sim[j][i]
                # else:
                #     similarity = self.cal_biterm_sim(bi, bi_2)
                #     Sim[i][j] = similarity
                # # lines below are for global biterms
                # if bi.equalTo(bi_2):
                #     continue
                # if self.Sim[bi.b_id][j] >= 0:
                #     similarity = self.Sim[bi.b_id][j]
                # else:
                #     similarity = self.cal_biterm_sim(bi, bi_2)
                #     self.Sim[bi.b_id][j] = similarity

                if similarity >= self.sim_thresh:
                    # print("similar biterms: (%s, %s) - (%s, %s)" %
                    #       (self.indexToWord[bi.wi], self.indexToWord[bi.wj],
                    #        self.indexToWord[bi_2.wi], self.indexToWord[bi_2.wj]))
                    lambda_b[i] += 1

        # print(lambda_b)
        sum_lambda_b = sum(lambda_b)
        pb_d = list(map(lambda x: (x + 1) / (sum_lambda_b + 1), lambda_b))
        return self.normalize_ndarray(np.asarray(pb_d))

    def cal_biterm_sim(self, bi_1, bi_2):
        # w1_1 = self.indexToWord[bi_1.wi]
        # w1_2 = self.indexToWord[bi_1.wj]
        # w2_1 = self.indexToWord[bi_2.wi]
        # w2_2 = self.indexToWord[bi_2.wj]
        # similarity = self.biterm_encoder.cal_biterm_similarity((w1_1, w1_2), (w2_1, w2_2))

        v1 = self.indexToVector[bi_1.wi]
        v2 = self.indexToVector[bi_1.wj]
        v3 = self.indexToVector[bi_2.wi]
        v4 = self.indexToVector[bi_2.wj]
        similarity = (spatial.distance.cosine(v1, v3) +
                      spatial.distance.cosine(v1, v4) +
                      spatial.distance.cosine(v2, v3) +
                      spatial.distance.cosine(v2, v4)) / 4
        # similarity = float(torch.cosine_similarity(v1, v3) +
        #                    torch.cosine_similarity(v1, v4) +
        #                    torch.cosine_similarity(v2, v3) +
        #                    torch.cosine_similarity(v2, v4)) / 4

        return similarity

    def cal_coherence(self, topic_word_num, topic_dict):
        print("\nEvaluating coherence score with %d topic words:" % topic_word_num)

        text = [d for docs in topic_dict.values() for d in docs]

        for k in range(self.K):
            if k not in topic_dict:
                continue

            topic_words_index = self.topic_words[k]
            score = 0
            for t in range(1, topic_word_num):
                for m in range(t):
                    D_vm = 0
                    D_vt_vm = 0
                    vm = topic_words_index[m]
                    vt = topic_words_index[t]
                    for doc in text:
                        if vm in doc:
                            D_vm += 1
                            if vt in doc:
                                D_vt_vm += 1
                    if D_vm == 0:
                        # 此处应该不能直接跳过，否则聚类效果不好的docs反而会使得分数较高
                        continue
                    score += math.log((D_vt_vm + 1) / D_vm)

            print("\tTopic: %d\tCoherence score: %f" % (k, score))

    def save_model(self, output_dir):
        pt = output_dir + "pz"
        print("\nwrite p(z): " + pt)
        self.save_pz(pt)

        pt2 = output_dir + "pw_z"
        print("write p(w|z): " + pt2)
        self.save_pw_z(pt2)

    # p(z) is determined by the overall proportions of biterms in it
    def save_pz(self, pt):
        self.pz = np.asarray(self.nb_z)
        self.pz = self.normalize_ndarray(self.pz, self.alpha)

        wf = open(pt, 'w')
        wf.write(str(self.pz.tolist()).strip("[]").replace(",", ""))

    def save_pw_z(self, pt):
        self.pw_z = np.ones((self.K, self.vocabulary_size))  # 生成5行2700列的矩阵。用来保存每个主题中，各个单词出现的概率。
        wf = open(pt, 'w')
        for k in range(self.K):
            for w in range(self.vocabulary_size):
                # 计算每个词在这个主题中出现的概率。
                self.pw_z[k][w] = (self.nw_z[k][w] + self.beta) / (self.nb_z[k] * 2 + self.vocabulary_size * self.beta)
                wf.write(str(self.pw_z[k][w]) + ' ')
            wf.write("\n")
