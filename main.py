from Instance import Inst
from example import Exam
from AlphaBet import AlphaBet
from hyperparameter import Hyperparameter
from CNN_model import CNN_text
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import cProfile

from torch.autograd import Variable

class AV_Perceptron:
    def __init__(self):
        self.word_AlphaBet = AlphaBet()
        self.tag_AlphaBet = AlphaBet()
        self.hyperpara = Hyperparameter()

    def read_file(self, path):
        f = open(path)
        W = []
        T = []
        m_word = []
        m_tag = []
        count = 0
        for line in f.readlines():
            info = line.strip().split('  ')
            word_list = []
            tag_list = []
            count += 1
            for idx in range(len(info)):
                result = Inst()
                item = info[idx].split('/')
                result.m_word = item[0]
                result.m_tag = item[1]
                word_list.append(result.m_word)
                tag_list.append(result.m_tag)
                W.append(item[0])
                T.append(item[1])
            m_word.append(word_list)
            m_tag.append(tag_list)
        f.close()
        return m_word, m_tag, W, T

    def create_alphabet(self, W, T):
        word_list = []
        tag_list = []
        for w in W:
            if w not in word_list:
                word_list.append(w)
        word_list.append(self.hyperpara.unknow)
        word_list.append(self.hyperpara.padding)

        for t in T:
            if t not in tag_list:
                tag_list.append(t)
        # tag_list.append(self.hyperpara.unknow)
        tag_list.append(self.hyperpara.padding)
        word_dict = self.word_AlphaBet.makeVocab(word_list)
        tag_dict = self.tag_AlphaBet.makeVocab(tag_list)
        self.hyperpara.unknow_id = self.word_AlphaBet.dict[self.hyperpara.unknow]
        self.hyperpara.padding_id = self.word_AlphaBet.dict[self.hyperpara.padding]
        # self.hyperpara.tag_unknow_id = self.word_AlphaBet.dict[self.hyperpara.unknow]
        self.hyperpara.tag_padding_id = self.tag_AlphaBet.dict[self.hyperpara.padding]
        self.hyperpara.embedding_num = len(self.word_AlphaBet.makeVocab(word_list))
        self.hyperpara.tag_size = len(self.tag_AlphaBet.makeVocab(tag_list))
        return word_dict, tag_dict

    def change(self, m_word, m_tag, wrod_dict, tag_dict):
        all_examples = []
        for id in range(len(m_word)):
            example = Exam()
            for idx in range(len(m_word[id])):
                if m_word[id][idx] in wrod_dict:
                    example.m_word_index.append(wrod_dict[m_word[id][idx]])
                else:
                    example.m_word_index.append(wrod_dict["#-unknow-#"])
                if m_tag[id][idx] in tag_dict:
                    example.m_tag_index.append(tag_dict[m_tag[id][idx]])
                # else:
                #     example.m_tag_index.append(wrod_dict["#-unknow-#"])
            all_examples.append(example)
        return all_examples

    def Variable(self, example):
        batch = self.hyperpara.batch
        maxLength = 0
        for i in range(len(example)):
            if len(example[i].m_word_index) > maxLength:
                maxLength = len(example[i].m_word_index)

        x = Variable(torch.LongTensor(batch, maxLength))
        y = Variable(torch.LongTensor(batch, maxLength))
        for i in range(len(example)):
            for n in range(len(example[i].m_word_index)):
                x.data[i][n] = example[i].m_word_index[n]
                y.data[i][n] = example[i].m_tag_index[n]
                for j in range(len(example[i].m_word_index), maxLength):
                    x.data[i][j] = self.hyperpara.padding_id
                    y.data[i][j] = self.hyperpara.tag_padding_id

        return x, y
        # x = Variable(torch.LongTensor(1, len(example.m_word_index)))
        # y = Variable(torch.LongTensor(1, len(example.m_tag_index)))
        # for n in range(len(example.m_word_index)):
        #     x.data[0][n] = example.m_word_index[n]
        #     y.data[0][n] = example.m_tag_index[n]
        # return x, y

    def getMaxIndex(self, score):
        row = score.size()[0]
        column = score.size()[1]
        max_list = []
        for i in range(row):
            max = score.data[i][0]
            maxindex = 0
            for idx in range(column):
                tmp = score.data[i][idx]
                if max < tmp:
                    max = tmp
                    maxindex = idx
            max_index = maxindex
            max_list.append(max_index)
        return max_list

    def load_my_vecs(self, path, vocab, freqs, k=None):
        word_vecs = {}
        with open(path, encoding="utf-8") as f:
            count = 0
            lines = f.readlines()[1:]
            for line in lines:
                values = line.split(" ")
                word = values[0]
                count += 1
                if word in vocab:  # whether to judge if in vocab
                    vector = []
                    for count, val in enumerate(values):
                        if count == 0:
                            continue
                        if count <= k:
                            vector.append(float(val))
                    word_vecs[word] = vector
        return word_vecs

    def add_unknown_words_by_uniform(self, word_vecs, vocab, k=100):
        list_word2vec = []
        oov = 0
        iov = 0
        for word in vocab:
            if word not in word_vecs:
                oov += 1
                word_vecs[word] = np.random.uniform(-0.25, 0.25, k).round(6).tolist()
                list_word2vec.append(word_vecs[word])
            else:
                iov += 1
                list_word2vec.append(word_vecs[word])
        # print("oov count", oov)
        # print("iov count", iov)
        return list_word2vec

    def train(self, path_train, path_test):
        m_word_train, m_tag_train, W_train, T_train = self.read_file(path_train)
        m_word_test, m_tag_test, W_tset, T_test = self.read_file(path_test)

        word_dict, tag_dict = self.create_alphabet(W_train, T_train)

        e_train = self.change(m_word_train, m_tag_train, word_dict, tag_dict)
        e_test = self.change(m_word_test, m_tag_test, word_dict,tag_dict)

        word2vec = self.load_my_vecs(path=self.hyperpara.word_embedding_path,
                                     vocab=word_dict, freqs=None, k=300)
        self.hyperpara.pretrained_weight = self.add_unknown_words_by_uniform(word_vecs=word2vec,
                                                                             vocab=word_dict,
                                                                             k=300)

        self.model = CNN_text(self.hyperpara)
        model_count = 0
        listOfAcc = []
        steps = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperpara.lr)
        total_num = len(e_train)
        print('train num:', total_num)
        for epoch in range(1, self.hyperpara.epochs):
            print("————————第{}轮迭代，共{}轮————————".format(epoch, self.hyperpara.epochs))
            correct = 0
            sum = 0
            total = 0
            random.shuffle(e_train)
            part = total_num // self.hyperpara.batch
            if total_num % self.hyperpara.batch != 0:
                part += 1
            self.model.train
            for idx in range(part):
                begin = idx * self.hyperpara.batch
                end = (idx + 1) * self.hyperpara.batch
                if end > total_num:
                    end = total_num
                batch_list = []
                batch_list_len = []
                for idy in range(begin, end):
                    batch_list.append(e_train[idy])
                    batch_list_len.append(len(e_train[idy].m_word_index))
                # self.model.train
                # for m in e_train:
                optimizer.zero_grad()
                x, y = self.Variable(batch_list)
                y = torch.cat(y, 0)
                logit = self.model(x)
                loss = F.cross_entropy(logit, y)
                # b = self.getMaxIndex(logit)
                batch = x.size()[0]
                length = x.size()[1]
                logit = logit.view(batch, length, self.hyperpara.tag_size)
                for m in range(len(batch_list_len)):
                    b = self.getMaxIndex(logit[0])
                    for n in range(batch_list_len[m]):

                # for n in range(len(b)):
                        if y.data[n] == b[n]:
                            correct += 1
                        sum += 1
                total += 1
                loss.backward()
                optimizer.step()
                steps += 1
                print("current: ", total, "loss: ", loss.data[0], "acc", correct / sum)

            if not os.path.isdir(self.hyperpara.save_dir): os.makedirs(self.hyperpara.save_dir)
            save_prefix = os.path.join(self.hyperpara.save_dir, 'snapshot')
            save_path = '{}_steps{}.pt'.format(save_prefix, steps)
            torch.save(self.model, save_path)
            test_model = torch.load(save_path)
            model_count += 1
            test_acc = self.eval(e_test)
            print('test acc: ', test_acc)

            listOfAcc.append(test_acc)
        max = 0
        for i in listOfAcc:
            if i > max:
                max = i
        print("the best result is : ", max)

    def eval(self, e_path):
        self.model.eval()
        correct = 0
        sum = 0
        for m in e_path:
            x, y = self.Variable(m)
            y = torch.squeeze(y, 0)
            logit = self.model(x)
            b = self.getMaxIndex(logit)
            for n in range(len(b)):
                if y.data[n] == b[n]:
                    correct += 1
                sum += 1
        acc = correct / sum * 100.0
        if os.path.exists("./Test_Result.txt"):
            file = open("./Test_Result.txt", "a")
        else:
            file = open("./Test_Result.txt", "w")
        file.write('\nEvaluation -  acc: {:.4f}%({}/{}) \n'.format(acc,
                                                                   correct,
                                                                   sum))
        file.close()
        return acc

path_train = './data/Train_Corpus.pos'
path_test = './data/Test_Corpus_Gold.pos'
a = AV_Perceptron()
a.train(path_train, path_test)

# def running():
#     a = AV_Perceptron()
#     a.train(path_train, path_test)
#
# if __name__ == "__main__":
#     #直接把分析结果打印到控制台
#     # cProfile.run("running()")
#     # 把分析结果保存到文件中
#     # cProfile.run("running()", filename="result.out")
#     #增加排序方式
#     cProfile.run("running()", filename="result.out", sort="cumulative")



