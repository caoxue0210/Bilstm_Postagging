import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class CNN_text(nn.Module):
    def __init__(self, parameters):
        super(CNN_text, self).__init__()
        self.hyperpara = parameters
        self.embed_num = self.hyperpara.embedding_num
        self.embed_dim = self.hyperpara.embedding_dim
        self.class_num = self.hyperpara.tag_size
        self.hidden_dim = self.hyperpara.LSTM_hidden_dim
        self.num_layers = self.hyperpara.num_layers

        self.embed = nn.Embedding(self.embed_num, self.embed_dim)
        # pretrain （glove）
        if self.hyperpara.word_Embedding:
            pretrained_weight = np.array(self.hyperpara.pretrained_weight)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.embed.weight.requires_grad = True

        '''    
        ------------------------LSTM-------------------------------------
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim,
                            dropout=self.hyperpara.dropout,
                            num_layers=self.num_layers)
        self.fc1 = nn.Linear(self.hidden_dim * 2, self.class_num)
        self.hidden = self.init_hidden(self.num_layers * 2, self.hyperpara.batch)
        self.dropout_embed = nn.Dropout(self.hyperpara.dropout_embed)
        ------------------------LSTM - ------------------------------------
        '''

        #----------------BiLSTM-----------------------

        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim,
                            dropout=self.hyperpara.dropout,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=True)

        self.fc1 = nn.Linear(self.hidden_dim * 2, self.class_num)
        self.hidden = self.init_hidden(self.num_layers * 2, self.hyperpara.batch)
        self.dropout_embed = nn.Dropout(self.hyperpara.dropout_embed)
        # self.fc1 = nn.Linear(self.embed_dim, self.class_num)

        #-------------------BiLSTM-------------------------


    def init_hidden(self, num_layers, batch_size):
        return (Variable(torch.zeros(num_layers * 2, batch_size, self.hidden_dim)),
                Variable(torch.zeros(num_layers * 2, batch_size, self.hidden_dim)))

    def forward(self, x):
        batch = x.size()[0]
        len = x.size()[1]
        x = self.embed(x)  # (N,W,D)
        x = self.dropout_embed(x)  #batch_size * len * embed_dim
        '''
        print(x.size())
        print(x.permute(0, 2, 1).size())
        print(torch.squeeze(x, 0).size())
        print(x.size()[1])
        '''
        # print(x.permute(1, 0, 2))
        # x = F.max_pool1d(x.permute(0, 2, 1), x.size()[1])  # [(N,Co), ...]*len(Ks)
        hidden = self.init_hidden(self.hyperpara.num_layers, batch)
        out, hidden = self.lstm(x, hidden)
        # out = torch.squeeze(out, 0)
        # logit = self.fc1(out)

        dim = out.size()[2]
        # print(out.size())
        out = torch.cat(out, 0)
        logit = self.fc1(out)  # (N,C)
        # logit = logit.view(batch, len, self.class_num)
        return logit

