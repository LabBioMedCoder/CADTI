#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)  # No one line is allowed.
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # to device
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)    
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def clones(module, N):
    return [copy.deepcopy(module) for _ in range(N)]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model_1, d_model_2, n_heads, d_dim):
        super(MultiHeadAttention, self).__init__()
        self.d_dim = d_dim
        
        self.n_heads = n_heads
        self.d_model_1 = d_model_1
        self.d_model_2 = d_model_2
        
        self.W_Q_dense = nn.Linear(self.d_model_1, self.d_dim * self.n_heads, bias=False)
        self.W_K_dense = nn.Linear(self.d_model_2, self.d_dim * self.n_heads, bias=False)
        self.W_V_dense = nn.Linear(self.d_model_2, self.d_dim * self.n_heads, bias=False)
        
        self.scale_product = ScaledDotProductAttention(self.d_dim)
        self.out_dense = nn.Linear(self.n_heads * self.d_dim, self.d_model_1, bias=False)
        # self.n_heads * self.d_dim = const
        
    def forward(self, Q, K, V, attn_mask):
        Q_spare, batch_size = Q, Q.size(0)
       
        q_s = self.W_Q_dense(Q).view(batch_size, -1, self.n_heads, self.d_dim).transpose(1, 2)
        # q_s:[128, 4, 1, 32]
        k_s = self.W_K_dense(K).view(batch_size, -1, self.n_heads, self.d_dim).transpose(1, 2)
        # k_s:[128, 4, 50, 32]
        self.v_s = self.W_V_dense(V).view(batch_size, -1, self.n_heads, self.d_dim).transpose(1, 2)
        # v_s:[128, 4, 50, 32]

        context, self.attn = self.scale_product(q_s, k_s, self.v_s, attn_mask)
        # context:[128, 4, 1, 32]  self.attn: [128, 4, 1, 50], [128, 4, 1, 545]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_dim)
        # context:[128, 1, 128]
        context = self.out_dense(context)
        # context:[128, 1, 64]
        return context + Q_spare
'''

class CNN(nn.Module):
    def __init__(self, features, time_size):  # (64,64)
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(features, 32, kernel_size=5, stride=1, padding=2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        #self.dense_1 = nn.Linear(32 * int(int(time_size/2)), 128)
        #self.dense_1 = nn.Linear(4352, 128)
        self.dense_1 = nn.Linear(128, 128)
        self.dense_2 = nn.Linear(128, 32)
        self.dense_3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid_func = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, emb_mat):
        output = torch.transpose(emb_mat, -1, -2)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.dropout(output)

        output = self.conv2(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.dropout(output)
        
        output = output.view(-1, output.size(1) * output.size(2))

        # fully connected layer
        output = self.dense_1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.dense_2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.dense_3(output)
        output = self.sigmoid_func(output)    
        return output
'''

class CrossAttention(nn.Module):
    def __init__(self, model_params, feature_size=16, n_heads=6, d_dim=32, pooling_dropout=0.01, linear_dropout=0.01):
        super(CrossAttention, self).__init__()

        self.max_d = model_params['max_drug_seq']  # 50
        self.max_p = model_params['max_protein_seq']  # 545
        self.emb_size = model_params['emb_size']  # 64
        self.dropout_rate = model_params['dropout_rate']  # 0.1
        self.input_dim_drug = model_params['input_dim_drug']  # 1500
        self.input_dim_target = model_params['input_dim_target']  # 15000

        self.demb = Embeddings(self.input_dim_drug, self.emb_size, self.max_d, self.dropout_rate)
        self.pemb = Embeddings(self.input_dim_target, self.emb_size, self.max_p, self.dropout_rate)

        self.att_list_1 = MultiHeadAttention(feature_size, feature_size, n_heads, d_dim)  # feature_size=64
        self.att_list_2 = MultiHeadAttention(feature_size, feature_size, n_heads, d_dim)

        # self.dense_1 = nn.Linear(feature_size, 32)
        self.dense_1 = nn.Linear(feature_size, 32)
        self.dense_2 = nn.Linear(32, 16)
        self.dense_3 = nn.Linear(16, 1)

        self.dropout_layer_pool = nn.Dropout(pooling_dropout)
        self.dropout_layer_linear = nn.Dropout(linear_dropout)
        self.sigmoid_func = nn.Sigmoid()
        self.relu_func = nn.ReLU()

        #self.CNN = CNN(feature_size, feature_size)

    def forward(self, input_1, input_2, attn_mask_1, attn_mask_2, input_3, input_4):
        # attn_mask_1: [128,50], attn_mask_2: [128,545]
        attn_mask_1 = attn_mask_1.unsqueeze(1).unsqueeze(2)
        attn_mask_2 = attn_mask_2.unsqueeze(1).unsqueeze(2)
        attn_mask_1 = (1.0 - attn_mask_1) * -10000.0
        attn_mask_2 = (1.0 - attn_mask_2) * -10000.0

        # attr_feature
        self.h_out_1 = self.demb(input_1)  # batch_size x seq_length x embed_size

        self.h_out_2 = self.pemb(input_2)

        # net_feature
        self.h_out_3 = input_3
        self.h_out_4 = input_4

        # self.out_1, _ = torch.max(self.h_out_1, dim=1)  # [128, 64]
        #self.out_2, _ = torch.max(self.h_out_2, dim=1)  # [128, 64]
        #self.drug = torch.cat((self.out_1, self.h_out_3), dim=1)
        #self.protein = torch.cat((self.out_2, self.h_out_4), dim=1)
        # out_1_q, out_1_k, out_1_v = self.drug, self.protein, self.protein
        # self.out_1_temp = self.att_list_1(out_1_q, out_1_k, out_1_v, None)
        # self.drug_protein= torch.cat((self.drug, self.protein), dim=1)




        #out_1_q, out_1_k, out_1_v = self.h_out_3, self.h_out_1, self.h_out_1
        #self.out_1_temp = self.att_list_1(out_1_q, out_1_k, out_1_v, None)
        # #
        #out_2_q, out_2_k, out_2_v = self.h_out_4, self.h_out_2, self.h_out_2
        #self.out_2_temp = self.att_list_1(out_2_q, out_2_k, out_2_v, None)
        # #
        #self.out_1, _ = torch.max(self.out_1_temp, dim=1)  # [128, 64]
        #self.out_2, _ = torch.max(self.out_2_temp, dim=1)  # [128, 64]
        # self.drug_protein = torch.cat((self.out_1, self.out_2), dim=1)
        # #
        # #
        # out_3 = self.dropout_layer_pool(self.drug_protein)


        # cross attention

        out_1_q, out_1_k, out_1_v = self.h_out_3, self.h_out_3, self.h_out_3
       # out_1_q: [128,64], out_1_k: [128,50,64], out_1_v: [128,50,64], attn_mask_1:[128,1,1,50]


        self.out_1_temp = self.att_list_1(out_1_q, out_1_k, out_1_v, None)
        #self.out_1_temp = self.att_list_1(out_1_q, out_1_k, out_1_v, attn_mask_1)  # drug cross-attention
        #out_1_temp: [128,128,64]

        out_2_q, out_2_k, out_2_v = self.h_out_4, self.h_out_4, self.h_out_4
        #out_2_q: [128,64], out_2_k:[128,545,64], out_2_v: [128,545,64], attn_mask_2:[128,1,1,545]

        self.out_2_temp = self.att_list_2(out_2_q, out_2_k, out_2_v, None)
        #self.out_2_temp = self.att_list_2(out_2_q, out_2_k, out_2_v, attn_mask_2)  # protein cross-attention
        #out_2_temp: [128,128,64]

        out_3_q, out_3_k, out_3_v = self.out_1_temp, self.out_2_temp, self.out_2_temp
        #out_3_q: [128,128,64], out_3_k: [128,128,64], out_3_v :[128,128,64], attn_mask_3: [?,?,?]

        self.out_3_temp = self.att_list_2(out_3_q, out_3_k, out_3_v, None)

        #out_3_q, out_3_k, out_3_v = self.drug, self.protein, self.protein
        # self.out_3_temp = self.att_list_2(out_3_q, out_3_k, out_3_v, None)

        out_3 = self.dropout_layer_pool(self.out_3_temp)



        self.out_3, _ = torch.max(out_3, dim=1)  # [128, 64]
        #self.out_3 = self.dropout_layer_pool(self.drug_protein)
        #FCL
        out = self.dense_1(self.out_3)
        out = self.relu_func(out)
        out = self.dropout_layer_linear(out)
        out = self.dense_2(out)
        out = self.relu_func(out)
        out = self.dropout_layer_linear(out)
        out = self.dense_3(out)
        out = self.sigmoid_func(out)

        """
        # FCL
        #print('out_1 {}'.format(out_1.shape)) # [128, 50, 384]
        #print('out_2 {}'.format(out_2.shape))  # [128, 545, 384]
         # neural network sequential combination for cross attention
        self.out_1, _ = torch.max(out_1, dim = 1) #[128, 384]
        self.out_2, _ = torch.max(out_2, dim = 1) #[128, 384]
        self.out = torch.cat((self.out_1, self.out_2), dim = 1) 
        #print('self.out {}'.format(self.out.shape))  # [128, 768]

        out = self.dense_1(out_3)
        out = self.relu_func(out)
        out = self.dropout_layer_linear(out)
        out = self.dense_2(out)
        out = self.relu_func(out)
        out = self.dropout_layer_linear(out)
        out = self.dense_3(out)
        out = self.sigmoid_func(out)
        """

        """
        # Notice: reset of self.CNN = CNN(feature_size, self.max_d+self.max_p)
        # CNN sequential combination  for cross attention
        out = torch.cat((out_1, out_2), dim = 1) 
        #print('self.out {}'.format(self.out.shape))  # [128, 595, 384]
        out = self.CNN(out)
        """

        # out = self.CNN(out_2)  # drug

        return out,train_feature






