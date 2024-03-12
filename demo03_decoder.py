import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from demo02_encoder import *
from demo01_input import *
from copy import deepcopy


class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        super(DecoderLayer, self).__init__()
        #selfattn:子注意力 scr:一般注意力
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        #克隆
        self.sublayer = clones(SublyerConnection(size = self.size,dropout = dropout),3)



    def forward(self,x,memory,source_mask,target_mask):
        # x 解码器端的原始输入 memory编码器的结果作为kv source_mask 一般注意力的掩码
        #target_mask 自注意力层的掩码
        #经i过第一个子层lianjei结构
        self_attn_x = self.sublayer[0](x,lambda x : self.self_attn(x,x,x,mask=target_mask))
        #jin入第二个子层连接结构
        source_x = self.sublayer[1](self_attn_x,lambda x : self.src_attn(x,memory,memory,mask=source_mask))
        #第三个
        result = self.sublayer[2](source_x,self.feed_forward)
        return result

def test_decoder_layer():
    #获得解码器的输入
    x = torch.ones(2,4,dtype=torch.long)
    #将x送入embeding
    vocab=1000
    d_model = 512
    my_embed = Embedding(vocab=vocab,d_model=d_model)
    embed_x = my_embed(x)
    print('embedding结果----->',embed_x.shape)
    my_pe = PositionEncoding(d_model=d_model,dropout=0.1)
    position_x = my_pe(embed_x)
    print('position结果------>',position_x.shape)

    #实例化多头注意力层
    head = 8
    embed_dim = d_model
    mha = MutiHeadAttention(head=head,embed_dim=embed_dim)
    self_attn = copy.deepcopy(mha)
    src_attn = copy.deepcopy(mha)
    #实例化前馈全连接层
    ff = FeedForward(d_model=d_model,d_ff=2048)
    feed_forward = copy.deepcopy(ff)

    #实例化解码器层
    my_decoder = DecoderLayer(size=d_model,self_attn=self_attn,src_attn=src_attn,
                              feed_forward=feed_forward,dropout=0.1)
    #将数据输入解码器层得到结果
    #获得编码器的结果memory
    memory = test_Encoder()
    mask = torch.zeros(1,4,4)
    target_mask = copy.deepcopy(mask)
    source_mask = copy.deepcopy(mask)
    result = my_decoder(position_x,memory,source_mask,target_mask)
    print('解码层result结果------>',result.shape)


class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        #layer；jiemaqi解码器层的对象 N 有几个
        self.layers = clones(layer,N)
        #定义规范化层
        self.norm = LayerNorm(layer.size)

    def forward(self,x,memory,source_mask,target_mask):
        for layer in self.layers:
            x = layer(x,memory,source_mask,target_mask)
        return self.norm(x)



def test_Decoder():
    x = torch.ones(2, 4, dtype=torch.long)
    # 将x送入embeding
    vocab = 1000
    d_model = 512
    my_embed = Embedding(vocab=vocab, d_model=d_model)
    embed_x = my_embed(x)
    # print('embedding结果----->', embed_x.shape)
    my_pe = PositionEncoding(d_model=d_model, dropout=0.1)
    position_x = my_pe(embed_x)
    # print('position结果------>', position_x.shape)

    # 实例化多头注意力层
    head = 8
    embed_dim = d_model
    mha = MutiHeadAttention(head=head, embed_dim=embed_dim)
    self_attn = copy.deepcopy(mha)
    src_attn = copy.deepcopy(mha)
    # 实例化前馈全连接层
    ff = FeedForward(d_model=d_model, d_ff=2048)
    feed_forward = copy.deepcopy(ff)

    # 实例化解码器层
    my_decoder = DecoderLayer(size=d_model, self_attn=self_attn, src_attn=src_attn,
                              feed_forward=feed_forward, dropout=0.1)
    # 将数据输入解码器层得到结果
    # 获得编码器的结果memory
    memory = test_Encoder()
    mask = torch.zeros(1, 4, 4)
    target_mask = copy.deepcopy(mask)
    source_mask = copy.deepcopy(mask)
    #shiolihua 解码器
    my_decoder = Decoder(layer=my_decoder,N=6)
    # result = my_decoder(position_x,memory,source_mask,target_mask)
    # # print('解码器的输入结果result------->',result.shape)
    # return result
    return my_decoder




if __name__=="__main__":
    # test_decoder_layer()
    test_Decoder()




