import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from demo01_input import *

def subsequent_mask(size):
    temp = np.triu(np.ones((1,size,size)),k=1).astype('uint8')
    return torch.from_numpy(1-temp)

#自注意力机制
def attention(query, key, value,mask=None,dropout=None):
    #q,k,v 代表输入张量，mask掩码张量 dropput 失活对象
    #获取词嵌入维度
    d_k = query.size()[-1]
    #计算权重
    scores = torch.matmul(query, key.transpose(-2,-1))/ math.sqrt(d_k)
    # print('scores.shape--->', scores.shape)

    #判断是否进行掩码
    if mask is not None:
        scores = scores.masked_fill(mask==0,-1e9)
    # print('scores--->', scores)

    #对上述分数进行softmax得到权重
    p_attn = F.softmax(scores,dim=-1)
    # print('p_attn--->', p_attn)

    if dropout is not None:
        p_attn = dropout(p_attn)

    #jisuan
    my_attention = torch.matmul(p_attn,value)

    return my_attention,p_attn

def test_attention():
    mask = subsequent_mask(50)
    print(mask)
    pe_tensor_x = test_PositionEncoding()
    query = key = value = pe_tensor_x
    my_attention, p_attn = attention(query,key,value,mask=mask)
    print('my_attention--->',my_attention.shape)
    print('p_attn--->',p_attn.shape)

# clones函数
def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MutiHeadAttention(nn.Module):
    def __init__(self,head,embed_dim,dropout=0.1):
        super().__init__()
        #确认整除
        assert embed_dim % head == 0
        self.head = head
        self.d_k = embed_dim //head

        #定义4个线性层
        self.linears = clones(module=nn.Linear(embed_dim,embed_dim),N=4)

        self.attn = None

        self.dropout = nn.Dropout(p=dropout)

    #注意力计算
    def forward(self,query,key,value,mask=None):
        #mask需要生维度
        if mask is not None:
            mask = mask.unsqueeze(0)


        batch_size = query.size(0)

        query,key,value = [model(x).view(batch_size,-1,self.head,self.d_k).transpose(1,2)
        for model ,x in zip(self.linears,(query,key,value))]

        # print('query---',query.shape)
        # print('key---',key.shape)
        # print('value---',value.shape)

        my_attention,p_attn = attention(query,key,value,mask=mask)

        #transpose之后，再用view之前用contiguous
        my_attention = my_attention.transpose(1,2).contiguous().view(batch_size,-1,self.head*self.d_k)
        return self.linears[-1](my_attention)

def test_MultiHeadAttention():
    mask = subsequent_mask(50)
    pe_tensor_x = test_PositionEncoding()
    query = key = value = pe_tensor_x
    head = 8
    dropout =0.1
    my_mha = MutiHeadAttention(head,embed_dim=512)
    x = my_mha(query,key,value,mask=mask)
    # print('my_attention---->',x)
    return x


#前馈全连接层
class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(FeedForward, self).__init__()
        #d_model 第一个输入维度 d_ff输出维度
        #定义第一个线性层
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
        #dinigyi dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        #先经过第一个
        x = self.linear1(x)
        #将第一步进行relu
        x = F.relu(x)
        #在 dropout
        x = self.dropout(x)
        #经过第2个全连接层
        output = self.linear2(x)
        # print('前馈全连接层output---->',output.shape)
        return output

def test_FeedForward():
    mask = subsequent_mask(50)
    pe_tensor_x = test_PositionEncoding()
    query = key = value = pe_tensor_x
    head = 8
    dropout =0.1
    my_mha = MutiHeadAttention(head,embed_dim=512)
    x = my_mha(query,key,value,mask=mask)
    my_feed = FeedForward(d_model=512,d_ff=300)
    my_feed(x)








# def test_MutiHeadAttention():
#     head = 8
#     embed_dim = 512
#     pe = test_PositionEncoding()
#     query=key=value =pe
#     my_multi_head_attention = MultiHeadAttention(head,embed_dim)
#     my_multi_head_attention(query,key,value)

class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm, self).__init__()
        #fea=词嵌入维度 eps 防止分母为0
        #a2系数
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)

            #guifan
        y = self.a2*(x - mean) /(std+self.eps)+self.b2
        return y

def test_LayerNorm():
    #yong 多头的结果规范化
    attn_x = test_MultiHeadAttention()
    print('attn_x',attn_x.shape)
    d_model = 512
    my_layernorm = LayerNorm(features=d_model)
    output = my_layernorm(attn_x)
    print('output.shape----->',output.shape)
    print('output----->',output)


#子层连接
class SublyerConnection(nn.Module):
    def __init__(self,size,dropout=0.1):
        super(SublyerConnection,self).__init__()
        #size 词嵌入维度
        self.norm = LayerNorm(size)#norm 层
        self.dropout = nn.Dropout(p=dropout)


    def forward(self,x,sublayer):
        #x 输入数据 sublayer函数入口，函数名
        myresult = x +self.dropout(sublayer(self.norm(x)))
        # myresult = x +self.dropout(self.norm(sublayer(x)))
        return myresult



def test_SublyerConnection():
    #digyi 输入
    x = torch.ones(2,4,dtype=torch.long)
    vocab_size = 1000
    d_model = 512#词嵌入维度
    #实例化
    my_embed = Embedding(vocab_size,d_model)
    #x输入myembed
    embed_x = my_embed(x)
    print('embed.shape进行单词embedding------>',embed_x.shape)
    #实例化位置编码
    my_pos = PositionEncoding(d_model,dropout=0.1)
    position_x = my_pos(embed_x)
    print('position_x含有位置信息---->',position_x.shape)
    #shilihua 多头注意力
    my_mha = MutiHeadAttention(head=8,embed_dim=d_model)
    #封装  实现一个匿名函数，返回多头注意力 机制的结果
    mask = torch.zeros(1,4,4)
    #lambda 函数x是形参，我们只有调用这个函数的时候才传递实参
    sublayer = lambda x :my_mha(x,x,x,mask=mask)

    #实例化子层连接
    my_subcon = SublyerConnection(size=d_model)
    #传参
    result = my_subcon(position_x,sublayer)
    print('实例化子层连接后result.shape------>',result.shape)


# todo:编码器
class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        super(EncoderLayer, self).__init__()
        #size:词嵌入维度 self_attn多头注意力对象  feed_forward前馈全连接层的对象 dropout
        #dinigyi 多头注意力机制对象的属性
        self.atten = self_attn
        self.ff = feed_forward
        self.size = size
        #clone 两个子层连接结构,yong 这个方法复制
        self.my_subcons = clones(SublyerConnection(size=self.size,dropout=dropout),2)

    def forward(self,x,mask):
        #jingguo 第一个子层
        attn_sublayer = lambda x :self.atten(x,x,x,mask=mask)#z这个x代表形参
        first_x = self.my_subcons[0](x,attn_sublayer)
        #经过第二个子层
        second_x = self.my_subcons[1](first_x,self.ff)

        return second_x#最终编码器的输出

def test_EncoderLayer():
    x = torch.ones(2, 4, dtype=torch.long)
    vocab_size = 1000
    d_model = 512  # 词嵌入维度
    # 实例化
    my_embed = Embedding(vocab_size, d_model)
    # x输入myembed
    embed_x = my_embed(x)
    # print('embed.shape进行单词embedding------>', embed_x.shape)
    # 实例化位置编码
    my_pos = PositionEncoding(d_model, dropout=0.1)
    position_x = my_pos(embed_x)
    # print('position_x含有位置信息---->', position_x.shape)
    # shilihua 多头注意力
    my_mha = MutiHeadAttention(head=8, embed_dim=d_model)
    #shili 前馈
    my_ff =FeedForward(d_model,d_ff=2048)
    # 封装  实现一个匿名函数，返回多头注意力 机制的结果
    mask = torch.zeros(1, 4, 4)

    #shili Encoderlayer
    size = d_model
    my_encoder = EncoderLayer(size,my_mha,my_ff,dropout=0.1)

    #送入参数
    result = my_encoder(position_x,mask)
    # print('编码层result的形状---->',result.shape)
    return result

# todo:定义编码器 N个层
class Encoder(nn.Module):
    def __init__(self,layer,N):
        super(Encoder, self).__init__()
        #layer:编码器层对象
        self.layers = clones(layer, N)
        # 第N个编码器层的结果需要Norm化
        self.norm = LayerNorm(layer.size)

    def forward(self,x,mask):
        #数据经过N个编码器层
        for layer in self.layers:
            x = layer(x,mask)

        return self.norm(x)


def test_Encoder():
    x = torch.ones(2, 4, dtype=torch.long)
    vocab_size = 1000
    d_model = 512  # 词嵌入维度
    # 实例化
    my_embed = Embedding(vocab_size, d_model)
    # x输入myembed
    embed_x = my_embed(x)
    # print('embed.shape进行单词embedding------>', embed_x.shape)
    # 实例化位置编码
    my_pos = PositionEncoding(d_model, dropout=0.1)
    position_x = my_pos(embed_x)
    # print('position_x含有位置信息---->', position_x.shape)
    # shilihua 多头注意力
    my_mha = MutiHeadAttention(head=8, embed_dim=d_model)
    # shili 前馈
    my_ff = FeedForward(d_model, d_ff=2048)
    # 封装  实现一个匿名函数，返回多头注意力 机制的结果
    mask = torch.zeros(1, 4, 4)
    # shili Encoderlayer
    size = d_model
    my_encoder = EncoderLayer(size, my_mha, my_ff, dropout=0.1)
    #实例化 Encoder
    my_encoder = Encoder(my_encoder,6)
    # result = my_encoder(position_x,mask)
    # # print('编码器的输出形状',result.shape)
    # return  result
    return my_encoder


























if __name__=='__main__':
    # test_MultiHeadAttention()
    # a = subsequent_mask(5)
    # print(a)

    # test_attention()
    # test_MultiHeadAttention()
    # test_FeedForward()

    # test_LayerNorm()

    # test_SublyerConnection()

    # test_EncoderLayer()

    test_Encoder()
