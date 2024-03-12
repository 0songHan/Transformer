import torch
import torch.nn as nn
import torch.nn.functional as F
from demo03_decoder import *
from demo01_input import *
from demo04_output import *
from demo02_encoder import *
from copy import deepcopy

class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,source_embed,target_embed,generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.generator = generator

    def forward(self,source,target,source_mask,target_mask):
        #source:原始的编码器输入文本 [batch_size,sequence_length]
        #target:原始解码器输入文本
        #source_mask：对padding进行mask（作用在编码器和解码器的第二层）
        #target_mask:jiemaqi第一层
        #获得编码器结果的输出
        encode_output = self.encode(source,source_mask)
        #将编码器输出结果送入到解码器

        decode_output = self.decode(target,encode_output,source_mask,target_mask)

        output = self.generator(decode_output)
        return output


    #定义编码器结果输出的函数
    def encode(self,source,source_mask):
        #d对source进行wordembedding position
        source = self.source_embed(source)
        return self.encoder(source,source_mask)
    def decode(self,target,memory,source_mask,target_mask):
        #dui target进行处理
        target = self.target_embed(target)
        return self.decoder(target,memory,source_mask,target_mask)

def test_encoderDecoder():
    #定义输入
    source = target = torch.ones(2,4,dtype=torch.long)
    source_mask = target_mask = torch.zeros(1,4,4)

    #2.定义模型实例化各个参数
    #2.1编码器

    my_encoder = test_Encoder()

    #2.2实例化解码器
    my_decoder =  test_Decoder()

    #实例化source_embed
    my_src_embed = Embedding(vocab=1000,d_model=512)
    my_tgt_embed = Embedding(vocab=1200,d_model=512)

    #实例化输出
    my_generator = Generator(d_model=512,vocab=1200)

    #3.实例化EncoderDecoder
    transformer = EncoderDecoder(encoder=my_encoder,decoder=my_decoder,
                   source_embed=my_src_embed,target_embed=my_tgt_embed,
                   generator=my_generator)

    #将数据送入transformer
    result = transformer(source,target,source_mask,target_mask)
    print('transformerd的输出结果------>',result.shape)

def make_model(source_vocab,target_vocab,N=6,
               d_model=512,d_ff=2048,head=8,dropout=0.1):
    #深拷贝对象
    c = copy.deepcopy
    #实例化多头注意力对象
    attn = MutiHeadAttention(head=head,embed_dim=d_model,dropout=dropout)
    #实例化前馈全连接层
    ff = FeedForward(d_model=d_model,d_ff=d_ff,dropout=dropout)
    #位置编码器对象
    pe = PositionEncoding(d_model=d_model,dropout=dropout)


    #实例化编码器对象
    encode_embed = Embedding(vocab=source_vocab,d_model=d_model)
    #解码器
    decode_embed = Embedding(vocab=target_vocab, d_model=d_model)
    encoderlayer = EncoderLayer(size=d_model,self_attn=c(attn),feed_forward=c(ff),dropout=dropout)
    decoderlayer = DecoderLayer(size=d_model,self_attn=c(attn),src_attn=c(attn),feed_forward=c(ff),dropout=dropout)
    #size,self_attn,src_attn,feed_forward,dropout


    #实例化EncoderDecoder
    model = EncoderDecoder(encoder=Encoder(encoderlayer,N),
                           decoder=Decoder(decoderlayer,N),
                           source_embed=nn.Sequential(encode_embed,pe),
                           target_embed=nn.Sequential(decode_embed,pe),
                           generator=Generator(d_model=d_model,vocab=target_vocab))

    #对维度大于1的参数进行xavier初始化
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

    return model






if __name__ == '__main__':
    # test_encoderDecoder()

    model = make_model(source_vocab=1000,target_vocab=2000)
    print(model)


