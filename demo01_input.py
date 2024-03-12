import torch
import torch.nn as nn
import math

#embedding
class Embedding(nn.Module):
    def __init__(self,vocab,d_model):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.embed =nn.Embedding(self.vocab,self.d_model)

    def forward(self,x):
        #将embedding的数据进行标准化，符合标准正太分布化，因为初始值是xavier初始化
        return self.embed(x)* math.sqrt(self.d_model)


def test_embedding():
    d_model = 6
    vocab = 100
    my_embedding = Embedding(vocab,d_model)
    x = torch.tensor([[10,45,6,7],[12,43,54,43]])
    embed = my_embedding(x)
    print(embed)
#position encoding
class PositionEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=60):
        super().__init__()
        #d_model 维度 dropout 随即失活 max句子最大长度

        #dropout
        self.dropout = nn.Dropout(p=dropout)
        #初始  位置编码矩阵
        pe = torch.zeros(max_len,d_model)

        #要填充全零张量
        #先定义位置列
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)*-(math.log(10000) / d_model))
        my_matmul = position * div_term
        # print('my_matmul--->',my_matmul.shape)

        #位置编码奇数列 sin
        pe[:,0::2] = torch.sin(my_matmul)
        #o shu 列 cos
        pe[:,1::2] = torch.cos(my_matmul)

        pe = pe.unsqueeze(0)

        #buffer 会自动有self.pe的属性,不参与模型训练
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + self.pe[:,:x.size()[1]]
        # print('position--->',x.shape)
        return self.dropout(x)

def test_PositionEncoding():
    d_model = 512
    vocab = 1000
    my_embedding = Embedding(vocab,d_model)
    x = torch.zeros(2,50,dtype=torch.long)
    # print('x.shape',x.shape)
    # print('x',x)
    embed = my_embedding(x)
    # print('embed--->',embed.shape)
    my_position = PositionEncoding(d_model=d_model,dropout=0.1,max_len=100)
    position = my_position(embed)
    return position






if __name__ == '__main__':
    # test_embedding()
    # max_len = 6
    # d_model =20
    # pe = torch.zeros(max_len, d_model)
    # 要填充全零张量
    # 先定义位置列
    # position = torch.arange(0, max_len).unsqueeze(1)
    # div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000) / d_model))
    # my_matmul = position * div_term
    # print('my_matmul--->',my_matmul.shape)
    # print('mpe--->',pe.shape)
    # pe[:, 0::2] = torch.sin(my_matmul)
    # print('pe[:, 0::2]--->', pe[:, 0::2].shape)
    # test_PositionEncoding()
    # myposition = PositionEncoding(d_model=d_model,dropout=0.1,max_len=max_len)
    # x = torch.zeros(1,1,20)
    # print(myposition(x))
    # print(x.shape)
    test_PositionEncoding()


