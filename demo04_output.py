import torch
import torch.nn as nn
import torch.nn.functional as F
from demo03_decoder import *

class Generator(nn.Module):
    def __init__(self,d_model,vocab):
        super(Generator,self).__init__()
        self.linear = nn.Linear(d_model,vocab)

    def forward(self,x):
        x = self.linear(x)
        x = F.log_softmax(x,dim=-1)
        return  x

def test_generator():
    d_model = 512
    vocab = 1000
    my_generator = Generator(d_model,vocab)
    x = torch.randn(2,4,512)
    # x = test_Decoder()
    result = my_generator(x)
    print('result--------->',result.shape)




if __name__ == '__main__':
    test_generator()
