import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
#from apex import amp
from torch.cuda import amp

# import matplotlib.pyplot as plt
# import seaborn
# seaborn.set_context(context="talk")
# %matplotlib inline
# Table 1: Post-LN Transformer v.s. Pre-LN Transformer
# ON LAYER NORMALIZATION IN THE TRANSFORMER ARCHITECTURE - ICLR 2020
# https://openreview.net/pdf?id=B1x8anVFPr


# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
# https://scale.com/blog/pytorch-improvements
# Making Pytorch Transformer Twice as Fast on Sequence Generation.
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

# ------------------------------------------------------
# https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
# https://stackoverflow.com/questions/46452020/sinusoidal-embedding-attention-is-all-you-need


# def triangle_mask(size):
#     mask = 1- np.triu(np.ones((1, size, size)),k=1).astype('uint8')
#     mask = torch.autograd.Variable(torch.from_numpy(mask))
#     return mask

'''
triangle_mask(10)

mask
array([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]], dtype=uint8)
'''


# ------------------------------------------------------

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding_1d(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.from_numpy(pos_encoding.astype('float32')).to(device)


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

#layer normalization
class Norm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.bias  = nn.Parameter(torch.zeros(dim))
        self.eps   = eps
    def forward(self, x):
        #return x
        z = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps)
        x = self.alpha*z + self.bias
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_head, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.d_k = dim // num_head
        self.num_head = num_head
        self.dropout = dropout

        self.q = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def attention(self, q, k, v, mask):
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # torch.Size([8, 4, 10, 10]) = batch_size, num_head, LqxLk

        if mask is not None:
            mask = mask.unsqueeze(1)
            #print(score.min())
            score = score.masked_fill(mask == 0, -6e4) #-65504
            #score = score.masked_fill(mask == 0, -half('inf'))
            # https://github.com/NVIDIA/apex/issues/93
            # How to use fp16 training with masked operations

        score = F.softmax(score, dim=-1)

        if self.dropout > 0:
            score = F.dropout(score, self.dropout, training=self.training)

        value = torch.matmul(score, v)
        return value


    def forward(self, q, k, v,mask=None):
        batch_size, T, dim = q.shape

        # perform linear operation and split into h heads
        k = self.k(k)
        q = self.q(q)
        v = self.v(v)

        k = k.reshape(batch_size, -1, self.num_head, self.d_k)
        q = q.reshape(batch_size, -1, self.num_head, self.d_k)
        v = v.reshape(batch_size, -1, self.num_head, self.d_k)

        # transpose to get dimensions batch_size * num_head * T * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        value = self.attention(q, k, v, mask)

        # concatenate heads and put through final linear layer
        value = value.transpose(1, 2).contiguous().reshape(batch_size, -1, self.dim)
        value = self.out(value)
        return value


#---
class TransformerEncodeLayer(nn.Module):
    def __init__(self, dim, ff_dim, num_head, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(dim)
        self.norm2 = Norm(dim)

        self.attn = MultiHeadAttention(dim, num_head, dropout=dropout)
        self.ff   = FeedForward(dim, ff_dim,dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.norm1(x)
        x1 = self.attn(x1, x1, x1) #self-attention
        x   = x + self.dropout1(x1)

        x2 = self.norm2(x)
        x2 = self.ff(x2)
        x  = x + self.dropout2(x2)
        return x

class TransformerEncode(nn.Module):
    def __init__(self, dim, ff_dim, num_head, encoder_num_layer, dropout, max_len):
        super().__init__()
        self.pos_encoding_1d = positional_encoding_1d(max_len, dim)
        self.encoder_num_layer = encoder_num_layer
        self.layer = nn.ModuleList([
            TransformerEncodeLayer(dim, ff_dim, num_head,dropout=dropout) for i in range(encoder_num_layer)
        ])
        self.norm = Norm(dim)

    def forward(self, x):
        seq_len = x.shape[1]
        dec_pos = self.pos_encoding_1d[:, :seq_len, :]
        x = x + dec_pos
        for i in range(self.encoder_num_layer):
            x = self.layer[i](x)
        x = self.norm(x)
        return x

#---
class TransformerDecodeLayer(nn.Module):
    def __init__(self, dim, ff_dim, num_head, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(dim)
        self.norm2 = Norm(dim)
        self.norm3 = Norm(dim)

        self.attn1 = MultiHeadAttention(dim, num_head, dropout=dropout)
        self.attn2 = MultiHeadAttention(dim, num_head, dropout=dropout)
        self.ff = FeedForward(dim, ff_dim,dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, mem, x_mask=None, mem_mask=None):
        x1 = self.norm1(x)
        x1 = self.attn1(x1, x1, x1, mask=x_mask)  # self-attention
        x  = x + self.dropout1(x1)

        if mem is not None:
            x2 = self.norm2(x)
            x2 = self.attn2(x2, mem, mem)  # encoder input
            x  = x + self.dropout2(x2)

        x3 = self.norm3(x)
        x3 = self.ff(x3)
        x  = x + self.dropout3(x3)
        return x

    def forward_last(self, x_last, x_cache, mem, mem_mask):

        x_last_norm = self.norm1(x_last)
        x1 = torch.cat([x_cache, x_last_norm], 1)
        x_cache = x1.clone() # update

        x1 = self.attn1(x_last_norm, x1, x1)
        x_last  = x_last + x1

        if mem is not None:
            x2 = self.norm2(x_last)
            x2 = self.attn2(x2, mem, mem, mem_mask)
            x_last = x_last + x2


        x3 = self.norm3(x_last)
        x3 = self.ff(x3)
        x_last = x_last + x3

        return x_last, x_cache




# https://github.com/alexmt-scale/causal-transformer-decoder/blob/master/causal_transformer_decoder/model.py
class TransformerDecode(nn.Module):
    def __init__(self, dim, ff_dim, num_head, decoder_num_layer, vocab_size, max_len, drop_rate=0):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_encoding_1d = positional_encoding_1d(max_len, dim)
        self.num_layer = decoder_num_layer
        self.layer = nn.ModuleList([
            TransformerDecodeLayer(dim, ff_dim, num_head,dropout=drop_rate) for i in range(decoder_num_layer)
        ])
        self.norm = Norm(dim)
        self.dropout = nn.Dropout(drop_rate)
        self.logit = nn.Linear(dim, vocab_size)
    def forward(self, x, mem, x_mask=None, mem_mask=None):
        seq_len = x.shape[1]
        dec_pos = self.pos_encoding_1d[:, :seq_len, :]
        x = self.token_embed(x)
        x = x + dec_pos
        for i in range(self.num_layer):
            x = self.layer[i](x, mem,  x_mask, mem_mask)
        x = self.norm(x)
        predicitons = self.logit(x)
        return predicitons

    def forward_last(self, x_last, x_cache, mem,  mem_mask=None):
        batch_size,t,dim = x_last.shape
        assert(t==1)
        for i in range(self.num_layer):
            x_last, x_cache[i] = self.layer[i].forward_last(x_last, x_cache[i], mem, mem_mask)
        x_last = self.norm(x_last)
        return x_last, x_cache


class Transformer(nn.Module):

    def __init__(self, dim, ff_dim, num_head,  vocab_size, max_len, encoder_num_layer=6,decoder_num_layer=6, drop_rate=0.1, tag=False):
        super().__init__()
        self.tag =tag
        self.encoder_dim = nn.Linear(1536, dim)
        self.encoder = TransformerEncode(dim=dim, ff_dim=ff_dim, num_head=num_head, encoder_num_layer=encoder_num_layer, dropout=drop_rate, max_len=max_len)
        self.decoder = TransformerDecode(dim=dim, ff_dim=ff_dim, num_head=num_head, decoder_num_layer=decoder_num_layer, vocab_size=vocab_size, max_len=max_len, drop_rate=drop_rate)
    def forward(self, label, mem, x_mask=None, mem_mask=None):
        with amp.autocast(self.tag):
            mem = self.encoder_dim(mem)
            mem = self.encoder(mem)
            predictions = self.decoder(label, mem, x_mask)
        return predictions



# check ################################################################
# https://github.com/alexmt-scale/causal-transformer-decoder/blob/master/tests/test_consistency.py
def run_check_fast_decode():

    batch_size = 2
    length=9

    dim       = 4
    num_head  = 2
    ff_dim    = dim * num_head
    num_layer = 3

    decoder = TransformerDecode(dim, ff_dim, num_head, num_layer)
    decoder.eval()

    #----
    mem = torch.rand(batch_size, 5, dim)
    first_x  = torch.rand(batch_size, 1, dim)

    #----
    x1 = first_x
    for t in range(length - 1):
        # create mask for autoregressive decoding
        mask = 1 - np.triu(np.ones((batch_size, (t+1), (t+1))), k=1).astype(np.uint8)
        mask = torch.autograd.Variable(torch.from_numpy(mask))
        y = decoder( x1, mem, x_mask=mask )
        x1 = torch.cat( [x1, y[:,-1:]], dim=1)

    print(x1)
    print(x1.shape)

    #----

    x2 = first_x
    x_cache = [torch.empty(batch_size,0,dim) for i in range(num_layer)]
    for t in range(length - 1):
        y, x_cache = decoder.forward_last( x2[:,-1:], x_cache, mem )
        x2 = torch.cat( [x2, y], dim=1)

        #print(x_cache[0].shape)

    print(x2)
    print(x2.shape)

    print(torch.eq(x1, x2))

    diff = torch.abs(x1-x2)
    print(diff)
    print(diff.max(),diff.min())


# main #################################################################
if __name__ == '__main__':
    run_check_fast_decode()




