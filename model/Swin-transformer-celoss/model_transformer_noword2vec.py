import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pdb
import math
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformer import MultiHeadedAttention, PositionwiseFeedForward
from transformer import EncoderDecoder, TransformerEncoder, EncoderLayer
from transformer import Decoder, DecoderLayer, Embeddings, Generator
from transformer import PositionalEncoding
import numpy as np
from torch.cuda import amp

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14, network="resnet101"):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        if network == 'resnet152':
            self.resnet = torchvision.models.resnet152(pretrained=True)
            self.resnet = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        elif network == 'resnet50':
            resnet = torchvision.models.resnet50(pretrained=True)  # pretrained ImageNet ResNet-101
            modules = list(resnet.children())[:-2]
            self.net = nn.Sequential(*modules)
            self.dim = 2048
        elif network == 'resnet101':
            resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
            modules = list(resnet.children())[:-2]
            self.resnet = nn.Sequential(*modules)
            self.dim = 2048
        elif network == 'densenet161':
            self.net = torchvision.models.densenet161(pretrained=True)
            self.net = nn.Sequential(*list(list(self.net.children())[0])[:-1])
            self.dim = 1920
        else:
            self.net = torchvision.models.vgg19(pretrained=True)
            self.net = nn.Sequential(*list(self.net.features.children())[:-1])
            self.dim = 512

        # resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        # modules = list(resnet.children())[:-2]
        # self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_transformer(N, d_model, tgt_vocab, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    transformer = EncoderDecoder(
            TransformerEncoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), N),
            # lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            jixiuyi,
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

def jixiuyi(x):
    return x

def make_transformer_decoder(N=2, d_model=512, d_ff=2048, h=8, dropout=0.1):
   
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N)
    for p in decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return decoder

class DecoderWithTransformer(nn.Module):

    def __init__(self, decoder_dim, vocab_size, dropout=0.1,tag=False,
                decoder_model="transformer"):
        super(DecoderWithTransformer, self).__init__()
        self.tag = tag
        self.decoder_model = decoder_model
        N=6
        h=8
        self.transformer = make_transformer(N=N, d_model=decoder_dim, tgt_vocab=vocab_size,d_ff=2048, h=h, dropout=0.1)

        self.fc_encoder = nn.Linear(1536, decoder_dim)

    def forward(self, encoded_captions, encoder_out, tgt_mask):

        with amp.autocast(self.tag):
            encoder_out = self.fc_encoder(encoder_out)
            src_masks = encoder_out.new_ones(encoder_out.shape[:2], dtype=torch.long)
            src_masks = src_masks.unsqueeze(-2)
            preds = self.transformer(encoder_out, encoded_captions, src_mask=src_masks, tgt_mask=tgt_mask)

        return preds

class DecoderWithTransformerDecoder(nn.Module):

    def __init__(self, decoder_dim, vocab_size, dropout=0.1,
                decoder_model="transformer-decoder"):
        super(DecoderWithTransformerDecoder, self).__init__()
        c = copy.deepcopy
        self.decoder_model = decoder_model
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=decoder_dim, nhead=8) #d_model:
        self.transformer = make_transformer_decoder(N=2, d_model=decoder_dim)

        self.embedding = nn.Sequential(Embeddings(decoder_dim, vocab_size), c(PositionalEncoding(decoder_dim, dropout)))

        self.generator = Generator(decoder_dim, vocab_size)

        self.fc_encoder = nn.Linear(2048, decoder_dim)

    def forward(self, encoder_out, encoded_captions, tgt_mask):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        encoder_out = self.fc_encoder(encoder_out)

        src_masks = encoder_out.new_ones(encoder_out.shape[:2], dtype=torch.long)
        src_masks = src_masks.unsqueeze(-2)
        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        
        #transformer decoder
        preds = self.transformer(embeddings, encoder_out, src_mask=src_masks, tgt_mask=tgt_mask)
        preds = self.generator(out)
        return preds, encoded_captions