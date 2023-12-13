import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer
from models.audio_encoder_config import AudioEncoderConfig
from models.audio_encoder import AudioEncoderModel
from tools.utils import align_word_embedding
from tools.file_io import load_json_file
import json
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, \
    TransformerDecoder, TransformerDecoderLayer
from models.Tokenizer import WordTokenizer




def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)




class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional audio_encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    



class TransformerModel(nn.Module):

    def __init__(self, config):
        super(TransformerModel, self).__init__()

        self.config = config
        self.is_keywords = config['keywords']

        # encoder
        encoder_config = AudioEncoderConfig(**config["audio_encoder_args"],
                                            audio_args=config["audio_args"])
        self.encoder = AudioEncoderModel(encoder_config)

        # tokenizer 
            # load tokenizer 
        self.tokenizer = WordTokenizer(config)

        # decoder settings
        self.decoder_only = config['decoder']['decoder_only']
        nhead = config['decoder']['nhead']  # number of heads in Transformer
        self.nhid = config['decoder']['nhid']  # number of expected features in decoder inputs
        nlayers = config['decoder']['nlayers']  # number of sub-decoder-layer in the decoder
        dim_feedforward = config['decoder']['dim_feedforward']  # dimension of the feedforward model
        activation = config['decoder']['activation']  # activation function of decoder intermediate layer
        dropout = config['decoder']['dropout']  # the dropout value


        self.pos_encoder = PositionalEncoding(self.nhid, dropout)
    

        vocab_path = config["path"]["vocabulary"]
        self.vocab = load_json_file(vocab_path)["vocabulary"]
        self.ntoken = len(self.vocab)
        self.sos_ind = self.vocab.index('<sos>')
        self.eos_ind = self.vocab.index('<eos>')


        decoder_layers = TransformerDecoderLayer(self.nhid,
                                                 nhead,
                                                 dim_feedforward,
                                                 dropout,
                                                 activation)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

        # linear layers
        # self.audio_linear = nn.Linear(1024, self.nhid, bias=True)
        self.enc_to_dec_proj = nn.Linear(encoder_config.hidden_size, self.nhid)
        self.dec_fc = nn.Linear(self.nhid, self.ntoken)
        self.generator = nn.Softmax(dim=-1)
        self.word_emb = nn.Embedding(self.ntoken, self.nhid)

        self.init_weights()


        # setting for pretrained word embedding
        if config['word_embedding']['freeze']:
            self.word_emb.weight.requires_grad = False
        if config['word_embedding']['pretrained']:
            self.word_emb.weight.data = align_word_embedding(self.vocab,
                                                             config['path']['word2vec'], config['decoder']['nhid'])
            
    def init_weights(self):
        # initrange = 0.1
        # self.word_emb.weight.data.uniform_(-initrange, initrange)
        init_layer(self.enc_to_dec_proj)
        # init_layer(self.dec_fc)

    def forward_encoder(self, audios):
        outputs = self.encoder(audios)
        outputs = self.enc_to_dec_proj(outputs.last_hidden_state)
        return outputs
    
    def generate_square_subsequent_mask(self, sz):

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        return mask

    def forward_decoder(self, encoder_outputs, texts, keyword=None, input_mask=None, target_mask=None, target_padding_mask=None):
        # tgt:(batch_size, T_out)
        # mem:(T_mem, batch_size, nhid)
        # print(f"audio shape ",encoder_outputs.shape)
        # print(f"keyword shape ",keyword.shape)
        encoder_outputs = encoder_outputs.permute(1,0,2)
        texts = texts.transpose(0, 1)
        if target_mask is None or target_mask.size()[0] != len(texts):
            device = texts.device
            target_mask = self.generate_square_subsequent_mask(len(texts)).to(device)

        if keyword is not None and self.is_keywords:
            keyword = self.word_emb(keyword)
            keyword = keyword.transpose(0, 1)
            # print(keyword.shape)
            # print(encoder_outputs.shape)
            encoder_outputs = torch.cat((encoder_outputs, keyword), dim=0)
        #print(encoder_outputs.shape)
        texts = self.word_emb(texts) * math.sqrt(self.nhid)

        texts = self.pos_encoder(texts)
        output = self.transformer_decoder(texts, encoder_outputs,
                                          memory_mask=input_mask,
                                          tgt_mask=target_mask,
                                          tgt_key_padding_mask=target_padding_mask)
        output = self.dec_fc(output)

        return output

    def forward(self, audios, texts, keyword=None, input_mask=None, target_mask=None, target_padding_mask=None):
        

        encoder_outputs = self.forward_encoder(audios)
        output = self.forward_decoder(encoder_outputs, texts, keyword=keyword,
                             input_mask=input_mask,
                             target_mask=target_mask,
                             target_padding_mask=target_padding_mask)
        

        return output