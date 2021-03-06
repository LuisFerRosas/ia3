from torch.functional import norm
from module import *
from utils import get_positional_table, get_sinusoid_encoding_table,create_mask
import hyperparams as hp
import copy
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)
import torch
from torch import Tensor
class MelEncoder(nn.Module):
    def __init__(self,embedding_size,posmel_size, num_hidden,NHEAD,num_encoder_layers,dim_feedforward:int = 512,dropout:float = 0.1):
        super(MelEncoder,self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=embedding_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.embeding=TokenEmbedding(posmel_size,embedding_size)
        self.positional_encoding=PositionalEncoding(emb_size=embedding_size,dropout=dropout)
        
        self.encoder_prenet = Prenet(hp.num_mels, num_hidden * 2, num_hidden, p=0.2)
        self.norm = Linear(num_hidden, num_hidden*2)
        
        
    
    def forward(self, mel_input, pos_mel,mel_mask,mel_padding_mask):
        # print("1.......")
        # print(mel_input.shape)
        mel_input=self.encoder_prenet(mel_input)
        # print("2.......")
        # print(mel_input.shape)
        mel_input=self.norm(mel_input)
        # print("3.......")
        # print(mel_input.shape)
        # print("pos_mel.......")
        # print(pos_mel.shape)
        mel_input=self.positional_encoding(self.embeding(pos_mel))+mel_input
        # print("4.......")
        # print(mel_input.shape)
        mel_input=mel_input.transpose(0,1)
        # print("5.......")
        # print(mel_input.shape)
        # print("mel_mask ........"+str(mel_mask.shape))
        # print("mel_padding_mask........"+str(mel_padding_mask.shape))
        memory=self.transformer_encoder(mel_input,mel_mask,mel_padding_mask)
        # print("transformer salidad ..."+str(memory.shape))
        
        return memory
    
class TextDecode(nn.Module):
    def __init__(self,emb_size,NHEAD,num_decoder_layers,maxlen,vocab_size,dim_feedforward:int = 512, dropout:float = 0.1):
        super(TextDecode, self).__init__()
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.embeding=TokenEmbedding(maxlen,emb_size)
        self.positional_encoding=PositionalEncoding(emb_size=emb_size,dropout=dropout)
        self.generator = nn.Linear(emb_size, vocab_size)
        
        
    def forward(self,memory,caracters,pos_text,text_mask,text_padding_mask,memory_key_padding_mask):
        
        # print("De..memory.......")
        # print(memory.shape)
        # print("pos_text")
        # print(pos_text.shape)
        embtext= self.embeding(pos_text)
        # print("EmbText ........."+str(embtext.shape))
        # embCaracter= self.embeding(caracters)
        # print("EmbCaracter ........."+str(embCaracter.shape))
        
        text=self.positional_encoding(embtext)+self.embeding(caracters)
        # print("text........")
        # print(text.shape)
        text=text.transpose(0,1)
        # print("textTranspose .."+str(text.shape))
        outs = self.transformer_decoder(text, memory, text_mask, None,text_padding_mask, memory_key_padding_mask)
        # print("salida Deco.....")
        # print(outs.shape)
        outs=self.generator(outs)
        # print("salida final...")
        # print(outs.shape)
        
        return outs
        
        
        
class ModelTransformer(nn.Module):
    """
    Transformer Network
    """
    def __init__(self,emb_size,NHEAD,num_decoder_layers,vocab_size,posmel_size,num_hidden,num_encoder_layers,maxlen):
        super(ModelTransformer, self).__init__()
        self.encoder = MelEncoder(embedding_size=emb_size,posmel_size=posmel_size,num_hidden=num_hidden,NHEAD=NHEAD,
                                  num_encoder_layers=num_encoder_layers)
        self.decoder = TextDecode(emb_size,NHEAD,num_decoder_layers,maxlen,vocab_size)

    def forward(self, characters, mel_input, pos_text, pos_mel):
        pos_mel2=pos_mel.transpose(0,1)#[1590,3]
        pos_text2=pos_text.transpose(0,1)#[]
        mel_mask, text_mask, mel_padding_mask, tex_padding_mask=create_mask(pos_mel2,pos_text2,0)
        
        memory = self.encoder(mel_input,pos_mel,mel_mask,mel_padding_mask)
        # print("//////////////////////////////////////")
        # print(memory)
        outs = self.decoder(memory,characters,pos_text,text_mask,tex_padding_mask,mel_padding_mask)

        return outs   
        
        
        
        
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + 
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size+1, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    
 
