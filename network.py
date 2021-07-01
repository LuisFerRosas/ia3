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
    def __init__(self, emb_size: int, dropout, maxlen: int = 6000):
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
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    
  


class TmusicTrasforms(nn.Module):
    def __init__(self,num_emotions,n_vocabulario_tgt):
        super().__init__() 
        
        ################ TRANSFORMER BLOCK #############################
        # maxpool the input feature map/tensor to the transformer 
        # a rectangular kernel worked better here for the rectangular input spectrogram feature map/tensor
        self.transformer_maxpool = nn.MaxPool2d(kernel_size=[1,4], stride=[1,4])
        
        self.src_tok_emb = nn.Embedding(2752*2+40+1, 512)
        self.tgt_tok_emb = nn.Embedding(n_vocabulario_tgt, 512)
        self.positional_encoding = PositionalEncoding(512, dropout=0.3)
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=40, # input feature (frequency) dim after maxpooling 40*282 -> 40*70 (MFC*time)
            nhead=4, # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            dim_feedforward=512, # 2 linear layers in each encoder block's feedforward network: dim 40-->512--->40
            dropout=0.4, 
            activation='relu' # ReLU: avoid saturation/tame gradient/reduce compute time
        )
        
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)
        
        transformer_decoder_layer=nn.TransformerDecoderLayer(
            d_model=512,
            nhead=4,
            dim_feedforward=512,
            dropout=0.4,
            activation='relu'
        )
        self.transformer_decoder=nn.TransformerDecoder(transformer_decoder_layer,num_layers=6)

        ############### 1ST PARALLEL 2D CONVOLUTION BLOCK ############
        # 3 sequential conv2D layers: (1,40,282) --> (16, 20, 141) -> (32, 5, 35) -> (64, 1, 8)
        self.conv2Dblock1 = nn.Sequential(
            
            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1, # input volume depth == input channel dim == 1
                out_channels=8, # expand output feature map volume's depth to 16
                kernel_size=3, # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
                    ),
            nn.BatchNorm2d(8), # batch normalize the output feature map before activation
            nn.ReLU(), # feature map --> activation map
            nn.MaxPool2d(kernel_size=2, stride=2), #typical maxpool kernel size
            nn.Dropout(p=0.3), #randomly zero 30% of 1st layer's output feature map in training
            
            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=8, 
                out_channels=16, # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
                    ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=5), # increase maxpool kernel for subsequent filters
            nn.Dropout(p=0.3), 

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
                    ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # increase maxpool kernel for subsequent filters
            nn.Dropout(p=0.3), 
            
            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64, # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
                    ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
        )
        ############### 2ND PARALLEL 2D CONVOLUTION BLOCK ############
        # 3 sequential conv2D layers: (1,40,282) --> (16, 20, 141) -> (32, 5, 35) -> (64, 1, 8)
        self.conv2Dblock2 = nn.Sequential(
        # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1, # input volume depth == input channel dim == 1
                out_channels=8, # expand output feature map volume's depth to 16
                kernel_size=3, # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
                    ),
            nn.BatchNorm2d(8), # batch normalize the output feature map before activation
            nn.ReLU(), # feature map --> activation map
            nn.MaxPool2d(kernel_size=2, stride=2), #typical maxpool kernel size
            nn.Dropout(p=0.3), #randomly zero 30% of 1st layer's output feature map in training
            
            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=8, 
                out_channels=16, # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
                    ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=5), # increase maxpool kernel for subsequent filters
            nn.Dropout(p=0.3), 

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
                    ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # increase maxpool kernel for subsequent filters
            nn.Dropout(p=0.3), 
            
            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64, # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
                    ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
        
        )

        ################# FINAL LINEAR BLOCK ####################
        # Linear softmax layer to take final concatenated embedding tensor 
        #    from parallel 2D convolutional and transformer blocks, output 8 logits 
        # Each full convolution block outputs (64*1*8) embedding flattened to dim 512 1D array 
        # Full transformer block outputs 40*70 feature map, which we time-avg to dim 40 1D array
        # 512*2+40 == 1064 input features --> 8 output emotions 
        self.fc1_linear = nn.Linear(2752*2+40,num_emotions) 
        
        ### Softmax layer for the 8 output logits from final FC linear layer 
        self.softmax_out = nn.Softmax(dim=1) # dim==1 is the freq embedding
        
        # define one complete parallel fwd pass of input feature tensor thru 2*conv+1*transformer blocks
    def forward(self,x,partitura_tok):
    
        ############ 1st parallel Conv2D block: 4 Convolutional layers ############################
        # create final feature embedding from 1st convolutional layer 
        # input features pased through 4 sequential 2D convolutional layers
        conv2d_embedding1 = self.conv2Dblock1(x) # x == N/batch * channel * freq * time
        print("conv2d_embedding1 : "+str(conv2d_embedding1.shape))
        # flatten final 64*1*8 feature map from convolutional layers to length 512 1D array 
        # skip the 1st (N/batch) dimension when flattening
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1) 
        print("conv2d_embedding1 flatten : "+str(conv2d_embedding1.shape))
        ############ 2nd parallel Conv2D block: 4 Convolutional layers #############################
        # create final feature embedding from 2nd convolutional layer 
        # input features pased through 4 sequential 2D convolutional layers
        conv2d_embedding2 = self.conv2Dblock2(x) # x == N/batch * channel * freq * time
        print("conv2d_embedding2 : "+str(conv2d_embedding2.shape))
        # flatten final 64*1*8 feature map from convolutional layers to length 512 1D array 
        # skip the 1st (N/batch) dimension when flattening
        conv2d_embedding2 = torch.flatten(conv2d_embedding2, start_dim=1) 
        print("conv2d_embedding2 flatten : "+str(conv2d_embedding2.shape))
        
        ########## 4-encoder-layer Transformer block w/ 40-->512-->40 feedfwd network ##############
        # maxpool input feature map: 1*40*282 w/ 1*4 kernel --> 1*40*70
        x_maxpool = self.transformer_maxpool(x)
        print("x_maxpool : "+str(x_maxpool.shape))
        # remove channel dim: 1*40*70 --> 40*70
        x_maxpool_reduced = torch.squeeze(x_maxpool,1)
        print("x_maxpool_reduced : "+str(x_maxpool_reduced.shape))
        # convert maxpooled feature map format: batch * freq * time ---> time * batch * freq format
        # because transformer encoder layer requires tensor in format: time * batch * embedding (freq)
        x = x_maxpool_reduced.permute(2,0,1) 
        print("x------>entrada transformer encoder : "+str(x.shape))
        # finally, pass reduced input feature map x into transformer encoder layers
        transformer_output = self.transformer_encoder(x)
        print("salida transformer encoder : "+str(transformer_output.shape))
        
        # create final feature emedding from transformer layer by taking mean in the time dimension (now the 0th dim)
        # transformer outputs 2x40 (MFCC embedding*time) feature map, take mean of columns i.e. take time average
        transformer_embedding = torch.mean(transformer_output, dim=0) # dim 40x70 --> 40
        print("transformer_embedding  : "+str(transformer_embedding.shape))
    
        complete_embedding = torch.cat([conv2d_embedding1, conv2d_embedding2,transformer_embedding], dim=1)  
        print("complete_embedding  : "+str(complete_embedding.shape))
        complete_embedding=complete_embedding.transpose(1,0)
        print("complete_embedding222  : "+str(complete_embedding.shape))
        memory = self.src_tok_emb(complete_embedding)
        print("memory  : "+str(memory.shape))
        # partitura_tok=partitura_tok.transpose(1,0)
        # print("partitura_tok  : "+str(partitura_tok.shape))
        partitura_tok= self.tgt_tok_emb(partitura_tok)
        print("partitura_tok2  : "+str(partitura_tok.shape))
        tgt_emb = self.positional_encoding(partitura_tok)
        print("tgt_emb  : "+str(tgt_emb.shape))

        ouput_decoder=self.transformer_decoder(tgt_emb,memory)
        print("ouput_decoder  : "+str(ouput_decoder.shape))
        ######### final FC linear layer, need logits for loss #########################
        output_logits = self.fc1_linear(complete_embedding)  
        print("output_logits  : "+str(output_logits.shape))
        ######### Final Softmax layer: use logits from FC linear, get softmax for prediction ######
        output_softmax = self.softmax_out(output_logits)
        print("output_softmax  : "+str(output_softmax.shape))
        # need output logits to compute cross entropy loss, need softmax probabilities to predict class
        return output_logits, output_softmax     
