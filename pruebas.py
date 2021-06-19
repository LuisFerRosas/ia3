from network import TokenEmbedding
import os

import torch.nn as nn
from preprocess import get_dataset,collate_fn_transformer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import hyperparams as hp
from utils import get_positional_table, get_sinusoid_encoding_table,create_mask
from network import TokenEmbedding
if __name__ == '__main__':
    dataset =get_dataset()
    dataloader = DataLoader(dataset, batch_size=hp.batch_size, collate_fn=collate_fn_transformer, drop_last=True, )
    pbar = tqdm(dataloader)
    variabel =next(iter(dataset))
    # print(variabel['mel'].shape)
    for i, data in enumerate(pbar):
        
        character, mel, mel_input, pos_text, pos_mel, _ = data
        mel_mask, text_mask, mel_padding_mask, text_padding_mask=create_mask(pos_mel,character,0)
        print("Caracter ////////////////")
        print(character)#[3,47]
        print("mel ////////////////")
        print(mel.shape)#[3,1590,80]
        
        print("mel_input ////////////////")
        print(mel_input.shape)
        print("pos_text ////////////////"+str(pos_text.shape))#[3,47]
        print(pos_text.eq(0).unsqueeze(1).repeat(1, character.size(1), 1).shape)
        print("pos_mel ////////////////")
        print(pos_mel.shape)#[3,1590]
        # print("_ ////////////////")
        # print(_)
       
        # print("mel_mask ///////////////")
        # print(mel_mask)
        # print("text_mask ///////////////")
        # print(text_mask)
        # print("mel_padding_mask ///////////////")
        # print(mel_padding_mask)
        # print("text_padding_mask ///////////////")
        # print(text_padding_mask)
        # tensorp=TokenEmbedding(1590,512)
        # res=tensorp(pos_mel)
        # print(res.shape)#[3,1590,512]
        if i==0:
            break
    