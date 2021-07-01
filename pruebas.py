from network import TokenEmbedding
import os
import torch
import torch.nn as nn
from preprocess import collate_fn_nuevo, obtenerDatos
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import hyperparams as hp
from utils import get_positional_table, get_sinusoid_encoding_table,create_mask
from network import TokenEmbedding
from network import TmusicTrasforms

if __name__ == '__main__':
    dataset =obtenerDatos(pathPartituras='datos2/partituras',pathVocabulario='datos2/vocabulario2.json',
                          pathArchivoNPY='archivosnumpy/train_audio_nombre.npy')
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=  collate_fn_nuevo)
    pbar = tqdm(dataloader)
    variabel =next(iter(dataset))
    print(variabel['partitura_tokenizada'].shape)
    for i, data in enumerate(pbar):
        
        partitura_tokenizada,wave_mfcc = data
       
        print("partitura_ tokenizada //////////////// dimencion : "+str(partitura_tokenizada.shape))
        print(partitura_tokenizada)
       
        print("wave_mfcc //////////////// dimencion: "+str(wave_mfcc.shape))
        print(wave_mfcc.shape)
        model=TmusicTrasforms(8,n_vocabulario_tgt=233)
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model=model.to(DEVICE)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        model.train()
        ouput=model(wave_mfcc,partitura_tokenizada)
        if i==0:
            break
    