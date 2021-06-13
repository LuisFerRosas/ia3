
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import  DataLoader
import numpy as np

def cargaDatos():
    data=[]
    sonido_dic=np.load("sonidos.npy",allow_pickle=True).item()
    partituras_dic=np.load("partituras.npy",allow_pickle=True).item()
    
    for sonido in sonido_dic:
        for part in partituras_dic:
            if part[:-4]==sonido[:-4]:
                audio=torch.tensor(sonido_dic[sonido],dtype=torch.float64)
                partitura=torch.tensor(partituras_dic[part])
                data.append((audio,partitura))
    
    return data

muestra= cargaDatos()
# print(muestra[0])
# print(len(completo[0][0]))
# print(audio.shape)

def generate_batch(data_batch):
    audio_batch, partitura_batch = [], []
    for (audio, partitura) in data_batch:
        audio_batch.append(torch.cat([torch.tensor([1]), audio, torch.tensor([2])], dim=0))
        partitura_batch.append( partitura)
    audio_batch = pad_sequence(audio_batch, padding_value=0)
    partitura_batch = pad_sequence(partitura_batch, padding_value=0)
    
    print(partitura_batch.shape,audio_batch.shape)
    return audio_batch, partitura_batch

# audio,part=generate_batch(muestra)
# print(audio.shape)
# print(part)

train_dataset= DataLoader(muestra,batch_size=3,collate_fn=generate_batch,shuffle=True)

x ,y=next(iter(train_dataset))
# print(x.shape)
# print(x)
# print(y.shape)
# print(y)
print(y[:-1, :])
