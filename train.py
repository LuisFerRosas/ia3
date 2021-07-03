from preprocess import get_dataset, DataLoader, collate_fn_transformer
from network import *
from utils import load_checkpoint

from tensorboardX import SummaryWriter
import os
from tqdm import tqdm

from preprocess import collate_fn_nuevo, obtenerDatos

load_model = True
NHEAD=8
num_decoder_layer=6
num_encoder_layer=6
vocab=121
post_mel_size=2500
num_hidden=hp.hidden_size
maxlen=130
NUM_EPOCHS=100
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
modelo=ModelTransformer(hp.embedding_size,NHEAD,num_decoder_layer,vocab,post_mel_size,num_hidden,num_encoder_layer,maxlen)

modelo=modelo.to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(
    modelo.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)
checkpoint_path = './checkpoint/checkpoint_transformer_75.pth.tar'

if load_model:
    load_checkpoint(torch.load(checkpoint_path), modelo, optimizer)



def main():
    dataset_train =obtenerDatos(pathPartituras='datos2/partituras',pathVocabulario='datos2/vocabulario2.json',
                          pathArchivoNPY='archivosnumpy/train_audio_nombre.npy')
    
    dataset_valid =obtenerDatos(pathPartituras='datos2/partituras',pathVocabulario='datos2/vocabulario2.json',
                          pathArchivoNPY='archivosnumpy/train_audio_nombre.npy')
   


    modelo.train()
    writer = SummaryWriter("runs/tranformer")
   
    estep=0
    for epoch in range(NUM_EPOCHS):
        dataloader_train = DataLoader(dataset_train, batch_size=1, collate_fn=  collate_fn_nuevo,shuffle=True)
        pbar = tqdm(dataloader_train)
        losses = 0
        for i, data in enumerate(pbar):
            estep=estep+1
            pbar.set_description("Processing at epoch %d"%epoch)
            character,  mel_input, pos_text, pos_mel, _ = data
            character = character.to(DEVICE)            
            mel_input = mel_input.to(DEVICE)
            pos_text = pos_text.to(DEVICE)            
            pos_mel = pos_mel.to(DEVICE)
           
           
            output=modelo(character,mel_input,pos_text,pos_mel)
            # print(output)
            if estep==1:
                writer.add_graph(modelo ,input_to_model=[character,mel_input,pos_text,pos_mel])
            
            
            # print("output modelo...."+str(output.shape))
            # print("output trasformado..."+str(output.reshape(-1, output.shape[-1]).shape))
            # print("caracter ......"+str(character.reshape(-1).shape))
            optimizer.zero_grad()
            loss = loss_fn(output.reshape(-1, output.shape[-1]), character.reshape(-1))
            output=output.transpose(0,1)
            loss2=loss.item()
            writer.add_scalar("loss :",loss2 ,estep)
            # print("/////////////////")
            # print(np.argmax(output[0].detach().numpy(),axis=1))
            print("loss..........."+str(loss2))
            # print("Epoch.........."+str(epoch))
            
            loss.backward()
            optimizer.step()
            losses += loss.item()
        writer.add_scalar("loss2 :",losses ,epoch)
        if epoch+1 % hp.save_step==0:
            t.save({'model':modelo.state_dict(),
                    'optimizer':optimizer.state_dict()},
                                os.path.join(hp.checkpoint_path,'checkpoint_transformer_%d.pth.tar' % epoch))
    writer.close()

            



if __name__ == '__main__':
    main()
    





