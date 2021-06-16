from preprocess import get_dataset, DataLoader, collate_fn_transformer
from network import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
from tqdm import tqdm
import hyperparams as hp
NHEAD=8
num_decoder_layer=6
num_encoder_layer=6
vocab=57
post_mel_size=2000
num_hidden=hp.hidden_size
maxlen=100
NUM_EPOCHS=10
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
modelo=ModelTransformer(hp.embedding_size,NHEAD,num_decoder_layer,vocab,post_mel_size,num_hidden,num_encoder_layer,maxlen)

modelo=modelo.to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(
    modelo.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)






def main():
    dataset=get_dataset()



    # writer = SummaryWriter()

    for epoch in range(NUM_EPOCHS):
        dataloader = DataLoader(dataset, batch_size=hp.batch_size, collate_fn=collate_fn_transformer, drop_last=True, )
        pbar = tqdm(dataloader)
        losses = 0
        for i, data in enumerate(pbar):
            pbar.set_description("Processing at epoch %d"%epoch)
            character, mel, mel_input, pos_text, pos_mel, _ = data
            character = character.to(DEVICE)
            mel = mel.to(DEVICE)
            mel_input = mel_input.to(DEVICE)
            
            # mel_input=mel_input[:, :-1]
            pos_text = pos_text.to(DEVICE)
            # pos_text=pos_text[:,:-1]
            pos_mel = pos_mel.to(DEVICE)
           
            # pos_mel=pos_mel[:, :-1]
            output=modelo(character,mel_input,pos_text,pos_mel)
            optimizer.zero_grad()
            loss = loss_fn(output.reshape(-1, output.shape[-1]), character.reshape(-1))
            print("loss..........."+str(loss))
            loss.backward()
            optimizer.step()
            losses += loss.item()



if __name__ == '__main__':
    main()






