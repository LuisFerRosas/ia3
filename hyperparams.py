# Audio
from sys import path


num_mels = 80
# num_freq = 1024
n_fft = 2048
sr = 22050
# frame_length_ms = 50.
# frame_shift_ms = 12.5
preemphasis = 0.97
frame_shift = 0.0125 # seconds
frame_length = 0.05 # seconds
hop_length = int(sr*frame_shift) # samples.
win_length = int(sr*frame_length) # samples.
n_mels = 80 # Número de bancos Mel a generar
power = 1.2 # Exponent for amplifying the predicted magnitude
min_level_db = -100
ref_level_db = 20
hidden_size = 256 #salida de la conv (vocab partitura)
embedding_size = 512
max_db = 100
ref_db = 20
    
n_iter = 60
# power = 1.5
outputs_per_step = 1

epochs = 10000
lr = 0.001
save_step = 2000
image_step = 500
batch_size = 3

cleaners='english_cleaners'

data_path = './nuevoDataset'
checkpoint_path = './checkpoint'
sample_path = './samples'
path_audio='./datos/audio'
path_partituras='./datos/partituras'
path_vocabularioPartitura='./vocabulario2.json'
path_audio_procesado='datos/audioProcesado/'