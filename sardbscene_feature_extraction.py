#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:23:53 2020

@author: Admin

Making features for speaker counting based on literature approaches:
    
    ∆ Wei, Determining number of speakers.... spectrograms, 8kHz stream uses 
    STFT frame size 25ms with 50% overlpa. 49 x 94 spectrogram image size
    
    Stoter, CountNet.... absolute value of STFT with Hann windows, 25ms frame 
    size and 10ms hop size. 5s samples at 16kHz produced 500x201 TF representation.
    also applied log(1+STFT) and mel40. 
    
    Zhang, End to end overlapped speech detection .... 
    500ms windows of speech, input to NN is 8000 dimensional raw waveform
    
    Valentin, OVerlapped speech....flattened spectrogram,signal envelope, and hisotgram 
    of speech signal as input
"""

import librosa
import librosa.display
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

#%% Read csv file as dataframe
df = pd.read_csv('/Users/michaelnigro/Documents/SARdBScene_annotations/train_urban.csv')

source_count_labels = df['Source Count'].to_numpy()
#%% Testing out melspectrogram code, plotting
#44100 sampling rate: 1024 = ~23ms(0.023s), 256=~5ms, 2048=~46ms
fft_size = 2048 #256  #1024 and 512 are good parameter
hop_size = 1024 #128
mel_filters = 64
# (2048,512,64) results in 64x862 melspec
# (1024,512,64) results in 64x862 melspec
# (2048,1024,64) results in 64x431 melspec, 46ms FFT length with 50% overlap
audiofile = df['Filename'][0]
y,sr = librosa.load(audiofile,sr=44100)
fmax = sr/2
spect = librosa.feature.melspectrogram(y=y, sr=sr, 
                                           n_fft=fft_size, 
                                           hop_length=hop_size, 
                                           n_mels=mel_filters,
                                           fmax=fmax)
mel_spect = librosa.power_to_db(spect, ref=np.max)
librosa.display.specshow(mel_spect, y_axis='mel', fmax=fmax, x_axis='time');
plt.title('Mel Spectrogram');
plt.colorbar(format='%+2.0f dB');

#%%
fft_size= 2048
hop_size = 1024 
mel_filters = 64
mel_spect_list=[]
for i in range(len(df)):
    audiofile = df['Filename'][i] #get files in order
    y,sr = librosa.load(audiofile,sr=44100)
    fmax = sr/2
    spect = librosa.feature.melspectrogram(y=y, sr=sr, 
                                           n_fft=fft_size, 
                                           hop_length=hop_size, 
                                           n_mels=mel_filters,
                                           fmax=fmax)
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    mel_spect_list.append(mel_spect)

mel_spect_list_np = np.array(mel_spect_list)
#%%
np.save('/Users/michaelnigro/Documents/SARdBScene_features/train_urban_sc_labels.npy',source_count_labels)
np.save('/Users/michaelnigro/Documents/SARdBScene_features/train_urban_melspec.npy',mel_spect_list_np)

#%% Urban Valid set
#%% Read csv file as dataframe
df = pd.read_csv('/Users/michaelnigro/Documents/SARdBScene_annotations/valid_urban.csv')

source_count_labels = df['Source Count'].to_numpy()

fft_size= 2048
hop_size = 1024 
mel_filters = 64
mel_spect_list=[]
for i in range(len(df)):
    audiofile = df['Filename'][i] #get files in order
    y,sr = librosa.load(audiofile,sr=44100)
    fmax = sr/2
    spect = librosa.feature.melspectrogram(y=y, sr=sr, 
                                           n_fft=fft_size, 
                                           hop_length=hop_size, 
                                           n_mels=mel_filters,
                                           fmax=fmax)
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    mel_spect_list.append(mel_spect)

mel_spect_list_np = np.array(mel_spect_list)
#%%
np.save('/Users/michaelnigro/Documents/SARdBScene_features/valid_urban_sc_labels.npy',source_count_labels)
np.save('/Users/michaelnigro/Documents/SARdBScene_features/valid_urban_melspec.npy',mel_spect_list_np)

#%% Urban Test set
#%% Read csv file as dataframe
split_scene = 'test_urban'

df = pd.read_csv(f'/Users/michaelnigro/Documents/SARdBScene_annotations/{split_scene}.csv')

source_count_labels = df['Source Count'].to_numpy()

fft_size= 2048
hop_size = 1024 
mel_filters = 64
mel_spect_list=[]
for i in range(len(df)):
    audiofile = df['Filename'][i] #get files in order
    y,sr = librosa.load(audiofile,sr=44100)
    fmax = sr/2
    spect = librosa.feature.melspectrogram(y=y, sr=sr, 
                                           n_fft=fft_size, 
                                           hop_length=hop_size, 
                                           n_mels=mel_filters,
                                           fmax=fmax)
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    mel_spect_list.append(mel_spect)

mel_spect_list_np = np.array(mel_spect_list)
#%%
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_sc_labels.npy',source_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_melspec.npy',mel_spect_list_np)

#%% Home Test set
#%% Read csv file as dataframe
split_scene = 'test_home'

df = pd.read_csv(f'/Users/michaelnigro/Documents/SARdBScene_annotations/{split_scene}.csv')

source_count_labels = df['Source Count'].to_numpy()

fft_size= 2048
hop_size = 1024 
mel_filters = 64
mel_spect_list=[]
for i in range(len(df)):
    audiofile = df['Filename'][i] #get files in order
    y,sr = librosa.load(audiofile,sr=44100)
    fmax = sr/2
    spect = librosa.feature.melspectrogram(y=y, sr=sr, 
                                           n_fft=fft_size, 
                                           hop_length=hop_size, 
                                           n_mels=mel_filters,
                                           fmax=fmax)
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    mel_spect_list.append(mel_spect)

mel_spect_list_np = np.array(mel_spect_list)
#%%
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_sc_labels.npy',source_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_melspec.npy',mel_spect_list_np)

#%% Home Valid set
#%% Read csv file as dataframe
split_scene = 'valid_home'

df = pd.read_csv(f'/Users/michaelnigro/Documents/SARdBScene_annotations/{split_scene}.csv')

source_count_labels = df['Source Count'].to_numpy()

fft_size= 2048
hop_size = 1024 
mel_filters = 64
mel_spect_list=[]
for i in range(len(df)):
    audiofile = df['Filename'][i] #get files in order
    y,sr = librosa.load(audiofile,sr=44100)
    fmax = sr/2
    spect = librosa.feature.melspectrogram(y=y, sr=sr, 
                                           n_fft=fft_size, 
                                           hop_length=hop_size, 
                                           n_mels=mel_filters,
                                           fmax=fmax)
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    mel_spect_list.append(mel_spect)

mel_spect_list_np = np.array(mel_spect_list)
#% saving variables
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_sc_labels.npy',source_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_melspec.npy',mel_spect_list_np)

#%% Read csv file as dataframe
split_scene = 'train_home'

df = pd.read_csv(f'/Users/michaelnigro/Documents/SARdBScene_annotations/{split_scene}.csv')

source_count_labels = df['Source Count'].to_numpy()

fft_size= 2048
hop_size = 1024 
mel_filters = 64
mel_spect_list=[]
for i in range(len(df)):
    audiofile = df['Filename'][i] #get files in order
    y,sr = librosa.load(audiofile,sr=44100)
    fmax = sr/2
    spect = librosa.feature.melspectrogram(y=y, sr=sr, 
                                           n_fft=fft_size, 
                                           hop_length=hop_size, 
                                           n_mels=mel_filters,
                                           fmax=fmax)
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    mel_spect_list.append(mel_spect)

mel_spect_list_np = np.array(mel_spect_list)
#% saving variables
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_sc_labels.npy',source_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_melspec.npy',mel_spect_list_np)
#%% Read csv file as dataframe
split_scene = 'valid_office'

df = pd.read_csv(f'/Users/michaelnigro/Documents/SARdBScene_annotations/{split_scene}.csv')

source_count_labels = df['Source Count'].to_numpy()
speaker_count_labels = df['Speaker Count'].to_numpy()
sfx_count_labels = df['SFX Count'].to_numpy()

fft_size= 2048
hop_size = 1024 
mel_filters = 64
mel_spect_list=[]
for i in range(len(df)):
    audiofile = df['Filename'][i] #get files in order
    y,sr = librosa.load(audiofile,sr=44100)
    fmax = sr/2
    spect = librosa.feature.melspectrogram(y=y, sr=sr, 
                                           n_fft=fft_size, 
                                           hop_length=hop_size, 
                                           n_mels=mel_filters,
                                           fmax=fmax)
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    mel_spect_list.append(mel_spect)

mel_spect_list_np = np.array(mel_spect_list)
#% saving variables
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_sc_labels.npy',source_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_spkc_labels.npy',speaker_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_sfxc_labels.npy',sfx_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_melspec.npy',mel_spect_list_np)

#%% Read csv file as dataframe
split_scene = 'test_office'

df = pd.read_csv(f'/Users/michaelnigro/Documents/SARdBScene_annotations/{split_scene}.csv')

source_count_labels = df['Source Count'].to_numpy()
speaker_count_labels = df['Speaker Count'].to_numpy()
sfx_count_labels = df['SFX Count'].to_numpy()

fft_size= 2048
hop_size = 1024 
mel_filters = 64
mel_spect_list=[]
for i in range(len(df)):
    audiofile = df['Filename'][i] #get files in order
    y,sr = librosa.load(audiofile,sr=44100)
    fmax = sr/2
    spect = librosa.feature.melspectrogram(y=y, sr=sr, 
                                           n_fft=fft_size, 
                                           hop_length=hop_size, 
                                           n_mels=mel_filters,
                                           fmax=fmax)
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    mel_spect_list.append(mel_spect)

mel_spect_list_np = np.array(mel_spect_list)
#% saving variables
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_sc_labels.npy',source_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_spkc_labels.npy',speaker_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_sfxc_labels.npy',sfx_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_melspec.npy',mel_spect_list_np)

#%% Read csv file as dataframe
split_scene = 'train_office'

df = pd.read_csv(f'/Users/michaelnigro/Documents/SARdBScene_annotations/{split_scene}.csv')

source_count_labels = df['Source Count'].to_numpy()
speaker_count_labels = df['Speaker Count'].to_numpy()
sfx_count_labels = df['SFX Count'].to_numpy()

fft_size= 2048
hop_size = 1024 
mel_filters = 64
mel_spect_list=[]
for i in range(len(df)):
    audiofile = df['Filename'][i] #get files in order
    y,sr = librosa.load(audiofile,sr=44100)
    fmax = sr/2
    spect = librosa.feature.melspectrogram(y=y, sr=sr, 
                                           n_fft=fft_size, 
                                           hop_length=hop_size, 
                                           n_mels=mel_filters,
                                           fmax=fmax)
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    mel_spect_list.append(mel_spect)

mel_spect_list_np = np.array(mel_spect_list)
#% saving variables
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_sc_labels.npy',source_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_spkc_labels.npy',speaker_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_sfxc_labels.npy',sfx_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_melspec.npy',mel_spect_list_np)

#%% Read csv file as dataframe
split_scene = 'valid_nature'

df = pd.read_csv(f'/Users/michaelnigro/Documents/SARdBScene_annotations/{split_scene}.csv')

source_count_labels = df['Source Count'].to_numpy()
speaker_count_labels = df['Speaker Count'].to_numpy()
sfx_count_labels = df['SFX Count'].to_numpy()

fft_size= 2048
hop_size = 1024 
mel_filters = 64
mel_spect_list=[]
for i in range(len(df)):
    audiofile = df['Filename'][i] #get files in order
    y,sr = librosa.load(audiofile,sr=44100)
    fmax = sr/2
    spect = librosa.feature.melspectrogram(y=y, sr=sr, 
                                           n_fft=fft_size, 
                                           hop_length=hop_size, 
                                           n_mels=mel_filters,
                                           fmax=fmax)
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    mel_spect_list.append(mel_spect)

mel_spect_list_np = np.array(mel_spect_list)
#% saving variables
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_sc_labels.npy',source_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_spkc_labels.npy',speaker_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_sfxc_labels.npy',sfx_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_melspec.npy',mel_spect_list_np)

#%% Read csv file as dataframe
split_scene = 'test_nature'

df = pd.read_csv(f'/Users/michaelnigro/Documents/SARdBScene_annotations/{split_scene}.csv')

source_count_labels = df['Source Count'].to_numpy()
speaker_count_labels = df['Speaker Count'].to_numpy()
sfx_count_labels = df['SFX Count'].to_numpy()

fft_size= 2048
hop_size = 1024 
mel_filters = 64
mel_spect_list=[]
for i in range(len(df)):
    audiofile = df['Filename'][i] #get files in order
    y,sr = librosa.load(audiofile,sr=44100)
    fmax = sr/2
    spect = librosa.feature.melspectrogram(y=y, sr=sr, 
                                           n_fft=fft_size, 
                                           hop_length=hop_size, 
                                           n_mels=mel_filters,
                                           fmax=fmax)
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    mel_spect_list.append(mel_spect)

mel_spect_list_np = np.array(mel_spect_list)
#% saving variables
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_sc_labels.npy',source_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_spkc_labels.npy',speaker_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_sfxc_labels.npy',sfx_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_melspec.npy',mel_spect_list_np)

#%% Read csv file as dataframe
split_scene = 'train_nature'

df = pd.read_csv(f'/Users/michaelnigro/Documents/SARdBScene_annotations/{split_scene}.csv')

source_count_labels = df['Source Count'].to_numpy()
speaker_count_labels = df['Speaker Count'].to_numpy()
sfx_count_labels = df['SFX Count'].to_numpy()

fft_size= 2048
hop_size = 1024 
mel_filters = 64
mel_spect_list=[]
for i in range(len(df)):
    audiofile = df['Filename'][i] #get files in order
    y,sr = librosa.load(audiofile,sr=44100)
    fmax = sr/2
    spect = librosa.feature.melspectrogram(y=y, sr=sr, 
                                           n_fft=fft_size, 
                                           hop_length=hop_size, 
                                           n_mels=mel_filters,
                                           fmax=fmax)
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    mel_spect_list.append(mel_spect)

mel_spect_list_np = np.array(mel_spect_list)
#% saving variables
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_sc_labels.npy',source_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_spkc_labels.npy',speaker_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_sfxc_labels.npy',sfx_count_labels)
np.save(f'/Users/michaelnigro/Documents/SARdBScene_features/{split_scene}_melspec.npy',mel_spect_list_np)
