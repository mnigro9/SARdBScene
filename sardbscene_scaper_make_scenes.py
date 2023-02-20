#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scaper
import numpy as np
import os
import soundfile as sf
import librosa
import glob
import pandas as pd
np.random.seed(123)
fs=44100 #44.1kHz sampling rate

#%% How to resample with librosa
y,f = librosa.load('/Users/michaelnigro/Documents/Spyder/SARdBScene_AMIchunks/train/EN2001a_0010_0020_3.wav',16000)
y2 = librosa.resample(y,f,44100)
sf.write('/Users/michaelnigro/Documents/testfile.wav',y2,44100,'PCM_32') #44.1khz and 32 bit
#%% Scaper parameters fixed
ref_db = -20
duration = 10.0 
'''
min_events = 1
max_events = 6

event_time_dist = 'truncnorm'
event_time_mean = 5.0
event_time_std = 2.0
event_time_min = 0.0
event_time_max = 10.0 #event duration will be randomly chosen from a truncated normal distribution with a mean of 5 seconds,
# standard deviation of 2 second, and min/max values of 0 and 10 seconds respectively
'''
source_time_dist = 'const'
source_time = 0.0 #source will start at 0s always. From the beginning of its recording

event_duration_dist = 'uniform'
event_duration_min = 0.5
event_duration_max = 10   #event will be between 0.5 and 10 s long

snr_dist = 'uniform'
snr_min = -5
snr_max = 5  #sounds will be within 10dB of one another, relative to -20dB

pitch_dist = 'uniform'
pitch_min = -3.0
pitch_max = 3.0

time_stretch_dist = 'uniform'
time_stretch_min = 0.8
time_stretch_max = 1.2

#%% For making urban and home mixtures (no speech)
'''
 Urban Scene has 1-8 sfx, 125 soundscapes each for valid/test and 650 for training. [1,2,3,4,5,6,7,8]
 
home has 1-9 sfx, 111 soundscapes each for valid/test, and 577 for training [1,2,3,4,5,6,7,8,9]

'''

scene = 'urban' #office, home, nature, urban
speaker_count=0 #set according to scene containing speech or not
split = 'train' #train, test, train
expected_counts_list = [1,2,3,4,5,6,7,8] #the source sfx counts to make mixes for
n_soundscapes = 650  ### specify according to scene and source count class

# OUTPUT FOLDER
outfolder = '/Users/michaelnigro/Documents/SARdBScene/'+split+'/'

# SCAPER SETTINGS
fg_folder = '/Users/michaelnigro/Documents/SARdBScene_sources/'+scene+'/'+split+'/'
bg_folder = '/Users/michaelnigro/Documents/background/'


    
data_folder=[] #keeps track of mixture annotations, information
ind = 0

for n_events in expected_counts_list: #the source counts
    print(f'Beginning {scene} scene generation for {n_events} sources')
    for n in range(n_soundscapes):
        ind=ind+1
        sources=[]
        label_list=[]
        print('Generating soundscape: {:d}/{:d}'.format(n+1, n_soundscapes))
    
        # create a scaper
        sc = scaper.Scaper(duration, fg_folder, bg_folder)#, random_state=seed)
        sc.protected_labels = []
        sc.ref_db = ref_db
    
    
        source_count = n_events+speaker_count #no speech class so same value as n_events. If speech class then do source_count=n_events+1
        for _ in range(n_events):
            sc.add_event(label=('choose', []), 
                     source_file=('choose', []), 
                     source_time=(source_time_dist, source_time), 
                     #event_time=(event_time_dist, event_time_mean, event_time_std, event_time_min, event_time_max), 
                     event_time=('uniform',0,10),
                     event_duration=(event_duration_dist, event_duration_min, event_duration_max),
                     #sources will be at most 10dB louder from one another
                     snr=(snr_dist, snr_min, snr_max),
                     pitch_shift=(pitch_dist, pitch_min, pitch_max),
                     #pitch_shift=None, #to not do augmentation set None
                     time_stretch=(time_stretch_dist, time_stretch_min, time_stretch_max))
    
        # generate
        audiofile = os.path.join(outfolder, f"{split}_{scene}_{ind:04}_{source_count}.wav")
        jamsfile = os.path.join(outfolder, f"{split}_{scene}_{ind:04}_{source_count}.jams")
        txtfile = os.path.join(outfolder, f"{split}_{scene}_{ind:04}_{source_count}.txt")
    
        mixture_audio, mixture_jam, annotation_list, stem_audio_lis= sc.generate(audiofile, jamsfile,
                allow_repeated_label=False, #set False so only one source per class type
                allow_repeated_source=False,
                fix_clipping=True,
                reverb=None,
                disable_sox_warnings=True,
                no_audio=False,
                txt_path=txtfile)
    
        #sf.write(audiofile,mixture_audio,fs,'PCM_16') #saving mixture as 44.1khz 16bit pcm format
    
        ann = mixture_jam.annotations.search(namespace='scaper')[0]

        # print the sampled parameters for each event in the annotation
        for event in ann:
            label_list.append(event.value['label'])
            sources.append(event.value['source_file'])
        
        
        data_folder.append([audiofile,sources,scene,label_list,source_count,speaker_count,n_events]) 
      # if speaker_count is 0, then sfx_count and source_count are the same value. Source_count treats 'speech' as single class, speaker_count is the detailed count for speech

# to save annotations in individual files by scene and data split

df = pd.DataFrame(data_folder,columns=['Filename','Source Files','Scene','Event Labels','Source Count','Speaker Count','SFX Count'])
#%
df.to_csv(f'/Users/michaelnigro/Documents/SARdBScene_annotations/{split}_{scene}.csv',index=False)
#%% clear df before next run
del df

#%% for office and nature

#save the source count labels in separate file
'''
office scene has 0-4 sfx plus speech 260(n_soundscapes) and 208(n_soundscapes2) per speaker count & sfx count combo in training and  
valid/test is 50(n_soundscapes) and 40(n_soundscapes2). 
expected_counts_list=[1,2,3,4] expected_counts_list2=[0,1,2,3,4]

nature scene
0-11 sfx plus speech
Training: n_soundscapes=93  n_soundscapes2=87
Valid/test: n_soundscapes=17  n_soundscapes2=17
expected_counts_list=[1,2,3,4,5,6,7,8,9,10,11]  expected_counts_list2=[0,1,2,3,4,5,6,7,8,9,10,11]
'''

spe1 = sorted(glob.glob('/Users/michaelnigro/Documents/Spyder/SARdBScene_AMIchunks/test/*_1.wav')) #gets a list of filenames in alphabetical order from the given path, wildcard char at end to get the right speaker countclass

#%%
scene = 'nature' #office, home, nature, urban
speaker_count=1 #set according to scene containing speech or not
split = 'valid' #train, test, train
expected_counts_list = [1,2,3,4,5,6,7,8,9,10,11] #the source sfx counts to make mixes for
expected_counts_list2 = [0,1,2,3,4,5,6,7,8,9,10,11]
n_soundscapes = 17  # number of examples per sfx count when no speakers
n_soundscapes2 = 17 #number of examples per count when speakers active



# OUTPUT FOLDER
outfolder = '/Users/michaelnigro/Documents/SARdBScene/'+split+'/'

# SCAPER SETTINGS
fg_folder = '/Users/michaelnigro/Documents/SARdBScene_sources/'+scene+'/'+split+'/'
bg_folder = '/Users/michaelnigro/Documents/background/'


data_folder=[] #keeps track of mixture annotations, information
ind = 0
for n_speakers in range(5): #loop through the expected number of speakers
    speech_tracker =0 #index for the speech files 
    print(f'Starting with {n_speakers} speakers...\n')
    if n_speakers==0:
        for n_events in expected_counts_list:
            print(f'Beginning {scene} scene generation for {n_events} sources')
            for n in range(n_soundscapes):
                ind=ind+1
                print(f'Generating mix {ind}, soundscape {n+1}/{n_soundscapes}')
                sources=[]
                label_list=[]
                # create a scaper
                sc = scaper.Scaper(duration, fg_folder, bg_folder)#, random_state=seed)
                sc.protected_labels = []
                sc.ref_db = ref_db
    
                source_count = n_events #no speech class so same value as n_events. If speech class then do source_count=n_events+1
                for _ in range(n_events):
                    sc.add_event(label=('choose', []), 
                                 source_file=('choose', []), 
                                 source_time=(source_time_dist, source_time), 
                                 #event_time=(event_time_dist, event_time_mean, event_time_std, event_time_min, event_time_max), 
                                 event_time=('uniform',0,10),
                                 event_duration=(event_duration_dist, event_duration_min, event_duration_max),
                                 #sources will be at most 10dB louder from one another
                                 snr=(snr_dist, snr_min, snr_max),
                                 pitch_shift=(pitch_dist, pitch_min, pitch_max),
                                 #pitch_shift=None, #to not do augmentation set None
                                 time_stretch=(time_stretch_dist, time_stretch_min, time_stretch_max))
    
                # generate
                audiofile = os.path.join(outfolder, f"{split}_{scene}_{ind:04}_{source_count}.wav")
                jamsfile = os.path.join(outfolder, f"{split}_{scene}_{ind:04}_{source_count}.jams")
                txtfile = os.path.join(outfolder, f"{split}_{scene}_{ind:04}_{source_count}.txt")
    
                mixture_audio, mixture_jam, annotation_list, stem_audio_lis= sc.generate(audiofile, jamsfile,
                                                                                         allow_repeated_label=False, #set False so only one source per class type
                                                                                         allow_repeated_source=False,
                                                                                         fix_clipping=True,
                                                                                         reverb=None,
                                                                                         disable_sox_warnings=True,
                                                                                         no_audio=False,
                                                                                         txt_path=txtfile)
                ann = mixture_jam.annotations.search(namespace='scaper')[0]

                # print the sampled parameters for each event in the annotation
                for event in ann:
                    label_list.append(event.value['label'])
                    sources.append(event.value['source_file'])
        
        
                data_folder.append([audiofile,sources,scene,label_list,source_count,n_speakers,n_events])
    
    if n_speakers>0: #mixture will contain speech
        speech_files = sorted(glob.glob('/Users/michaelnigro/Documents/Spyder/SARdBScene_AMIchunks/'+split+'/*_'+str(n_speakers)+'.wav'))
        for n_events in expected_counts_list2:
            if n_events==0:
                print(f'Beginning {scene} scene generation for {n_events} sources')
                for n in range(n_soundscapes2):
                    ind=ind+1
                    print(f'Generating mix {ind}, soundscape {n+1}/{n_soundscapes2}')
                    sources=[]
                    label_list=[]
                    source_count = n_events+speaker_count
                    audiofile = os.path.join(outfolder, f"{split}_{scene}_{ind:04}_{source_count}.wav")
                
                    y,f = librosa.load(speech_files[speech_tracker],16000)
                    sources.append(speech_files[speech_tracker])
                    label_list.append('speech')
                    speech_tracker = speech_tracker+1
                    speech = librosa.resample(y,f,44100)
                
                    sf.write(audiofile,speech,44100,'PCM_32')
                    data_folder.append([audiofile,sources,scene,label_list,source_count,n_speakers,n_events])
            if n_events>0:
                print(f'Beginning {scene} scene generation for {n_events} sources')
                for n in range(n_soundscapes2):
                    ind=ind+1
                    print(f'Generating mix {ind}, soundscape {n+1}/{n_soundscapes2}')
                    sources=[]
                    label_list=[]

                    # create a scaper
                    sc = scaper.Scaper(duration, fg_folder, bg_folder)#, random_state=seed)
                    sc.protected_labels = []
                    sc.ref_db = ref_db
    
                    source_count = n_events+speaker_count #no speech class so same value as n_events. If speech class then do source_count=n_events+1
                    for _ in range(n_events):
                        sc.add_event(label=('choose', []), 
                                 source_file=('choose', []), 
                                 source_time=(source_time_dist, source_time), 
                                 #event_time=(event_time_dist, event_time_mean, event_time_std, event_time_min, event_time_max), 
                                 event_time=('uniform',0,10),
                                 event_duration=(event_duration_dist, event_duration_min, event_duration_max),
                                 #sources will be at most 10dB louder from one another
                                 snr=(snr_dist, snr_min, snr_max),
                                 pitch_shift=(pitch_dist, pitch_min, pitch_max),
                                 #pitch_shift=None, #to not do augmentation set None
                                 time_stretch=(time_stretch_dist, time_stretch_min, time_stretch_max))
    
                    # generate
                    audiofile = os.path.join(outfolder, f"{split}_{scene}_{ind:04}_{source_count}.wav")
                    jamsfile = os.path.join(outfolder, f"{split}_{scene}_{ind:04}_{source_count}.jams")
                    txtfile = os.path.join(outfolder, f"{split}_{scene}_{ind:04}_{source_count}.txt")
    
                    mixture_audio, mixture_jam, annotation_list, stem_audio_lis= sc.generate(audiofile, jamsfile,
                                                                                         allow_repeated_label=False, #set False so only one source per class type
                                                                                         allow_repeated_source=False,
                                                                                         fix_clipping=True,
                                                                                         reverb=None,
                                                                                         disable_sox_warnings=True,
                                                                                         no_audio=False,
                                                                                         txt_path=txtfile)
                    ann = mixture_jam.annotations.search(namespace='scaper')[0]
                    
                    y,f = librosa.load(speech_files[speech_tracker],16000)
                    sources.append(speech_files[speech_tracker])
                    label_list.append('speech')
                    speech_tracker = speech_tracker+1
                    speech = librosa.resample(y,f,44100)
                    y2,f = librosa.load(audiofile,44100) #need to load the just made sfx mix to work properly
                    mix = speech+y2
                    sf.write(audiofile,mix,44100,'PCM_32')
                    
                    
                    # print the sampled parameters for each event in the annotation
                    for event in ann:
                        label_list.append(event.value['label'])
                        sources.append(event.value['source_file'])
        
        
                    data_folder.append([audiofile,sources,scene,label_list,source_count,n_speakers,n_events])
# to save annotations in individual files by scene and data split

df = pd.DataFrame(data_folder,columns=['Filename','Source Files','Scene','Event Labels','Source Count','Speaker Count','SFX Count'])
#%
df.to_csv(f'/Users/michaelnigro/Documents/SARdBScene_annotations/{split}_{scene}.csv',index=False)
#%% clear df before next run
del df
#%%

