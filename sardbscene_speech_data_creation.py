#!/usr/bin/env python3

import os
import librosa
import soundfile as sf
import numpy as np
import glob

#%% Use AMI meeting corpus Headset mix
# individual headset audio recordings from AMI meeting corpus are combined into a single audio recording by summation of each individual recording. 
#%5 speaker meetings 
# files = ['EN2001a','EN2001d','EN2001e']


#%% Functions for dataset creation 
def xml_parse(name,num_chunks,inter):
    '''
    input arguments: name - the meeting name according to AMI corpus, num_chunks - the number of chunks available from the selected meeting recording, inter - the duration of a single chunk in seconds.
    ''' 
    import glob
    import xml.etree.ElementTree as ET
    import numpy as np
    nonos = [',','.','?','!','uh','um','mm','ah','huh','hmm','uh-huh','mm-hmm','uh-uh','mm-mm','Uh','Um','Mm','Ah','Huh','Hmm','Uh-huh','Uh-Huh','Mm-hmm','Mm-Hmm','Uh-Uh','Uh-uh','Mm-mm','Mm-Mm'] #list of non speech text

    filenames = sorted(glob.glob('/Users/michaelnigro/Documents/ami_manual_1.6.1/words/'+name+'*.xml'))
    speechstarts= []
    speechends = []
    totalwords = []
    
    for i,file in enumerate(filenames): #going through the word files for each speaker in a meeting, making lists of each word, start time, and end time
        
        tree = ET.parse(file)  #set path to xml file
        root = tree.getroot()
        starttime=[]
        endtime = []
        words=[]
        
        for child in range(len(root)):  #iterate over each line in xml file 'root'
            #print(child.tag, child.attrib)
            label = root[child].attrib  #gets 'labels' of each line. A line indicates a single word
            
            if len(label)==3:  # condition met for line containing text
                w = root[child].text
                if (type(w) is str) and (any(x in w for x in nonos)==False): #check its a word said and not punctuation    
                    words.append(root[child].text)
                    starttime.append(float(label['starttime']))  #get numeric value of start end times for a spoken word
                    endtime.append(float(label['endtime']))
                    
        totalwords.append(words)        
        speechstarts.append(starttime)
        speechends.append(endtime)
   
    t=0
    speaker_counts = []
    speaker_counts_plus =[] ### to get the number of continuous speech parts in a segment.
    
    while t<num_chunks*inter: #10:  #should set to length of audio file
        tn = t+inter #10
        speak_count=0

        if len(speechstarts)==4:
            if any(t<x<tn for x in speechstarts[0]) and any(t<x<tn for x in speechends[0]):
                speak_count=speak_count+1
            if any(t<x<tn for x in speechstarts[1]) and any(t<x<tn for x in speechends[1]):
                speak_count=speak_count+1
            if any(t<x<tn for x in speechstarts[2]) and any(t<x<tn for x in speechends[2]): #checks if any value is in time interval
                speak_count=speak_count+1
            if any(t<x<tn for x in speechstarts[3]) and any(t<x<tn for x in speechends[3]): #checks if any value is in time interval
                speak_count=speak_count+1

        if len(speechstarts)==5:
            if any(t<x<tn for x in speechstarts[0]) and any(t<x<tn for x in speechends[0]):
                speak_count=speak_count+1
            if any(t<x<tn for x in speechstarts[1]) and any(t<x<tn for x in speechends[1]):
                speak_count=speak_count+1
            if any(t<x<tn for x in speechstarts[2]) and any(t<x<tn for x in speechends[2]): #checks if any value is in time interval
                speak_count=speak_count+1
            if any(t<x<tn for x in speechstarts[3]) and any(t<x<tn for x in speechends[3]): #checks if any value is in time interval
                speak_count=speak_count+1
            if any(t<x<tn for x in speechstarts[4]) and any(t<x<tn for x in speechends[4]):
                speak_count=speak_count+1
           
        speaker_counts.append(speak_count)        
        t=tn

    scriptfull=[]  #list containing a transcript for each segment of audio
    whospoke_counts = [] #list of counts for the number of speech segments per speaker 

    if len(speechstarts)==4:
        no_more_speakers_flag=0  #tracks if all speakers have finished talking 
        for i in range(num_chunks): 
            times = np.array([speechstarts[0][0], speechstarts[1][0], speechstarts[2][0], speechstarts[3][0]])
            current_spk_word = np.argmin(times)
            t=i*inter
            tn = t+inter
            #print('doing time...',t,'-',tn)
            script=[]
            speaker_startcounts = [0,0,0,0,0] #tracking the number of speech segments for each speaker
            spk_count_plus_tally = 0
           
            tmp = 52 #temp variable to check if its a continuous speech
            
            while True:
                times = np.array([speechstarts[0][0], speechstarts[1][0], speechstarts[2][0], speechstarts[3][0]])
                #print(times)
                current_spk_word = np.argmin(times)
                if (speechstarts[current_spk_word][0]>tn): #checks if current word starts in present interval
                    #print('first break')
                    break
                if (speechends[current_spk_word][0]>tn):  #checks if current word ends in present interval
                    #print('second break')
                    break
                if no_more_speakers_flag==4: 
                    #print('third break')
                    break
                
                if (speechstarts[current_spk_word][0]>t and speechends[current_spk_word][0]<tn): #if current word is inside interval
                    script.append([totalwords[current_spk_word][0],current_spk_word])
                    if tmp != current_spk_word: #checks if we're changing the active speaker, then tally for the continuous speech segment counts
                        speaker_startcounts[current_spk_word]+=1 
                        spk_count_plus_tally+=1
                    tmp = current_spk_word #update temp variable to be latest speaker
                #now get rid of current speaker words and get new speaker
                totalwords[current_spk_word].pop(0)
                speechstarts[current_spk_word].pop(0)
                speechends[current_spk_word].pop(0)

                if len(speechstarts[current_spk_word])==0: #if a speaker has run out of words, just add padding
                    speechstarts[current_spk_word].append(1000000)
                    speechends[current_spk_word].append(1000010)
                    no_more_speakers_flag+=1 
                
            scriptfull.append(script)
            speaker_counts_plus.append(spk_count_plus_tally)
            whospoke_counts.append(speaker_startcounts)
            
    if len(speechstarts)==5: #same as previous code section but for 5 speakers max now
        no_more_speakers_flag=0   
        for i in range(num_chunks): 
            times = np.array([speechstarts[0][0], speechstarts[1][0], speechstarts[2][0], speechstarts[3][0], speechstarts[4][0]])
            current_spk_word = np.argmin(times)
            t=i*inter
            tn = t+inter
            #print('doing time...',t,'-',tn)
            script=[]
            speaker_startcounts = [0,0,0,0,0]
            spk_count_plus_tally = 0
            
            tmp = 52 #temp variable to check if its a continuous speech
            
            #if (speechstarts[current_spk_word][0]>t and speechends[current_spk_word][0]<tn): #making sure the current word is in the interval
            #while ((speechstarts[0][0]>t and speechends[0][0]<tn) or (speechstarts[1][0]>t and speechends[1][0]<tn) or (speechstarts[2][0]>t and speechends[2][0]<tn) or (speechstarts[3][0]>t and speechends[3][0]<tn) or (speechstarts[4][0]>t and speechends[4][0]<tn)):
            while True:
                times = np.array([speechstarts[0][0], speechstarts[1][0], speechstarts[2][0], speechstarts[3][0], speechstarts[4][0]])
                #print(times)
                current_spk_word = np.argmin(times)
                if (speechstarts[current_spk_word][0]>tn): 
                    #print('first break')
                    break
                if (speechends[current_spk_word][0]>tn): 
                    #print('second break')
                    break
                if no_more_speakers_flag==5:
                    #print('third break')
                    break
                
                if (speechstarts[current_spk_word][0]>t and speechends[current_spk_word][0]<tn):
                    script.append([totalwords[current_spk_word][0],current_spk_word])
                    if tmp != current_spk_word:
                        speaker_startcounts[current_spk_word]+=1
                        spk_count_plus_tally+=1
                    tmp = current_spk_word #update temp variable to be latest speaker
                #now get rid of current speaker and get new speaker
                totalwords[current_spk_word].pop(0)
                speechstarts[current_spk_word].pop(0)
                speechends[current_spk_word].pop(0)

                if len(speechstarts[current_spk_word])==0:
                    speechstarts[current_spk_word].append(1000000)
                    speechends[current_spk_word].append(1000010)
                    no_more_speakers_flag+=1 
                
            scriptfull.append(script)
            speaker_counts_plus.append(spk_count_plus_tally)
            whospoke_counts.append(speaker_startcounts)
    

    return speaker_counts,scriptfull,speaker_counts_plus,whospoke_counts


def scripture(words,ss):  ####NOT NEEDED JUST PROOF OF CONCEPT ######
    import numpy as np
    length_script = len(words[0])+len(words[1])+len(words[2])+len(words[3])
    script=[]
    for i in range(length_script):
        times = np.array([ss[0][0], ss[1][0], ss[2][0], ss[3][0]])  #current start time for each speaker
        current = np.argmin(times)  #position of word
    
        script.append(words[current][0])
        words[current].pop(0)
        ss[current].pop(0)
        if len(ss[current])==0:
            ss[current].append(10000)
    return script


    
#%% 2. 10s speech segment generated
'''
Going through each AMI file, chunk into 10s segments, save chunks with number 
of speakers.
'''
from pydub import AudioSegment, effects
from pydub.utils import make_chunks
import pandas as pd

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


# AMI files for non-scenario meetings listed here (EN,IB,IN)
ami_files = ['EN2001a','EN2001b','EN2001d','EN2001e',
             'EN2002a','EN2002b','EN2002d',  #EN2002c missing words doc D
             'EN2004a','EN2005a','EN2006a','EN2006b',  #EN2003a missing word doc D
             'EN2009d', #EN2009bc missing word doc D
             'ES2002a','ES2002c','ES2002d',
             'ES2003a','ES2003b','ES2003c','ES2003d',
             'ES2004a','ES2004b','ES2004c','ES2004d',
             'ES2005a','ES2005b','ES2005c','ES2005d',
             'ES2006a','ES2006b','ES2006c',
             'ES2007a','ES2007b','ES2007c','ES2007d',
             'ES2008a','ES2008b','ES2008d',
             'ES2009a','ES2009b','ES2009c','ES2009d',
             'ES2010a','ES2010b','ES2010c','ES2010d',
             'ES2011a','ES2011b','ES2011c','ES2011d',
             'ES2012a','ES2012b','ES2012c','ES2012d',
             'ES2013a','ES2013b','ES2013c','ES2013d',
             'ES2014a','ES2014b','ES2014c','ES2014d',
             'ES2015a','ES2015b','ES2015c','ES2015d',
             'ES2016a','ES2016b','ES2016c','ES2016d',
             'IB4001','IB4002','IB4003','IB4004','IB4005','IB4010','IB4011',
             'IN1002','IN1005','IN1007','IN1008','IN1009','IN1012','IN1013','IN1014','IN1016', #in1001 missind word doc
             'IS1000a','IS1000b','IS1000c','IS1000d',
             'IS1001a','IS1001b','IS1001c','IS1001d',
             'IS1002b','IS1002c','IS1002d',
             'IS1003a','IS1003b','IS1003c','IS1003d',
             'IS1004a','IS1004b','IS1004c','IS1004d',
             'IS1005a','IS1005b','IS1005c',
             'IS1006a','IS1006b','IS1006c','IS1006d',
             'IS1007a','IS1007b','IS1007c','IS1007d',
             'IS1008a','IS1008b','IS1008c','IS1008d',
             'IS1009a','IS1009b','IS1009c','IS1009d',
             'TS3003a','TS3003b','TS3003c','TS3003d',
             'TS3004a','TS3004b','TS3004c','TS3004d',
             'TS3005a','TS3005b','TS3005c','TS3005d',
             'TS3006a','TS3006b','TS3006c','TS3006d',
             'TS3007a','TS3007b','TS3007c','TS3007d',
             'TS3008a','TS3008b','TS3008c','TS3008d',
             'TS3009a','TS3009b','TS3009c','TS3009d',
             'TS3010a','TS3010b','TS3010c','TS3010d',
             'TS3011a','TS3011b','TS3011c','TS3011d',
             'TS3012a','TS3012b','TS3012c','TS3012d'
             ]
testy = ['IS1009b','EN2002a','IS1009c','TS3003a','EN2002b','TS3003b',#'EN2002c',
        'TS3003c','EN2002d',
        'TS3003d','ES2004a','TS3007a','ES2004b','TS3007b','ES2004c','TS3007c','ES2004d',
        'TS3007d','ES2014a','ES2014b','ES2014c','ES2014d','IS1009a','IS1009d']

validy = ['ES2003a','ES2003b','ES2003c','ES2003d','ES2011a','ES2011b','ES2011c','ES2011d',
         'IB4001','IB4002','IB4003','IB4004','IB4010','IB4011',
         'IS1008a','IS1008b','IS1008c','IS1008d','TS3004a','TS3004b','TS3004c','TS3004d',
         'TS3006a','TS3006b','TS3006c','TS3006d']

trainy = ['ES2002a','ES2002c','ES2002d',
          'ES2005a','ES2005b','ES2005c','ES2005d',
          'ES2006a', 'ES2006b','ES2006c',
          'ES2007a', 'ES2007b','ES2007c','ES2007d',
          'ES2008a','ES2008b','ES2008d',
          'ES2009a','ES2009b','ES2009c','ES2009d',
          'ES2010a','ES2010b','ES2010c','ES2010d',
          'ES2012a','ES2012b','ES2012c','ES2012d',
          'ES2013a','ES2013b','ES2013c','ES2013d',
          'ES2015a','ES2015b','ES2015c','ES2015d',
          'ES2016a','ES2016b','ES2016c','ES2016d',
          'IS1000a','IS1000b','IS1000c','IS1000d',
          'IS1001a','IS1001b','IS1001c','IS1001d',
          'IS1002b','IS1002c','IS1002d',
          'IS1003a','IS1003b','IS1003c','IS1003d',
          'IS1004a','IS1004b','IS1004c','IS1004d',
          'IS1005a','IS1005b','IS1005c',
          'IS1006a','IS1006b','IS1006c','IS1006d',
          'IS1007a','IS1007b','IS1007c','IS1007d',
          'TS3005a','TS3005b','TS3005c','TS3005d',
          'TS3008a','TS3008b','TS3008c','TS3008d',
          'TS3009a','TS3009b','TS3009c','TS3009d',
          'TS3010a','TS3010b','TS3010c','TS3010d',
          'TS3011a','TS3011b','TS3011c','TS3011d',
          'TS3012a','TS3012b','TS3012c','TS3012d',
          'EN2001a','EN2001b','EN2001d','EN2001e',
          'EN2004a', 'EN2005a','EN2006a','EN2006b','EN2009d',
          'IN1002', 'IN1005', 'IN1007', 'IN1008',
          'IN1009', 'IN1012', 'IN1013', 'IN1014', 'IN1016']
#%% Function to annotate a group of AMI meetings
def annotate_save_data(segment_length_s,list_of_files,data_split):
    '''
    inputs: 'segment_length_s': desired fixed duration of segments in seconds
            'list_of_files': list containing the filenames for ami meeting recordings, 
            'data_split': string, as 'train/', 'valid/', or 'test/'
    output: df: dataframe of meeting reocrding segment id and labels
    '''
    whereisaudio = '/Users/michaelnigro/Documents/Spyder/AMI_headset/'
    folder = '/Users/michaelnigro/Documents/Spyder/SARdBScene_AMIchunks/'  #set correct folder directory
    savehere = folder+data_split
    
    chunk_length_ms = segment_length_s*1000 #10000 # value is in miliseconds. Rounds up when making chunks if not divisible
    #cs = int(chunk_length_ms/1000) #chunk length in seconds
    spkcount_hist = [0,0,0,0,0,0]
    starttimes_list=[]
    labels = []

    
    for file in list_of_files: #ami_files:  
        myaudio = AudioSegment.from_file(whereisaudio+file+'.wav',"wav")  #load one file
        chunks = make_chunks(myaudio, chunk_length_ms) #makes 10s chunks as AudioSegment objects
        print(file)
        num_speakers,script,num_starttimes,whospoke_counts  = xml_parse(file,len(chunks),segment_length_s)  #get number of speakers in each chunk
        for i, chunk in enumerate(chunks):
            start = segment_length_s*i  #eg. 0,10,20,30....
            stop = segment_length_s + start #eg. 10,20,30,40....
            nspk = num_speakers[i]  #number of speakers in that chunk according to words.xml file 
            nstarts = num_starttimes[i]
            nwhospoke = whospoke_counts[i]
            nwhospoke.sort(reverse=True) #put in descending order (largest counts first)
        
            #nwhospoke=''.join(str(e) for e in nwhospoke) #to get list into string
            if nspk!=0:  #dont save 0 speaker segments
                spkcount_hist[nspk]+=1 #keeping track of the number of data samples per speaker class
                starttimes_list.append(nstarts)
                labels.append([f"{file}_{start:04}_{stop:04}", nspk, nstarts, nwhospoke])
                chunk_name = f"{file}_{start:04}_{stop:04}_{nspk}" #"_{nstarts}_{nwhospoke}"
                #print(chunk_name)
                normalized_chunk = match_target_amplitude(chunk, -20.0) #normalize to perceived loudness average of -20dbfs
                normalized_chunk.export(savehere+f'{nspk}/'+chunk_name+'.wav', format="wav")  #naming convention: 'meeting id','start_time','stop_time',num of speakers)
                with open(savehere+f"{file}_{start:04}_{stop:04}_script.txt",'w') as f:  #save script
                    #f.write('\n'.join(script[i]))
                    for tup in script[i]:
                        f.write(' '.join(str(x) for x in tup) + '\n')
                
    df = pd.DataFrame(labels,columns=['Segment','Number of Speakers','Number of Speeches','Number of Speeches per Speaker'])
    # save dataframe of labels as "set split_SARdBScene_speech_labels"
    
    return df,spkcount_hist,starttimes_list,labels

#%% Trying out the annotator code

segment_length_s = 10 #set the desired length of audio segments or chunks
whereisaudio = '/Users/michaelnigro/Documents/Spyder/AMI_headset/'
savehere = '/Users/michaelnigro/Documents/Spyder/SARdBScene_AMIchunks/'  #set correct folder directory
chunk_length_ms = segment_length_s*1000 #10000 # value is in miliseconds. Rounds up when making chunks if not divisible
cs = int(chunk_length_ms/1000) #chunk length in seconds
spkcount_hist = [0,0,0,0,0,0]
starttimes_list=[]
labels=[]
for file in trainy: #ami_files:  
    myaudio = AudioSegment.from_file(whereisaudio+file+'.wav',"wav")  #load one file
    chunks = make_chunks(myaudio, chunk_length_ms) #makes 10s chunks as AudioSegment objects
    print(file)
    num_speakers,script,num_starttimes,whospoke_counts  = xml_parse(file,len(chunks),cs)
    
    for i, chunk in enumerate(chunks):
        start = cs*i  #eg. 0,10,20,30....
        stop = cs+cs*i #eg. 10,20,30,40....
        nspk = num_speakers[i]  #number of speakers in that chunk according to words.xml file 
        nstarts = num_starttimes[i]
        nwhospoke = whospoke_counts[i]
        nwhospoke.sort(reverse=True) #put in descending order (largest counts first)
        
            #nwhospoke=''.join(str(e) for e in nwhospoke) #to get list into string
        if nspk!=0:  #dont save 0 speaker segments
            spkcount_hist[nspk]+=1 #keeping track of the number of data samples per speaker class
            starttimes_list.append(nstarts)
            labels.append([f"{file}_{start:04}_{stop:04}", nspk, nstarts, nwhospoke])
            chunk_name = f"{file}_{start:04}_{stop:04}_{nspk}" #"_{nstarts}_{nwhospoke}"
            #print(chunk_name)
            #normalized_chunk = match_target_amplitude(chunk, -20.0) #normalize to perceived loudness average of -20dbfs
            if chunk.max_dBFS > 0: #checking the max dBFS of each chunk, if greater than 0 it's clipping and not good
                print(chunk_name)


#%% Training set about 24.5h
segment_length_s = 10
df_train,spkcount_train,starts_train, labels_train = annotate_save_data(segment_length_s, trainy, 'train/')  
df_train = df_train.sort_values(by="Segment")     
#%
df_train.to_csv('/Users/michaelnigro/Documents/Spyder/train_SARdBScene_speechlabels.csv',index=False)
#%% Validation set about 7h balanced
df_valid,spkcount_valid,starts_valid, labels_valid = annotate_save_data(segment_length_s, validy, 'valid/')  
df_valid = df_valid.sort_values(by="Segment")     
#%
df_valid.to_csv('/Users/michaelnigro/Documents/Spyder/valid_SARdBScene_speechlabels.csv',index=False)
#%% Test set about 5.6h
df_test,spkcount_test,starts_test, labels_test = annotate_save_data(segment_length_s, testy, 'test/')  
df_test = df_test.sort_values(by="Segment")     
#%
df_test.to_csv('/Users/michaelnigro/Documents/Spyder/test_SARdBScene_speechlabels.csv',index=False)
#%%

#%% 2. Getting RTTM files for speech segments
#ns,words = xml_parse('EN2001a',290,10)                
#ns,words,ss,se = xml_parse('EN2001a',290,10)
import xml.etree.ElementTree as ET
tree = ET.parse('/Users/michaelnigro/Documents/ami_manual_1.6.1/words/ES2002c.A.words.xml')
root = tree.getroot()
lab = root[10].text
print(root[10].attrib['endtime'])
'''
testy = ['IS1009b','EN2002a','IS1009c','TS3003a','EN2002b','TS3003b','EN2002c',
        'TS3003c','EN2002d',
        'TS3003d','ES2004a','TS3007a','ES2004b','TS3007b','ES2004c','TS3007c','ES2004d',
        'TS3007d','ES2014a','ES2014b','ES2014c','ES2014d','IS1009a','IS1009d']

validy = ['ES2003a','ES2003b','ES2003c','ES2003d','ES2011a','ES2011b','ES2011c','ES2011d',
         'IB4001','IB4002','IB4003','IB4004','IB4010','IB4011',
         'IS1008a','IS1008b','IS1008c','IS1008d','TS3004a','TS3004b','TS3004c','TS3004d',
         'TS3006a','TS3006b','TS3006c','TS3006d']

trainy = ['ES2002a','ES2002c','ES2002d',
          'ES2005a','ES2005b','ES2005c','ES2005d',
          'ES2006a', 'ES2006b','ES2006c',
          'ES2007a', 'ES2007b','ES2007c','ES2007d',
          'ES2008a','ES2008b','ES2008d',
          'ES2009a','ES2009b','ES2009c','ES2009d',
          'ES2010a','ES2010b','ES2010c','ES2010d',
          'ES2012a','ES2012b','ES2012c','ES2012d',
          'ES2013a','ES2013b','ES2013c','ES2013d',
          'ES2015a','ES2015b','ES2015c','ES2015d',
          'ES2016a','ES2016b','ES2016c','ES2016d',
          'IS1000a','IS1000b','IS1000c','IS1000d',
          'IS1001a','IS1001b','IS1001c','IS1001d',
          'IS1002b','IS1002c','IS1002d',
          'IS1003a','IS1003b','IS1003c','IS1003d',
          'IS1004a','IS1004b','IS1004c','IS1004d',
          'IS1005a','IS1005b','IS1005c',
          'IS1006a','IS1006b','IS1006c','IS1006d',
          'IS1007a','IS1007b','IS1007c','IS1007d',
          'TS3005a','TS3005b','TS3005c','TS3005d',
          'TS3008a','TS3008b','TS3008c','TS3008d',
          'TS3009a','TS3009b','TS3009c','TS3009d',
          'TS3010a','TS3010b','TS3010c','TS3010d',
          'TS3011a','TS3011b','TS3011c','TS3011d',
          'TS3012a','TS3012b','TS3012c','TS3012d',
          'EN2001a','EN2001b','EN2001d','EN2001e',
          'EN2004a', 'EN2005a','EN2006a','EN2006b','EN2009d',
          'IN1002', 'IN1005', 'IN1007', 'IN1008',
          'IN1009', 'IN1012', 'IN1013', 'IN1014', 'IN1016']
'''
def xml_rttm(name,typ):
    import glob
    import xml.etree.ElementTree as ET
    import numpy as np
    import pandas as pd
    nonos = [',','.','?','!','uh','um','mm','ah','huh','hmm','uh-huh','mm-hmm','uh-uh','mm-mm'
             'Uh','Um','Mm','Ah','Huh','Hmm','Uh-huh','Uh-Huh','Mm-hmm','Mm-Hmm','Uh-Uh','Uh-uh','Mm-mm','Mm-Mm']
    filenames = sorted(glob.glob('/Users/michaelnigro/Documents/ami_manual_1.6.1/words/'+name+'*.xml'))

    speechstarts= []
    speechends = []
    totalwords = []
    d=[]
    for i,file in enumerate(filenames):
        tree = ET.parse(file)  #set path to xml file
        root = tree.getroot()
        starttime=[]
        endtime = []
        words=[]
        spk = file[-11] 
        for child in range(len(root)):  #iterate over each line in xml file 'root'
            #print(child.tag, child.attrib)
            label = root[child].attrib  #gets 'labels' of each line. A line indicates a single word
            
            if len(label)==3:  # condition met for line containing text
                w = root[child].text
                if (type(w) is str) and (any(x in w for x in nonos)==False): #check its a word said and not punctuation    
                    words.append(root[child].text)
                    starttime.append(float(label['starttime']))  #get numeric value of start end times for a spoken word
                    endtime.append(float(label['endtime']))
                    start = float(label['starttime'])
                    duration = round(float(label['endtime']) - float(label['starttime']),2)
                    st = int(np.floor(start/10)*10) #37.../10
                    en = int(st+10)
                    fname = f'{typ}_{name}_{st:04}_{en:04}'
                    d.append(['SPEAKER',fname,1,start,duration,'<NA>','<NA>',spk,'<NA>','<NA>'])
                    #d.append(f'SPEAKER {name} 1 {start} {duration} <NA> <NA> {spk} <NA> <NA>')
                    #with open('testAMI.rttm') as d:
                    #    d.write(f'SPEAKER {name} 1 {start} {duration} <NA> <NA> {spk} <NA> <NA>')
                df = pd.DataFrame(d,columns=['Type','ID','Channel ID','Turn Onset','Turn Duration','Ortho','Spk type','Spk name','Score','Lookahead'])
                df = df.sort_values(by='Turn Onset')
    return df
#%%
d1 = xml_rttm('EN2001a','train')
#%%
import pandas as pd

for i,f in enumerate(trainy):
    print(f)
    d = xml_rttm(f,'train') #returns sorted dataframe as RTTM format
    if i==0:
        df = d
    if i>0:
        df = pd.concat([df,d],ignore_index=True)


#%%
df.to_csv('/Users/michaelnigro/Documents/Spyder/trainAMI.rttm',sep=' ',index=False,header=False)


#%%
for i,f in enumerate(validy):
    d = xml_rttm(f,'valid') #returns sorted dataframe as RTTM format
    if i==0:
        dftr = d
    if i>0:
        dftr = pd.concat([dftr,d],ignore_index=True)
#%%
dftr.to_csv('/Users/michaelnigro/Documents/Spyder/validAMI.rttm',sep=' ',index=False,header=False)

#%%
for i,f in enumerate(testy):
    print(f)
    d = xml_rttm(f,'test') #returns sorted dataframe as RTTM format
    if i==0:
        dftr = d
    if i>0:
        dftr = pd.concat([dftr,d],ignore_index=True)
#%%
dftr.to_csv('/Users/michaelnigro/Documents/Spyder/testAMI.rttm',sep=' ',index=False,header=False)

