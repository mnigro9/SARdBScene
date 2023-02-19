# Signal Analysis Research group's database of Source Counting and audio ENvironment Evaluation (SARdBScene) 

The audio scene analysis dataset SARdBScene is developed for audio source counting research as well as for other audio-based tasks for comprehensive audio SARdBScene is developed to promote research for audio source counting and present a comprehensive audio scene analysis dataset that covers a variety of scenarios and audio-based tasks. It contains 80 hours of synthetic audio scene mixtures depicting four distinct environments/scenes (office, home, nature, and urban) with detailed annotations that make it a unique collection of curated data in the audio analysis landscape.
The dataset can be used for audio source counting tasks mainly, but also for acoustic scene classification, sound event classification/detection, and speaker diarization.

# Download

The dataset can be downloaded from here .

The audio data folder structure is as follows:
SARdBScene/
- test/
- train/
- valid/
-	valid_urban/
-		valid_urban_1000_8.wav ... continued for other audio files in the data split
-	valid_nature/
-	valid_office/
-	valid_home/
-	valid_urban_1000_8.txt
-	valid_urban_1000_8.jams ... continued .txt and .jams files for remaining audio files in data split
– SARdBScene folder contains all the audio mixtures for the dataset organized by data split partition and acoustic scene class
– file name: ‘split_scene_filenumber_sourcecount.wav’ for {‘split’: ’train’, ‘valid’, ’test’}	{‘scene’: ‘office’, ‘home’, ‘urban’, ‘nature’}
– .txt and .jams files are the files produced by Scaper [1] when making audio mixtures. Note they do not contain annotations of the ‘speech’ sound event.

## Data Annotation files
Folder SARdBScene_annotations
{‘split’: ’train’, ‘valid’, ’test’}	{‘scene’: ‘office’, ‘home’, ‘urban’, ‘nature’}

— ‘split_scene.csv’ files contain the SARdBScene filename, original source files used in data mixing process, scene class, sound event classes, source counts, speaker counts, and sound effects counts.

Speech centric annotations: making use of these requires some extra steps to cross-reference the speaker recordings with the final SARdBScene mixtures. Using the ‘split_scene.csv’ files to determine what speaker recording is in each SARdBScene mixture since we didn’t update the file naming 

- testAMI.rttm, validAMI.rttm, and trainAMI.rttm files correspond to speaker annotation (typical format for speaker diarization); audio segment naming corresponds with the meeting ID names from the AMI corpus and the 10 s time boundary in seconds (ex. first row of ‘testAMI.rttm’ is:
SPEAKER test_IS1009b_0030_0040 1 37.32 0.47 <NA> <NA> A <NA> <NA>
where ‘test_IS1009b_0030_0040’ corresponds with AMI meeting ‘IS1009b’ for the 30-40 s interval; the third last column indicates a speaker ID ‘A’; columns 4 and 5 indicate a segment of speaker ‘A’ starting at 37.32 s for 0.47s duration. This corresponds with the original AMI corpus meeting recordings, column 4 can be normalized for the segment interval (i.e. in this case for the 10 s interval from 30-40 s (normalized as 0-10 s) the speaker A start time of 37.32 s would normalize to 7.32 s)

- test_SARdBScene_speechlabels.csv, valid_SARdBScene_speechlabels.csv, and train_SARdBScene_speechlabels.csv contain additional annotations for each 10 s segment related to speaker counting: the number of speakers, the total number of continuous speeches, and the number of speeches per speaker in a segment


# Baseline
The baseline system for audio source counting presented in ICASSP2023 paper follows a ResNet architecture using log mel spectrogram as input features.

Files for baseline (feature extraction and ResNet models)
- sardbscene_feature_extraction.py
— - file for generating mel spectrogram features for the dataset. organized according to the data split (train/valid/test) and scene class

- SARdBScene_baseline.pynb
- - notebook performing model training and evaluation for audio source counting and speaker counting in the different scenes.


Folder ‘SARdBScene_features’ containing features and labels from running sardbscene_feature_extraction.py:
the corresponding label files and spectrogram files are in order following the ‘SARdBScene_annotations’ order; keep this in mind before any shuffling of the data
- split_scene_melspec.npy files contain log mel spectrograms for the corresponding data split and acoustic scene; data used in baseline results
- split_scene_sfxc_labels.npy contains source counting labels for sound effects (every sound type but speech) as sound effect counting
- split_scene_spkc_labels.npy contains speaker counting labels
- split_scene_sc_labels.npy contains the overall source counting labels (speech and non-speech sound types, speech is treated as a singular class regardless of the number of speakers)

# Citing
If using the SARdBScene dataset or any of this repository please cite our paper:

M. Nigro and S. Krishnan, “SARdBScene: Dataset and ResNet
Baseline for Audio Scene Source Counting and Analysis,” in
ICASSP 2023 - 2023 IEEE International Conference on Acous-
tics, Speech and Signal Processing (ICASSP), 2023 (forthcom-
ing).

As well, be sure to consider the references provided below since SARdBScene makes use of these works and materials.

# References

[1] Justin Salamon, Duncan MacConnell, Mark Cartwright, Peter Li, and Juan Pablo Bello, “Scaper: A library for soundscape synthesis and augmentation,” in 2017 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). Oct. 2017, pp. 344–348, IEEE.

[2] Frederic Font, Gerard Roma, and Xavier Serra, “Freesound technical demo,” in Proceedings of the 21st ACM international conference on Multimedia, New York, NY, USA, Oct. 2013, MM ’13, pp. 411–412, ACM.

[3] Nicolas Turpault, Romain Serizel, Justin Salamon, and Ankit Parag Shah, “Sound Event Detection in Domestic Environments with Weakly Labeled Data and Soundscape Synthesis,” in Proceedings of the Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE2019). 2019, pp. 253–257, New York University.

[4] Romain Serizel, Nicolas Turpault, Ankit Shah, and Justin Salamon, “Sound Event Detection in Synthetic Domestic Environments,” in ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). May 2020, pp. 86–90, IEEE.

[5] Justin Salamon; Christopher Jacoby; Juan Pablo Bello, “A Dataset and Taxonomy for Urban Sound Research,” in Proceedings of the 22nd ACM International Conference on Multimedia, 2014, pp. 1041–1044.

[6] Jean Carletta, “Unleashing the killer corpus: experiences in creating the multi-everything AMI Meeting Corpus,” Language Resources and Evaluation, vol. 41, no. 2, pp. 181–190, Nov. 2007.

[7] Karol J. Piczak, “ESC: Dataset for Environmental Sound Classification,” in Proceedings of the 23rd ACM international conference on Multimedia, New York, NY, USA, Oct. 2015, pp. 1015–1018, ACM.
