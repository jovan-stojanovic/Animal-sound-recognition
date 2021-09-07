# Animal sound recognition using deep learning techniques

This repo contains code in Python for an application of the sound recognition techniques from this paper: **PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition** [1] on animal sound recognition.
The database used for this is Google Audioset, a big dataset of classified audio, from the Youtube-8M project, containing ”632 audio event classes and a collection of 2,084,320 human-labeled 10-second sound clips drawn from YouTube videos” (see [2]).
The idea was to apply the technique used in https://github.com/qiuqiangkong/audioset_tagging_cnn to animal sound recognition and observe the results.  

## Environments
The codebase is developed with Python 3.7. Install requirements as follows:
```
pip install -r requirements.txt
```

## 1. Download dataset
The dataset used for this project contains all classes from the Google AudioSet that describe animal sounds. The audios were downloaded from Youtube using the technique presented in
https://github.com/qiuqiangkong/audioset_tagging_cnn. Sounds were then packed to hdf5 format as in the general sound recognition project.
All data were not included in this repo due to their important size.

## 2. Train
Three models were trained: CNN14, ResNet38 and Wavegram-Logmel-CNN.

## Results
We have managed to show that models that are exclusively trained on animal sounds data provide better results than general purpose models. 
The best sound prediction model obtained so far on the AudioSet database, the Wavegram-Logmel-CNN, with a mean average precision (mAP) of 0.439, has been surpassed by our ResNet38 and CNN14 models trained on animal sounds with a mAP of 0.551 and 0.561 respectively.

Understandably, these models are trained on different data and their results may not be compared withease, especially the mAP, as showed in the class-wise analysis.  There is a structural class effect that needs to betaken into account,  as some types of sounds are more complex to classify than others,  which is unknown at thebeginning.  But, we have showed that the model which is best for general purpose training will not always be thesame that that which will be applied to a specific group of data from the same source.

Future  research  may  focus  on  confirming  or  denying  the  above  hypothesis  we  have  made  on  trainingdifferent groups compared to aggregates.  Tuning the parameters and discovering the ones that fit well for this kindof problems would also be important.  Other models may be tested, such as the MobileNetV1, that due to it’s lighter network may be possible to incorporate in a smartphone application.

## Reference
[1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." arXiv preprint arXiv:1912.10211 (2019).
[2] https://research.google.com/audioset/index.html
