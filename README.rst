`Animal sound recognition using deep learning techniques`
=========================================================

This is a project for the `second year master's degree (TIDE) <https://formations.pantheonsorbonne.fr/fr/catalogue-des-formations/master-M/master-econometrie-statistiques-KBURDRPJ//master-parcours-traitement-de-l-information-et-data-science-en-entreprise-tide-formation-initiale-et-apprentissage-KBUREJV4.html>`_ at the `Paris I Panthéon-Sorbonne University <https://www.pantheonsorbonne.fr/>`_, Deep Learning course.

This repo contains code in Python for an application of the sound recognition techniques from this paper: `PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition <https://ieeexplore.ieee.org/document/9229505>`_ [1]_ on animal sound recognition.
The database used for this is Google Audioset, a big dataset of classified audio, from the Youtube-8M project, containing ”632 audio event classes and a collection of 2,084,320 human-labeled 10-second sound clips drawn from YouTube videos” (`see <https://ieeexplore.ieee.org/abstract/document/7952261>`_ [2]_).
The idea was to apply the technique used in `this repository <https://github.com/qiuqiangkong/audioset_tagging_cnn>`_ to animal sound recognition and observe the results.  

1. Download dataset
-------------------

The dataset used for this project contains all classes from the Google AudioSet that describe animal sounds. 
Sounds were then packed to hdf5 format.
All data were not included in this repo due to their important size.

2. Train
--------

Three models were trained: CNN14, ResNet38 and Wavegram-Logmel-CNN.

3. Results
----------

We have managed to show that models that are exclusively trained on animal sounds data provide better results than general purpose models. 
The best sound prediction model obtained so far on the AudioSet database, the Wavegram-Logmel-CNN, with a mean average precision (mAP) of 0.439, has been surpassed by our ResNet38 and CNN14 models trained on animal sounds with a mAP of 0.551 and 0.561 respectively.

.. image:: /visualisations/four_figures_final.PNG

Understandably, these models are trained on different data and their results may not be compared withease, especially the mAP, as showed in the class-wise analysis.

.. image:: /visualisations/CNN14_classwise_results.PNG

There is a structural class effect that needs to betaken into account,  as some types of sounds are more complex to classify than others,  which is unknown at the beginning.  But, we have showed that the model which is best for general purpose training will not always be thesame that that which will be applied to a specific group of data from the same source.

Future  research  may  focus  on  confirming  or  denying  the  above  hypothesis  we  have  made  on  training different groups compared to aggregates.  Tuning the parameters and discovering the ones that fit well for this kind of problems would also be important. Other models may be tested, such as the MobileNetV1, that due to it’s lighter network may be possible to incorporate in a smartphone application.

Environments/Dependencies
-------------------------

The codebase is developed with Python 3.7. Install requirements as follows::

pip install -r requirements.txt

References
----------

.. [1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." (2019).

.. [2] Gemmeke et al., "Audio Set: An ontology and human-labeled dataset for audio events," 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), New Orleans, LA, USA, 2017, pp. 776-780, doi: 10.1109/ICASSP.2017.7952261.
