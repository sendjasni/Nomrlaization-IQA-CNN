## Motivation and context

As no defined rules allows to determine an image quality by a score, assigning a quality score to a set of images is very tedious the way it is done nowadays because it needs polling people and meaning their quality opinion score (MOS). Furthermore, as it actually depends on human perception, visual conditions such as ambiant lighting condition, screen quality, gamma, colors, light reflection and so on, are numerous and difficult to keep consistent, particularly if the scoring is done in different rooms in parallel. More difficulties to keep results consistent with such subjective score studies on this paper, in the context of audio sample quality [[1]](https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-P.1401-202001-I!!PDF-E&type=items)
    
This is why training Deep Neural Networks to predict a quality score is very interesting. Finally, as confusion exists on whether or not images should be normalized or preprocess in a certain way before training, this project aims to clarify this aspect and test easily normalisation and training performances.

## Description

This repo contains the code used for the Investigating normalization methods for *CNN-based image quality assessment* 
(ICIP-2022) paper.

With this code, you will be able to :
- Train on IQA (Image Quality Assesment) databases used in the paper (LIVE Release2, CSIQ, TID2013)
- Separate automatically in approximatically  80%/20% for training dataset/test dataset, in respect to the constraint 
  not to have distorted images from the same pristine image in different datasets.
- Use preprocessing normalization on training dataset and test dataset automatically

## Pre-requistics

This repo does not provide any database for memory concerns. To work as intended, a folder "database/" should be created
at the root of the project. training/distorded images should be in database/dst_imgs. An .csv file should also be in
database/, to link each img name with their own *Mean Opinion Score* (MOS).

## How to use

Database should always be transformed in .npy binary files format, this is done by assigning some variables in 
`constants.py`
```python
RUN = 'preprocess'
PREPROCESS = 'nothing'
```

Then, without changing the database:
```python
RUN = 'train_from_npy'
PREPROCESS = 'scale_0_1'
``` 

## Functionning of code and Google Colab use

Google Colab is well-supported with the code, this is the reason why the entry-point to run any code is only one file:

In Colab environnement, this entry-point file would be a ``.ipynb`` file,  with these lines at the beginning of the file, wrapping the 
constants 
```
from dataclasses import dataclass
 
@dataclass
class Constants_colab:

   [...]
   
cst = Constants_colab()
```

In local environnement, the entry-point would be a ``.py`` file, with 
```
import constants as cst
```

This allows in either way to give constants as an argument ``cst`` to the script to execute, and to access constants 
with python-object-alike syntax using ``cst.[constant_variable_name]``

Don't forget to modify path variables in the ``constants`` file or dataclass.

## Authors
* Abderrezzaq Sendjasni, Ph.D.   [(abderrezzaq.sendjasni@univ-poitiers.fr)](mailto:abderrezzaq.sendjasni@univ-poitiers.fr)
* David Traparic, Ph.D.   
* Chaker Larabi, Prof.   [(chaker.larabi@univ-poitiers.fr)](mailto:chaker.larabi@univ-poitiers.fr)

## How to cite:
```
@INPROCEEDINGS{9897268,
  author={Sendjasni, Abderrezzaq and Traparic, David and Larabi, Mohamed-Chaker},
  booktitle={IEEE International Conference on Image Processing (ICIP)}, 
  title={Investigating Normalization Methods for CNN-Based Image Quality Assessment}, 
  year={2022},
  volume={},
  number={},
  pages={4113-4117},
  doi={10.1109/ICIP46576.2022.9897268}
}
```
