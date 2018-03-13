# BirdNet
Term project for CSCI 431 _Introduction to Computer Vision_ at RIT, 2018.

Gregory Goh [@ShinyTeeth](https://github.com/ShinyTeeth)
Gleb Promokhov [@glebpro](https://github.com/glebpro)

## Abstract:
Experiments in classifying birds.

[[Slides](slides.pdf)][[Paper](paper.pdf)]

## Technicals

#### Downloads
To download code: `$ git clone https://github.com/glebpro/computervision2018`

To download images, source and segmentations: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
Unzip to this repo folder into `data/CUB_200_2011` and `data/segmentations`.

#### Scripts

`$ python scripts/preprocess.py` will confirm you have the correct project file structure, and apply the image segmentations, split images into training/validation/testing folders into `/data/split_segmented_images`

`$ python scripts/BirdNetClassifer.py` will generate a model, evaluate it, and save it into [/models](/models)

#### Requirements
[python](https://www.python.org/) >= 3.4, [matplotlib](https://matplotlib.org/) >= 2.0.2, [opencv-python](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html) >= 3.4.0.12, [Keras](https://keras.io/) >= 2.0.9

#### License
MIT licensed. See the bundled [LICENSE](/LICENSE) file for more details.
