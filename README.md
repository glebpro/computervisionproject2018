# BirdNet
Term project for CSCI 431 _Introduction to Computer Vision_ at RIT, 2018.

Gregory Goh [@ShinyTeeth](https://github.com/ShinyTeeth)
Gleb Promokhov [@glebpro](https://github.com/glebpro)

## Abstract:
This project explores how convolutional neural nets can be used for image classification tasks. From a public dataset of 11,788 images of birds belonging to 200 species classes, we built and trained a CNN classifier with comparable classification accuracy. We trained several identical models using the raw images, images with the birds segmented out, and images with just the bird's head to examine how reducing the input data dimensions affect classification accuracy.


[[Slides](slides.pdf)][[Paper](paper.pdf)]

## Technicals

#### Downloads
To download code: `$ git clone https://github.com/glebpro/computervisionproject2018.git`

To download images, source and segmentations: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
Unzip to this repo folder into `data/CUB_200_2011` and `data/segmentations`.

#### Scripts

`$ python scripts/preprocess.py` will confirm you have the correct project file structure, and apply the image segmentations, split images into training/validation/testing folders into `/data/split_segmented_head_images`

`$ python scripts/BirdNetClassifer.py` will generate a model, evaluate it, and save it into [/models](/models)

#### Requirements
[python](https://www.python.org/) >= 3.4, [matplotlib](https://matplotlib.org/) >= 2.2.2, [opencv-python](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html) >= 3.4.0.12, [Keras](https://keras.io/) >= 2.0.9, [ipython](https://ipython.org/) >= 6.3.0, [numpy](http://www.numpy.org/) >=1.14.2, [vis](http://vispy.org/) >= 0.0.5a1, [keras_vis](https://github.com/raghakot/keras-vis) >= 0.4.1

`$ pip install -r requirements.txt`

#### License
MIT licensed. See the bundled [LICENSE](/LICENSE) file for more details.
