#
#   A classifier of birds.
#
#   @author Gleb Promokhov
#

import os
import pathlib
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__+"/.."))

class BirdNetClassifer(object):
    """
    A classifier of birds.
    """

    def __init__(self, classes, labels, image_paths):

        self._train(classes, labels, image_paths)


    def _train(self, classes, labels, image_paths):
        


def main():

    image_data_path = PROJECT_ROOT + "/data/CUB_200_2011/"
    segmented_data_path = PROJECT_ROOT + "/data/segmented_images"

    # get class labels vector
    classes = open(image_data_path + "classes.txt").readlines()
    classes = [r.split()[1].strip() for r in classes]

    # get source image classes
    labels = open(image_data_path + "image_class_labels.txt").readlines()
    labels = [int(r.split()[1].strip())-1 for r in labels]

    # load segmented image paths
    image_paths = []
    for filepath in pathlib.Path(segmented_data_path).glob('**/*'):
        image_paths.append(filepath.absolute())

    bnc = BirdNetClassifer(classes, labels, image_paths)


if __name__ == "__main__":
    main()
