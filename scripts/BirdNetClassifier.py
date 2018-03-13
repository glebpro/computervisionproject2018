#
#   A classifier of birds.
#
#   @author Gleb Promokhov
#

import os
import pathlib
import datetime

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__+"/.."))

class BirdNetClassifer(object):
    """
    A classifier of birds.
    """

    def __init__(self, classes, training_images_dir, validation_images_dir):

        # constants
        self.classes = classes
        self.IMG_WIDTH = 150
        self.IMG_HEIGHT = 150
        self.EPOCHS = 1
        self.BATCH_SIZE = 16

        self.N_TRAINING = 0
        self.N_VALIDATION = 0
        for filepath in pathlib.Path(training_images_dir).glob('**/*'):
            self.N_TRAINING+=1
        for filepath in pathlib.Path(validation_images_dir).glob('**/*'):
            self.N_VALIDATION+=1

        # blank model
        self.model = self._make_blank_model()
        # self._train(training_images_dir, validation_images_dir)

    def load(self, load_path):
        self.model.load_weights(load_path)

    def save(self, save_path):
        self.model.save_weights(save_path)

    def train(self, training_images_dir, validation_images_dir):
        """
        Fit model with images.
        """
        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
                            rescale=1. / 255,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
                            training_images_dir,
                            target_size=(self.IMG_WIDTH, self.IMG_HEIGHT),
                            batch_size=self.BATCH_SIZE,
                            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
                                validation_images_dir,
                                target_size=(self.IMG_WIDTH, self.IMG_HEIGHT),
                                batch_size=self.BATCH_SIZE,
                                class_mode='categorical')

        self.model.fit_generator(
            train_generator,
            steps_per_epoch=self.N_TRAINING // self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_data=validation_generator,
            validation_steps=self.N_VALIDATION // self.BATCH_SIZE)

    def evaluate(self, testing_images_dir):
        """
        Evaluate model from testing images.
        """
        datagen = ImageDataGenerator(rescale=1. / 255)

        eval_generator = datagen.flow_from_directory(
                            testing_images_dir,
                            target_size=(self.IMG_WIDTH, self.IMG_HEIGHT),
                            batch_size=self.BATCH_SIZE)

        stats = self.model.evaluate_generator(eval_generator, steps=len(eval_generator))

        print(stats)

    def _make_blank_model(self):
        """
        Instantiate a blank model.
        """

        input_shape = ()

        if K.image_data_format() == 'channels_first':
            input_shape = (3, self.IMG_WIDTH, self.IMG_HEIGHT)
        else:
            input_shape = (self.IMG_WIDTH, self.IMG_HEIGHT, 3)

        # define layers of model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.classes)))
        model.add(Activation('sigmoid'))

        model.compile(loss='categorical_crossentropy',
                        optimizer='sgd', # Stochastic gradient descent
                        metrics=['accuracy'])

        return model


def main():

    # define training/validation/testing images directories
    training_images_dir = PROJECT_ROOT + "/data/split_segmented_images/train"
    validation_images_dir = PROJECT_ROOT + "/data/split_segmented_images/validation"
    testing_images_dir = PROJECT_ROOT + "/data/split_segmented_images/test"

    # get class labels vector
    classes = open(PROJECT_ROOT + "/data/CUB_200_2011/classes.txt").readlines()
    classes = [r.split()[1].strip() for r in classes]

    # build+save classifier
    bnc = BirdNetClassifer(classes, training_images_dir, validation_images_dir)
    bnc.train(training_images_dir, validation_images_dir)
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    bnc.save('models/BirdNetModel_%s.h5' % timestamp)
    # bnc.load(PROJECT_ROOT+"/models/BirdNetModel_13-03-2018_17:16:24.h5")

    bnc.evaluate(testing_images_dir)

if __name__ == "__main__":
    main()
