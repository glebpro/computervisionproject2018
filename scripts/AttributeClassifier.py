#
#   A classifier of bird attributes.
#
#   @author Greg Goh
#

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from preprocess import load_attributes, image_classes

def main():
    attr_name2id, cert_name2id, img_attr_labels, img_attr_dict = load_attributes()
    clss_name2id, img_clss_labels = image_classes()

    clss_id2name = dict(reversed(item) for item in clss_name2id.items())

    class_ids = clss_id2name.keys()
    onehots = np_utils.to_categorical(class_ids)

    train_data = np.zeros((11789, 313))
    for v in img_attr_labels:
        train_data[int(v[0])][int(v[1])] = int(v[2]) * int(v[3])
    
    # print(train_data[1])

    # model = Sequential()
    # model.add(Dense(8, input_dim=313, activation='relu'))
    # model.add(Dense(200, activation='softmax'))

    # model.compile(loss='categorical_crossentropy',
    #                     optimizer='sgd', # Stochastic gradient descent
    #                     metrics=['accuracy'])

    # model.fit(x_train, y_train,
    #       epochs=20,
    #       batch_size=128)
    # score = model.evaluate(x_test, y_test, batch_size=128)
    # return score

if __name__ == '__main__':
    main()