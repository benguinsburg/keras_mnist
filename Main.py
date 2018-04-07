import tensorflow as tf
import numpy as np
import random

# Image manipulation.
import PIL.Image
from PIL import ImageTk
import tkinter as tk

import matplotlib.pyplot as plt


import keras
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, \
                            BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model, model_from_json

from keras import optimizers


keras.backend.set_image_data_format('channels_last')
keras.backend.set_learning_phase(1)


################################################################################################################
                                    #Helper Functions#
################################################################################################################

def load_image(filename):
    image = PIL.Image.open(filename)

    return np.float32(image)


def save_image(image, filename):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)

    # Convert to bytes.
    image = image.astype(np.uint8)

    # Write the image-file in jpeg-format.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')


def plot_image(image):
    # Assume the pixel-values are scaled between 0 and 255.

    if False:
        # Convert the pixel-values to the range between 0.0 and 1.0
        image = np.clip(image / 255.0, 0.0, 1.0)

        # Plot using matplotlib.
        plt.imshow(image, interpolation='lanczos')
        plt.show()
    else:
        # Ensure the pixel-values are between 0 and 255.
        image = np.clip(image, 0.0, 255.0)

        # Convert pixels to bytes.
        image = image.astype(np.uint8)

        # Convert to a PIL-image and display it.
        image = PIL.Image.fromarray(image)
        image.show()
        # display(image)


def display(img):
    root = tk.Tk()
    tkimage = ImageTk.PhotoImage(img)
    tk.Label(root, image=tkimage).pack()
    root.mainloop()

def normalize_image(x):
    # Get the min and max values for all pixels in the input.
    x_min = x.min()
    x_max = x.max()

    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm


def resize_image(image, size=None, factor=None):
    # If a rescaling-factor is provided then use it.
    if factor is not None:
        # Scale the numpy array's shape for height and width.
        size = np.array(image.shape[0:2]) * factor

        # The size is floating-point because it was scaled.
        # PIL requires the size to be integers.
        size = size.astype(int)
    else:
        # Ensure the size has length 2.
        size = size[0:2]

    # The height and width is reversed in numpy vs. PIL.
    size = tuple(reversed(size))

    # Ensure the pixel-values are between 0 and 255.
    img = np.clip(image, 0.0, 255.0)

    # Convert the pixels to 8-bit bytes.
    img = img.astype(np.uint8)

    # Create PIL-object from numpy array.
    img = PIL.Image.fromarray(img)

    # Resize the image.
    img_resized = img.resize(size, PIL.Image.LANCZOS)

    # Convert 8-bit pixel values back to floating-point.
    img_resized = np.float32(img_resized)

    return img_resized

from itertools import compress
class MaskableList(list):
    def __getitem__(self, index):
        try: return super(MaskableList, self).__getitem__(index)
        except TypeError: return MaskableList(compress(self, index))

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def load_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model


################################################################################################################
                                    #Model Definitions#
################################################################################################################

def AlphaModel(input_shape, classes):
    """
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    X_input = Input(input_shape)

    X = Flatten()(X_input)

    X = Dense(512, activation='relu',
              kernel_initializer='TruncatedNormal')(X)
    X = Dropout(0.1)(X)

    X = Dense(256, activation='relu',
              kernel_initializer='TruncatedNormal')(X)
    X = Dropout(0.2)(X)

    X = Dense(64, activation='relu',
              kernel_initializer='TruncatedNormal')(X)
    X = Dropout(0.3)(X)

    X = Dense(128, activation='relu',
              kernel_initializer='TruncatedNormal')(X)
    X = Dropout(0.5)(X)

    X = Dense(classes, activation='softmax',
              kernel_initializer='TruncatedNormal')(X)

    model = Model(inputs=X_input, outputs=X, name='AlphaModel')

    return model

################################################################################################################
                                    #Session#
################################################################################################################

from keras.datasets import mnist

num_classes = 10
epochs = 20
batch_size = 512
load_flag = False

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)




if load_flag:
    model = load_model()
else:
    model = AlphaModel(x_train.shape[1:], classes=num_classes)

model.compile(optimizer=optimizers.Adamax(), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x = x_train, y = y_train, epochs = epochs, batch_size = batch_size)


print("FINISHED FITTING")

preds = model.evaluate(x_test, y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


save_model(model)



