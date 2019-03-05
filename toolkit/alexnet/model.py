from custom_layers import SplitTensor, LRN2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Concatenate, Activation, Dense, Dropout, Flatten, Input
from keras.models import Model

from keras import backend as K
from keras.utils.conv_utils import convert_kernel
import tensorflow as tf



def alexnet(weights_path=None):
    inputs = Input([227, 227, 3])

    X = Conv2D(96, (11, 11), strides=(4, 4), activation='relu')(inputs)

    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = LRN2D()(X)
    X = ZeroPadding2D((2, 2))(X)
    X = Concatenate(axis=1)(
        [Conv2D(128, (5, 5), activation='relu')(SplitTensor(ratio_split=2, id_split=i)(X)) for i in range(2)])

    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = LRN2D()(X)
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(384, (3, 3), activation='relu')(X)

    X = ZeroPadding2D((1, 1))(X)
    X = Concatenate(axis=1)(
        [Conv2D(192, (3, 3), activation='relu')(SplitTensor(ratio_split=2, id_split=i)(X)) for i in range(2)])

    X = ZeroPadding2D((1, 1))(X)
    X = Concatenate(axis=1)(
        [Conv2D(128, (3, 3), activation='relu')(SplitTensor(ratio_split=2, id_split=i)(X)) for i in range(2)])

    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = Flatten()(X)
    X = Dense(4096, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(4096, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(1000)(X)
    prediction = Activation('softmax')(X)

    model = Model(input=inputs, output=prediction)
    if weights_path is not None:
        model.load_weights(weights_path)
    return model

def convert_weights(model, weights_path, outputFileName='alex_tensorflow.h5'):
    # convert weights from theano format to tensorflow format
    ops = []
    for layer in model.layers:
        if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
            original_w = K.get_value(layer.W)
            converted_w = convert_kernel(original_w)
            ops.append(tf.assign(layer.W, converted_w).op)
    K.get_session().run(ops)
    model.save_weights(outputFileName)

if __name__ == '__main__':
    model = alexnet()
    weights_path = 'alexnet_theano.h5'
    convert_weights(model,weights_path)
