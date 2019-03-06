from custom_layers import SplitTensor, LRN2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Concatenate, Activation, Dense, Dropout, Flatten, Input
from keras.models import Model
from keras import backend as K
import sys


def alexnet(weights_path=None):
    K.set_image_data_format('channels_first')
    inputs = Input([3, 227, 227])

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
    model.summary()
    if weights_path is not None:
        model.load_weights(weights_path)
    return model


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Using default setting, no-weight.'
        model = alexnet()
    else:
        model = alexnet(sys.argv[1])

    

