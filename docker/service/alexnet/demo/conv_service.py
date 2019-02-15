from service.generic_service import GenericService
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input
from keras.models import Model
import numpy as np


class Service(GenericService):
    def __init__(self):
        GenericService.__init__(self)

        data = Input([220, 220, 3])

        X = Conv2D(48, (11, 11), strides=(4, 4))(data)
        X = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(X)

        X = Conv2D(128, (5, 5))(X)
        X = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(X)

        X = Conv2D(192, (3, 3))(X)

        X = Conv2D(192, (3, 3))(X)
        X = Conv2D(192, (3, 3))(X)
        X = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(X)

        X = Flatten()(X)
        self.model = Model(data, X)

    def predict(self, input):
        return self.model.predict(np.array([input]))

    def send(self, output):
        pass

    def __repr__(self):
        return 'alexnet.demo.conv'
