from service.generic_service import GenericService
from keras.layers import Dense, Input
from keras.models import Model
import numpy as np


class Service(GenericService):
    def __init__(self):
        GenericService.__init__(self)

        data = Input([4608])
        X = Dense(4096)(data)
        X = Dense(4096)(X)
        X = Dense(1000)(X)
        self.model = Model(data, X)

    def predict(self, input):
        return self.model.predict(np.array([input]))

    def send(self, output):
        pass

    def __repr__(self):
        return 'alexnet.demo.fc'
