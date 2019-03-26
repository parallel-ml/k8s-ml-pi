"""
Bounding box service takes care of model inference after darknet-53.
"""
from service.generic_service import GenericService
from service.yolo.util import load_yolo_model
from keras.layers import Input
import os
import numpy as np

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)


class Service(GenericService):
    def __init__(self):
        GenericService.__init__(self)
        self.model = load_yolo_model((154, 252), {'add_11': Input([40, 40, 256]),
                                                  'add_19': Input([20, 20, 512])})

    def predict(self, input):
        input = [self.to_numpy(input[0], [1, 20, 20, 512]), self.to_numpy(input[1], [1, 20, 20, 512]),
                 self.to_numpy(input[2], [1, 40, 40, 256])]
        results = self.model.predict(input)
        for i in range(len(results)):
            results[i] = results[i].tobytes()
        return results

    def simulate(self, size):
        return np.array([np.random.random_sample(size)])

    def to_numpy(selfs, bytes, size):
        return np.fromstring(bytes, np.float32).reshape(size)

    def send(self, output):
        return output

    def __repr__(self):
        return 'yolo.demo.bb-service'
