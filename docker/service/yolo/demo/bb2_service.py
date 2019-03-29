"""
Bounding box service takes care of model inference after darknet-53.
"""
from service.generic_service import GenericService
from service.yolo.util import load_yolo_model
import service.yolo.util as model_util
from keras.layers import Input
import numpy as np


class Service(GenericService):
    def __init__(self):
        GenericService.__init__(self)
        load_yolo_model([189, 252], {'add_11': Input([40, 40, 256]), 'add_19': Input([20, 20, 512])})
        self.model = model_util.model
        self.graph = model_util.graph

    def predict(self, input):
        data = [self.to_numpy(input[0], [1, 10, 10, 512]), self.to_numpy(input[1], [1, 20, 20, 512]),
                self.to_numpy(input[2], [1, 40, 40, 256])]
        with self.graph.as_default():
            results = self.model.predict(data)
        for i in range(len(results)):
            results[i] = results[i].tobytes()
        return results

    def simulate(self, size):
        return np.array([np.random.random_sample(size)])

    def to_numpy(self, bytes, size):
        return np.fromstring(bytes, np.float32).reshape(size)

    def __repr__(self):
        return 'yolo.demo.bb2-service'
