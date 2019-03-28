from service.generic_service import GenericService
from service.yolo.util import load_yolo_model
import service.yolo.util as model_util
import os
import numpy as np

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)


class Service(GenericService):
    def __init__(self):
        GenericService.__init__(self)
        load_yolo_model()
        self.model = model_util.model
        self.graph = model_util.graph

    def predict(self, input):
        input = np.fromstring(input[0], np.float64).reshape([1, 320, 320, 3])
        with self.graph.as_default():
            results = self.model.predict(input)
        for i in range(len(results)):
            results[i] = results[i].tobytes()
        return results

    def send(self, output):
        return output

    def __repr__(self):
        return 'yolo.demo.test-service'
