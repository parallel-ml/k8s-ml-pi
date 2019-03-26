from service.generic_service import GenericService
from service.yolo.util import load_yolo_model
import os
import numpy as np

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)


class Service(GenericService):
    def __init__(self):
        GenericService.__init__(self)
        self.model = load_yolo_model()

    def predict(self, input):
        input = np.fromstring(input[0], np.float64).reshape([1, 320, 320, 3])
        results = self.model.predict(input)
        for i in range(len(results)):
            results[i] = results[i].tobytes()
        return results

    def send(self, output):
        return output

    def __repr__(self):
        return 'yolo.demo.test-service'
