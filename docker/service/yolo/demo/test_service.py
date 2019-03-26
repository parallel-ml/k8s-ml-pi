from service.generic_service import GenericService
from service.yolo.util import load_yolo_model
import os

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)


class Service(GenericService):
    def __init__(self):
        GenericService.__init__(self)
        self.model = load_yolo_model()

    def predict(self, input):
        results = self.model.predict(input)
        for i in range(len(results)):
            results[i] = results[i].tobytes()
        return results

    def send(self, output):
        return output

    def __repr__(self):
        return 'yolo.demo.test-service'
