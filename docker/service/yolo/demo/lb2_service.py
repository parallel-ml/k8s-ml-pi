from service.generic_service import GenericService
from service.yolo.util import load_yolo_model
import service.yolo.util as model_util
import numpy as np


class Service(GenericService):
    def __init__(self):
        GenericService.__init__(self)
        load_yolo_model([94, 153])
        self.model = model_util.model
        self.graph = model_util.graph

    def predict(self, input):
        data = [np.fromstring(input[0], np.float32).reshape([1, 40, 40, 256])]
        with self.graph.as_default():
            output = self.model.predict(data)
        return [np.array(output, copy=True).tobytes(), output.tobytes(), data[0].tobytes()]

    def __repr__(self):
        return 'yolo.demo.lb2-service'
