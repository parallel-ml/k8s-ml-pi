"""
Bounding box service takes care of model inference after darknet-53.
"""
from service.generic_service import GenericService
from service.yolo.util import load_yolo_model
import service.yolo.util as model_util
import avro.ipc as ipc
import avro.protocol as protocol
import os
import numpy as np

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)

PROTOCOL = protocol.parse(open(DIR_PATH + '/../../../resource/protocol/msg.avpr').read())


class Service(GenericService):
    def __init__(self):
        GenericService.__init__(self)
        load_yolo_model([154, 188])
        self.model = model_util.model
        self.graph = model_util.graph

    def predict(self, input):
        data = [self.to_numpy(input[0], [1, 20, 20, 512])]
        with self.graph.as_default():
            result = self.model.predict(data)

        results = [result.tobytes(), input[1], input[2]]
        return results

    def simulate(self, size):
        return np.array([np.random.random_sample(size)])

    def to_numpy(self, bytes, size):
        return np.fromstring(bytes, np.float32).reshape(size)

    def send(self, output):
        client = ipc.HTTPTransceiver('192.168.1.104', 8080)
        requestor = ipc.Requestor(PROTOCOL, client)

        packet = dict()
        packet['input'] = output

        result = requestor.request('forward', packet)
        client.close()
        return result

    def __repr__(self):
        return 'yolo.demo.bb1-service'
