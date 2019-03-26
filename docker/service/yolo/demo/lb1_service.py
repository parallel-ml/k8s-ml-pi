from service.generic_service import GenericService
from service.yolo.util import load_yolo_model
import avro.ipc as ipc
import avro.protocol as protocol
import numpy as np
import os

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)

PROTOCOL = protocol.parse(open(DIR_PATH + '/../../../resource/protocol/msg.avpr').read())


class Service(GenericService):
    def __init__(self):
        GenericService.__init__(self)
        self.model = load_yolo_model([2, 93])

    def predict(self, input):
        input = np.fromstring(input[0], np.float64).reshape([1, 320, 320, 3])
        return self.model.predict(input)

    def send(self, output):
        client = ipc.HTTPTransceiver('lb2-service', 8080)
        requestor = ipc.Requestor(PROTOCOL, client)

        packet = dict()
        packet['input'] = [output.tobytes()]

        result = requestor.request('forward', packet)
        client.close()
        return result

    def __repr__(self):
        return 'yolo.demo.lb1-service'
