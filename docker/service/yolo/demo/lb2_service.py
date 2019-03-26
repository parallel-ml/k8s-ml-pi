from service.generic_service import GenericService
from service.yolo.util import load_yolo_model
import numpy as np
import avro.ipc as ipc
import avro.protocol as protocol
import os

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)

PROTOCOL = protocol.parse(open(DIR_PATH + '/../../../resource/protocol/msg.avpr').read())


class Service(GenericService):
    def __init__(self):
        GenericService.__init__(self)
        self.model = load_yolo_model([94, 153])

    def predict(self, input):
        input = np.fromstring(input[0], np.float32).reshape([1, 40, 40, 256])
        output = self.model.predict(input)
        return [np.array(output, copy=True), output, input]

    def send(self, output):
        client = ipc.HTTPTransceiver('bb-service', 8080)
        requestor = ipc.Requestor(PROTOCOL, client)

        packet = dict()
        output = [element.tobytes() for element in output]
        packet['input'] = output

        result = requestor.request('forward', packet)
        client.close()
        return result

    def __repr__(self):
        return 'yolo.demo.lb2-service'
