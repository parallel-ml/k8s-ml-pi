from service.generic_service import GenericService
from keras.layers import Conv2D, Input, Add
from keras.models import Model
import numpy as np
import avro.ipc as ipc
import avro.protocol as protocol
import os
from threading import Thread

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)

# read output packet format.
PROTOCOL = protocol.parse(open(DIR_PATH + '/../../../resource/protocol/msg.avpr').read())


class Service(GenericService):
    def __init__(self):
        GenericService.__init__(self)

        data = Input([220, 220, 3])

        X = Conv2D(32, (3, 3))(data)
        residual = Conv2D(64, (3, 3))(X)

        X = Conv2D(32, (1, 1))(residual)
        X = Conv2D(64, (3, 3))(X)
        X = Add()([residual, X])

        residual = Conv2D(128, (3, 3))(X)
        for _ in range(2):


    def predict(self, input):
        return self.model.predict(np.array([input]))

    def send(self, output):
        Thread(target=self.request, args=(output,)).start()

    @staticmethod
    def request(output):
        client = ipc.HTTPTransceiver('lb2-service', 8080)
        requestor = ipc.Requestor(PROTOCOL, client)

        packet = dict()
        packet['input'] = output.tobytes()
        packet['input_shape'] = list(output.shape)
        packet['input_type'] = str(output.dtype)

        requestor.request('forward', packet)
        client.close()

    def __repr__(self):
        return 'yolo.demo.layer-block-1-service'
