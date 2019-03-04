from service.generic_service import GenericService
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input, ZeroPadding2D, merge
from keras.models import Model
from customlayers import crosschannelnormalization, splittensor
import numpy as np
import avro.ipc as ipc
import avro.protocol as protocol
import os
from threading import Thread

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)

# read output packet format.
PROTOCOL = protocol.parse(open(DIR_PATH + '/../../../resource/protocol/msg.avpr').read())
weights_path='alexnet_weights.h5'

class Service(GenericService):
    def __init__(self):
        GenericService.__init__(self)

        data = Input([227, 227, 3])

        conv_1 = Conv2D(96, (11, 11), activation='relu', strides=(4, 4))(data)
        conv_2 = MaxPooling2D(strides=(2, 2), pool_size=(3, 3))(X)
        conv_2 = crosschannelnormalization(name='convpool_1')(conv_2)
        conv_2 = ZeroPadding2D((2, 2))(conv_2)
        conv_2 = merge([
                    Conv2D(128, 5, 5, activation='relu', name='conv_2_' + str(i + 1))(
                        splittensor(ratio_split=2, id_split=i)(conv_2)
                    ) for i in range(2)], mode='concat', concat_axis=1, name='conv_2')
        conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
        conv_3 = crosschannelnormalization()(conv_3)
        conv_3 = ZeroPadding2D((1, 1))(conv_3)
        conv_3 = Conv2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

        conv_4 = ZeroPadding2D((1, 1))(conv_3)
        conv_4 = merge([
                        Conv2D(192, 3, 3, activation='relu', name='conv_4_' + str(i + 1))(
                            splittensor(ratio_split=2, id_split=i)(conv_4)
                        ) for i in range(2)], mode='concat', concat_axis=1, name='conv_4')

        conv_5 = ZeroPadding2D((1, 1))(conv_4)
        conv_5 = merge([
                        Conv2D(128, 3, 3, activation='relu', name='conv_5_' + str(i + 1))(
                            splittensor(ratio_split=2, id_split=i)(conv_5)
                        ) for i in range(2)], mode='concat', concat_axis=1, name='conv_5')

        output = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)

        output = Flatten()(output)
        self.model = Model(data, X)
        self.model.load_weights(weights_path)

    def predict(self, input):
        return self.model.predict(np.array([input]))

    def send(self, output):
        Thread(target=self.request, args=(output,)).start()

    @staticmethod
    def request(output):
        client = ipc.HTTPTransceiver('fc-service', 8080)
        requestor = ipc.Requestor(PROTOCOL, client)

        packet = dict()
        packet['input'] = output.tobytes()
        packet['input_shape'] = list(output.shape)
        packet['input_type'] = str(output.dtype)

        requestor.request('forward', packet)
        client.close()

    def __repr__(self):
        return 'alexnet.demo.conv'
