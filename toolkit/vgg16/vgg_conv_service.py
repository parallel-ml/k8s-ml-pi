from service.generic_service import GenericService
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.models import Model, load_model
import numpy as np
import avro.ipc as ipc
import avro.protocol as protocol
import os
from threading import Thread

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)

# read output packet format.
PROTOCOL = protocol.parse(open(DIR_PATH + '/../../../resource/protocol/msg.avpr').read())
weights_path='vgg_notop.h5'

class Service(GenericService):
    def get_config(filepath='resource/layer-name.txt'):
        with open(filepath) as f:
            data = f.read()
        f.close()
        data = data.split('\n')
        return data

    def get_model(layers):
        img_input = Input([224, 224, 3])
        x = img_input
        if 'block1' in layers: 
            x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
            x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        if 'block2' in layers: 
            x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
            x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        if 'block3' in layers: 
            x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
            x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
            x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        if 'block4' in layers:
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        if 'block5' in layers: 
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        return Model(img_input, x, name='vgg16')

    def load_weight(model,weigt_path=weights_path):
        model_weights = load_model(weigt_path)
        for layer in model.layers:
            layer_name = layer.get_config()['name']
            layer_weights = model_weights.get_layer(layer_name).get_weights()
            layer.set_weights(layer_weights)
        return model

    def __init__(self):
        GenericService.__init__(self)
        layers = self.get_config()
        self.model = self.get_model(layers)
        self.model = self.load_weight(self.model)
    
    def predict(self, input):
        return self.model.predict(np.array([input]))

    def send(self, output):
        Thread(target=self.request, args=(output,)).start()

    @staticmethod
    def request(output):
        client = ipc.HTTPTransceiver('vgg-conv-service', 8080)
        requestor = ipc.Requestor(PROTOCOL, client)

        packet = dict()
        packet['input'] = output.tobytes()
        packet['input_shape'] = list(output.shape)
        packet['input_type'] = str(output.dtype)

        requestor.request('forward', packet)
        client.close()

    def __repr__(self):
        return 'vgg.demo.conv'
