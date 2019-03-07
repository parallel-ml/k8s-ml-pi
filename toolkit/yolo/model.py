from keras.models import Model
from keras.layers import Input, Conv2D, Add, ZeroPadding2D, BatchNormalization, LeakyReLU
import yaml

LAYERS = ['conv2d_1',
          'batch_normalization_1',
          'leaky_re_lu_1',
          'zero_padding2d_1',
          'conv2d_2',
          'batch_normalization_2',
          'leaky_re_lu_2',
          'conv2d_3',
          'batch_normalization_3',
          'leaky_re_lu_3',
          'conv2d_4',
          'batch_normalization_4',
          'leaky_re_lu_4',
          'add_1']
MODEL_TOPO = 'resource/model-structure.json'
MODEL_WEIGHT = 'resource/yolo.h5'


def load_model_from_weights():
    layers = []
    prev_layer = None
    layername_2_idx = dict()
    with open(MODEL_TOPO) as f:
        model_config = yaml.safe_load(f)
        for layer in LAYERS:
            class_name = model_config[layer]['class_name']
            config = model_config[layer]['config']
            input_shape = model_config[layer]['input_shape']

            if len(layers) == 0:
                layers.append(Input(input_shape))
                prev_layer = layers[-1]

            if class_name == 'Conv2D':
                prev_layer = Conv2D(**config)(prev_layer)
            elif class_name == 'ZeroPadding2D':
                prev_layer = ZeroPadding2D(**config)(prev_layer)
            elif class_name == 'BatchNormalization':
                prev_layer = BatchNormalization(**config)(prev_layer)
            elif class_name == 'LeakyReLU':
                prev_layer = LeakyReLU(**config)(prev_layer)
            elif class_name == 'Add':
                shortcut = model_config[layer]['residual']
                prev_layer = Add()([layers[layername_2_idx[shortcut]], prev_layer])
            else:
                raise ValueError('Current layer is not supported')
            layers.append(prev_layer)
            layername_2_idx[layer] = len(layers) - 1

    model = Model(layers[0], layers[-1])
    model.load_weights(MODEL_WEIGHT, by_name=True)
    model.summary()


if __name__ == '__main__':
    load_model_from_weights()
