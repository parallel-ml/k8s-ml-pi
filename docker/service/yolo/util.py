""" Utility methods for YOLO model image detection and bound box drawing. """
import yaml
from keras.layers import Conv2D, UpSampling2D, BatchNormalization, Add, Concatenate, LeakyReLU, Input, ZeroPadding2D, \
    Lambda
from keras.models import Model


def load_yolo_model(layer_range=(2, 252), pre_built=dict()):
    """
        Helper method for loading YOLO model with customized topology and according weights

        Args:
            layer_range: Specify which layers to load, number points to lower range and upper
            range of layer in layername file. This range is inclusive.
    """
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))

    lower, upper = layer_range
    lower -= 1
    layer_names = []
    with open(dir_path + '/resource/layers-names') as f:
        for i, line in enumerate(f.read().splitlines()[lower:upper]):
            layer_names.append(line)

    layers, output_layers = [], []
    layer_name_2_idx = dict()

    with open(dir_path + '/resource/model-structure-for-profile.json') as f:
        model_config = yaml.safe_load(f)
        for layer in layer_names:
            class_name = model_config[layer]['class_name']
            config = model_config[layer]['config']
            input_shape = model_config[layer]['input_shape']
            prev_layers = model_config[layer]['prev']
            output = True if 'output' in model_config[layer] else False

            # previous layers picked from dictionary
            if len(layers) == 0:
                layers.append(Input(input_shape))
                prev_layer = layers[-1]
            elif len(prev_layers) == 1:
                prev_layer = layers[layer_name_2_idx[prev_layers[0]]]
            elif len(prev_layers) == 2:
                prev_layer = [
                    layers[layer_name_2_idx[name]] if name in layer_name_2_idx else Lambda(lambda x: x)(pre_built[name])
                    for name in prev_layers]
            else:
                raise ValueError('Previous layer error is not handled')

            if class_name == 'Conv2D':
                prev_layer = Conv2D(**config)(prev_layer)
            elif class_name == 'ZeroPadding2D':
                prev_layer = ZeroPadding2D(**config)(prev_layer)
            elif class_name == 'BatchNormalization':
                prev_layer = BatchNormalization(**config)(prev_layer)
            elif class_name == 'LeakyReLU':
                prev_layer = LeakyReLU(**config)(prev_layer)
            elif class_name == 'Add':
                prev_layer = Add(**config)(prev_layer)
            elif class_name == 'UpSampling2D':
                prev_layer = UpSampling2D(**config)(prev_layer)
            elif class_name == 'Concatenate':
                prev_layer = Concatenate(**config)(prev_layer)
            else:
                raise ValueError('Current layer is not supported')

            # append the new layer to the list and add entry for translating the
            # layer name to its position in the list
            layers.append(prev_layer)
            if output:
                output_layers.append(prev_layer)
            layer_name_2_idx[layer] = len(layers) - 1

    input_list = [layers[0]]
    if 'add_19' in pre_built:
        input_list.append(pre_built['add_19'])
    if 'add_11' in pre_built:
        input_list.append(pre_built['add_11'])

    if len(output_layers) > 0:
        model = Model(input_list, output=[layer for layer in output_layers])
    else:
        model = Model(input_list, output=layers[-1])
    model.summary()
    model.load_weights(dir_path + '/resource/yolo.h5', by_name=True)
    return model
