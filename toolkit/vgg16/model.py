# Code adopted from the original Keras VGG16 implementation

from keras.layers import Conv2D, MaxPooling2D, Input
from keras.models import Model, load_model

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

    # Create model.
    return Model(img_input, x, name='vgg16')

def load_weight(model,weigt_path='vgg_notop.h5'):
    model_weights = load_model(weigt_path)
    for layer in model.layers:
        layer_name = layer.get_config()['name']
        layer_weights = model_weights.get_layer(layer_name).get_weights()
        layer.set_weights(layer_weights)
    return model

if __name__ == '__main__':
    layers = get_config()
    model = get_model(layers)
    model = load_weight(model)

