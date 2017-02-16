import os

filename = os.path.join(os.path.expanduser('~'), '.keras', 'keras.json')
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, "w") as f:
    f.write('{"backend": "theano","floatx": "float32","epsilon": 1e-07,"image_dim_ordering": "th"}')

from scipy.misc import imread, imresize

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.layers.pooling import GlobalAveragePooling2D

def get_squeezenet(nb_classes=1000, path_to_weights=None,dim_ordering='tf'):
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 227, 227))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(227, 227, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only available")
    x = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid', name='conv1')(input_img)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64, dim_ordering=dim_ordering)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64, dim_ordering=dim_ordering)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128, dim_ordering=dim_ordering)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128, dim_ordering=dim_ordering)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192, dim_ordering=dim_ordering)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192, dim_ordering=dim_ordering)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256, dim_ordering=dim_ordering)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256, dim_ordering=dim_ordering)
    x = Dropout(0.5, name='drop9')(x)

    x = Convolution2D(nb_classes, 1, 1, border_mode='valid', name='conv10_',activation='relu')(x)
    out = GlobalAveragePooling2D()(x)
    model = Model(input=input_img, output=[out])
    if path_to_weights:
        model.load_weights(path_to_weights,by_name=True)
    return model


sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

# Modular function for Fire Node
def fire_module(x, fire_id, squeeze=16, expand=64, dim_ordering='tf'):
    s_id = 'fire' + str(fire_id) + '/'
    if dim_ordering is 'tf':
        c_axis = 3
    else:
        c_axis = 1

    x = Convolution2D(squeeze, 1, 1, border_mode='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, 1, 1, border_mode='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, 3, 3, border_mode='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = merge([left, right], mode='concat', concat_axis=c_axis, name=s_id + 'concat')
    return x