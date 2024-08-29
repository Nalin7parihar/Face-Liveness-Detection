import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape, Dense, Activation
from tensorflow.keras.layers import Multiply, Add, DepthwiseConv2D, BatchNormalization
from tensorflow.keras.utils import get_custom_objects

def h_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6

def h_swish(x):
    return x * h_sigmoid(x)

# Đăng ký hàm kích hoạt tùy chỉnh
get_custom_objects().update({'h_swish': Activation(h_swish)})

def squeeze_excite_block(input, ratio=4):
    filters = input.shape[-1]
    se = GlobalAveragePooling2D()(input)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='hard_sigmoid', use_bias=False)(se)
    return Multiply()([input, se])

def bottleneck(x, exp, out, s, squeeze):
    skip = x
    x = Conv2D(exp, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(h_swish)(x)
    
    x = DepthwiseConv2D(3, strides=s, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(h_swish)(x)
    
    if squeeze:
        x = squeeze_excite_block(x)
    
    x = Conv2D(out, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    
    if s == 1 and skip.shape[-1] == out:
        return Add()([skip, x])
    else:
        return x

class MobileNetV3:
    @staticmethod
    def build(width, height, depth, classes):
        inputs = Input(shape=(height, width, depth))
        
        x = Conv2D(16, (3, 3), strides=(2, 2), padding='same', use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation(h_swish)(x)
        
        x = bottleneck(x, 16, 16, 2, True)
        x = bottleneck(x, 72, 24, 2, False)
        x = bottleneck(x, 88, 24, 1, False)
        x = bottleneck(x, 96, 40, 2, True)
        x = bottleneck(x, 240, 40, 1, True)
        x = bottleneck(x, 240, 40, 1, True)
        x = bottleneck(x, 120, 48, 1, True)
        x = bottleneck(x, 144, 48, 1, True)
        x = bottleneck(x, 288, 96, 2, True)
        x = bottleneck(x, 576, 96, 1, True)
        x = bottleneck(x, 576, 96, 1, True)
        
        x = Conv2D(576, (1, 1), use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation(h_swish)(x)
        
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 576))(x)
        x = Conv2D(1024, (1, 1), use_bias=False)(x)
        x = Activation(h_swish)(x)
        
        x = Conv2D(classes, (1, 1), padding='same')(x)
        x = Reshape((classes,))(x)
        outputs = Activation('softmax')(x)
        
        model = Model(inputs, outputs)
        return model

# Hàm để tải mô hình với các hàm kích hoạt tùy chỉnh
def load_model_with_custom_objects(model_path):
    return load_model(model_path, custom_objects={'h_swish': Activation(h_swish)})