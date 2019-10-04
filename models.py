## 定义、加载模型
from keras.models import Sequential
## 定义网络层
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
## 定义卷积网络
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers import LSTM,Bidirectional
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import ELU


# number of convolutional filters to use at each layer
nb_filters = [  32,   # 1st conv layer
                32    # 2nd
            ]

# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [3, 3]

# level of convolution to perform at each layer (CONV x CONV)
# nb_conv = [5,5]
nb_conv = [3,3]

activation = 'relu'

def base_model(img_rows, img_cols, img_depth, channel):
    model = Sequential()
    model.add(Conv3D(
        nb_filters[0],
        # kernel_dim1=nb_conv[0],  # depth
        # kernel_dim2=nb_conv[0],  # rows
        # kernel_dim3=nb_conv[0],  # cols
        (nb_conv[0], nb_conv[0], nb_conv[0]),
        # input_shape=(1, img_rows, img_cols, img_depth),
        input_shape=(img_rows, img_cols, img_depth, channel),
        activation = activation
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))


    model.add(Conv3D(
        nb_filters[0]*2,
        # kernel_dim1=nb_conv[0],  # depth
        # kernel_dim2=nb_conv[0],  # rows
        # kernel_dim3=nb_conv[0],  # cols
        (nb_conv[0], nb_conv[0], nb_conv[0]),
        # input_shape=(1, img_rows, img_cols, img_depth),
        activation = activation
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))

    return model

def model(img_rows, img_cols, img_depth, Num_CLASSES,channel):

    model = base_model(img_rows, img_cols, img_depth, channel)
    model.add(Dropout(0.2))

    shape = model.get_output_shape_at(0)
    model.add(Reshape((shape[-1],shape[1]*shape[2]*shape[3])))

    # LSTM layer
    model.add(LSTM(512, return_sequences=True, kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(ELU(alpha=1.0))
    model.add(Dropout(.25))

    ## 1,500,000左右
    model.add(Flatten())

    # ## 131,072
    # model.add(Dense(2**17, init='normal', activation='relu'))

    # model.add(Dropout(0.5))

    # ## 16384
    # model.add(Dense(2**14, init='normal', activation='relu'))

    # model.add(Dropout(0.5))

    ## 1024
    # model.add(Dense(2**10, init='normal', activation='relu'))

    # model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(128, init='normal', activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(Num_CLASSES,init='normal'))

    model.add(Activation('softmax'))

    print(nb_filters[0], 'filters')
    print('input shape: ', img_rows, 'rows, ', img_cols, 'cols, ', img_depth, 'depth')

    return model
