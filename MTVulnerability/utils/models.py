from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input, BatchNormalization

from keras.models import Model
from keras.regularizers import l2
from tensorflow import keras
import tensorflow as tf
from utils.multiOutputModel import MultiOutputModel
from utils.cifar100VGG import Cifar100vgg
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.utils import multi_gpu_model

import sys

# Example LeNet classifier architecture with Keras & ART
# To obtain good performance in adversarial training on CIFAR-10, use a larger architecture
def build_lenet(input_shape=(32, 32, 1), nb_classes=10, loss=None, metrics = None):
    img_input = Input(shape=input_shape)
    conv2d_1 = Conv2D(
        6,
        (5, 5),
        padding="valid",
        kernel_regularizer=l2(0.0001),
        activation="relu",
        kernel_initializer="he_normal",
        input_shape=input_shape,
    )(img_input)
    conv2d_1_bn = BatchNormalization()(conv2d_1)
    conv2d_1_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv2d_1_bn)
    conv2d_2 = Conv2D(16, (5, 5), padding="valid", activation="relu", kernel_initializer="he_normal")(conv2d_1_pool)
    conv2d_2_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv2d_2)
    flatten_1 = Flatten()(conv2d_2_pool)
    dense_1 = Dense(120, activation="relu", kernel_initializer="he_normal")(flatten_1)
    dense_2 = Dense(84, activation="relu", kernel_initializer="he_normal")(dense_1)
    img_output = Dense(nb_classes, activation="softmax", kernel_initializer="he_normal")(dense_2)
    model = Model(img_input, img_output)


    if metrics is None:
        metrics = ["accuracy"]
    if loss is None:
        loss = keras.losses.categorical_crossentropy


    model.compile(
        loss=loss, optimizer="adam", metrics=metrics
    )

    return model


def build_kerasnet(input_shape=(32, 32, 3), nb_classes=10, loss=None, metrics = None):
 model, outputs = MultiOutputModel(nb_classes, None,None).assemble_onebranch_model(input_shape)

 if metrics is None:
     metrics = ["accuracy"]

 if loss is None:
     loss = keras.losses.mean_squared_error

 model.compile(
     loss=loss, optimizer="adam", metrics=metrics
 )

 return model


def build_vggNet(input_shape=(32, 32, 3), nb_classes=None, loss=None, metrics = None):
    if nb_classes is None:
        nb_classes=  [20,100]


    vgg = Cifar100vgg(train=False,num_classes1=nb_classes[0], num_classes2=nb_classes[1], x_shape=input_shape)
    model = vgg.model

    if metrics is None:
         metrics = ["accuracy"]
    if loss is None:
         loss = keras.losses.mean_squared_error

    model.compile(
         loss=loss, optimizer="adam", metrics=metrics
    )

    return model


def build_Noutput_net(input_shape=(32, 32, 3), nb_classes=10, loss=None, metrics = None, N=2, nb_classes2=None,
                      nb_classes3=None, pre_trained=None, branch_on_flatten=True, distributed=True):
    remaining_classes = None
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    if strategy.num_replicas_in_sync <2:
        distributed = False
    def _build(nb_classes,nb_classes2,nb_classes3, remaining_classes, metrics,loss):
        if N==1:
            model, outputs = MultiOutputModel(nb_classes, nb_classes2, remaining_classes,
                                              pre_trained=pre_trained).assemble_onebranch_model(input_shape)
        else:
            if nb_classes2 is None:
                nb_classes2 = nb_classes
            if nb_classes3 is None:
                nb_classes3 = nb_classes
            if N>2:
                remaining_classes = [nb_classes3 for i in range(2,N)]

            model, outputs = MultiOutputModel(nb_classes, nb_classes2, remaining_classes, pre_trained=pre_trained)\
                .assemble_full_model(input_shape, branch_on_flatten=branch_on_flatten)


        if metrics is None:
            metrics = ["accuracy"]
        else:
            metrics = dict(zip(outputs, metrics))
        if loss is None:
            loss = keras.losses.mean_squared_error
        else:
            loss =dict(zip(outputs, loss))

        model.compile(
            loss=loss, optimizer="adam", metrics=metrics
        )

        return model

    if distributed:
        with strategy.scope():
            return _build(nb_classes,nb_classes2,nb_classes3, remaining_classes, metrics,loss)
    else:
        return _build(nb_classes,nb_classes2,nb_classes3, remaining_classes, metrics,loss)

def get_single_output_by_dataset(dataset):
    datasets = ("mnist","cifar","cifar100","cifar100_hierarchical","face")
    methods = (build_lenet,build_kerasnet,build_kerasnet )
    dic = dict()


def load_model(dataset: object = None, nb_epochs: object = None, title: object = None, nb_classes: object = 1, path: object = None) -> object:

    def top_k_accuracy(y_true, y_pred):
        k = nb_classes
        top_pred = tf.math.top_k(
            y_pred, k=k, sorted=True, name=None
        )[1]

        top_true = tf.math.top_k(
            y_true, k=k, sorted=True, name=None
        )[1]

        equality = top_pred-top_true
        from tensorflow.python.framework import dtypes
        return 1- tf.math.count_nonzero(equality,dtype=dtypes.int32) / tf.size(equality)


    if dataset =="cityscape" and path is None:
        return load_cityscapes_model()

    output_model = "output/models/{}/epochs_{}/{}_top_{}/model.h5".format(dataset, nb_epochs, title,nb_classes) if path is None else path
    custom_objects= {}
    if nb_classes is None or nb_classes > 1:
        custom_objects = {"top_k_accuracy": top_k_accuracy}
    my_model = keras.models.load_model(output_model, custom_objects=custom_objects)

    return my_model


def load_cityscapes_model(architecture="unet",load_weights='ressources/cityscapes/unet.h5', parameters=None):
    sys.path.append("./cityscapes_Unet")
    from cityscapes_Unet.multiclassunet import Unet
    from cityscapes_Unet.dilatednet import DilatedNet
    from cityscapes_Unet.batch_training import dice_coeff

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():

#img_height, img_width, nclasses=3, filters=64
#img_height, img_width, nclasses, use_ctx_module=False, bn=False
        """
        The model was trained on cityscape-dataset with classes merged into 8 main classes. 
        The model was trained for 300k iterations with lr=0.001, and no wd.
        """
        if architecture=="unet":
            params = {"img_height":256, "img_width":256, "nclasses":3, "filters":64}
            params = {**params, **parameters}
            model = Unet(**params)

        if architecture=="dilatenet":
            params = {"img_height": 256, "img_width": 256, "nclasses": 3, "use_ctx_module": False,"bn":False}
            params = {**params, **parameters}
            model = DilatedNet(**params)

        if load_weights is not None:
            model.load_weights(load_weights)

    if load_weights is None:
        compile(optimizer='adam', loss='categorical_crossentropy', metrics=[dice_coeff, 'accuracy'])
        tb = TensorBoard(log_dir='ressources/tensorboard/logs', write_graph=True)
        mc = ModelCheckpoint(mode='max', filepath='ressources/cityscapes/checkpoint.h5', monitor='acc', save_best_only='True',
                             save_weights_only='True', verbose=1)
        es = EarlyStopping(mode='max', monitor='acc', patience=6, verbose=1)



