## Adapted from https://github.com/geifmany/cifar-vgg/blob/master/cifar100vgg.py

from __future__ import print_function
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers

class Cifar100vgg:
    def __init__(self,train=True,num_classes1=20, num_classes2=100, x_shape=None ):
        self.num_classes = num_classes1
        self.num_classes2 = num_classes2
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3] if x_shape is None else x_shape

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        #else:
        #    self.model.load_weights('cifar100vgg.h5')

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        weight_decay = self.weight_decay
        activation = "relu"
        dropout = 0.2

        inputs = Input(shape=self.x_shape)
        x = Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay))(inputs)

        x =Activation(activation)(x)
        x =BatchNormalization()(x)
        x =Dropout(dropout)(x)

        x =Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x =Activation(activation)(x)
        x =BatchNormalization()(x)

        x =MaxPooling2D(pool_size=(2, 2))(x)

        x =Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x =Activation(activation)(x)
        x =BatchNormalization()(x)
        x =Dropout(dropout)(x)

        x = Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation(activation)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)


        def branch(x,num_classes):
            x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
            x = Activation(activation)(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout)(x)

            x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
            x = Activation(activation)(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout)(x)

            x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
            x = Activation(activation)(x)
            x = BatchNormalization()(x)

            x = MaxPooling2D(pool_size=(2, 2))(x)

            x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
            x = Activation(activation)(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout)(x)

            x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
            x = Activation(activation)(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout)(x)

            x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
            x = Activation(activation)(x)
            x = BatchNormalization()(x)

            x = MaxPooling2D(pool_size=(2, 2))(x)

            x1 =Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
            x1 =Activation(activation)(x1)
            x1 =BatchNormalization()(x1)
            x1 =Dropout(dropout)(x1)

            x1 =Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x1)
            x1 =Activation(activation)(x1)
            x1 =BatchNormalization()(x1)
            x1 =Dropout(dropout)(x1)

            x1 =Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x1)
            x1 =Activation(activation)(x1)
            x1 =BatchNormalization()(x1)

            x1 =MaxPooling2D(pool_size=(2, 2))(x1)
            x1 =Dropout(dropout)(x1)

            x1 =Flatten()(x1)
            x1 =Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(x1)
            x1 =Activation(activation)(x1)
            x1 =BatchNormalization()(x1)

            x1 = Dropout(dropout)(x1)
            x1 =Dense(num_classes)(x1)
            x1 =Activation('softmax')(x1)
            return x1

        x1 = branch(x,self.num_classes)
        x2 = branch(x,self.num_classes2)

        outputs = [x1,x2]

        model = Model(inputs=inputs,
                      outputs=outputs,
                      name="multi_class_net")

        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        print(mean)
        print(std)
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 121.936
        std = 68.389
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model):

        #training parameters
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20

        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)


        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)


        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)



        #optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


        # training process in a for loop with learning rate drop every 25 epoches.

        historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)
        model.save_weights('cifar100vgg.h5')
        return model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 100)
    y_test = keras.utils.to_categorical(y_test, 100)

    model = cifar100vgg()

    predicted_x = model.predict(x_test)
    residuals = (np.argmax(predicted_x,1)!=np.argmax(y_test,1))
    loss = sum(residuals)/len(residuals)
    print("the validation 0/1 loss is: ",loss)
