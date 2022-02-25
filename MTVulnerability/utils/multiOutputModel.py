from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Flatten, Input



from keras.applications import Xception, ResNet101V2
import tensorflow as tf
import numpy as np



class MultiOutputModel():
    """
    Used to generate our multi-output model. This CNN contains two branches, one first class, other for
    second class. Each branch contains a sequence of Convolutional Layers that is defined
    on the make_default_hidden_layers method.
    """

    def make_default_hidden_layers(self, inputs, branch_on_flatten=True):
        """
        Used to generate a default set of hidden layers. The structure used in this network is defined as:

        Conv2D -> BatchNormalization -> Pooling -> Dropout
        """
        x = Conv2D(16, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        if branch_on_flatten:
            x = Conv2D(32, (3, 3), padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=-1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)
            x = Flatten()(x)
        return x


    def make_keras_hidden_layers(self, input_shape, inputs, architecture="xception", nb_classes=None):
        _input_shape = (128,128,3)
        from tensorflow.keras.layers.experimental.preprocessing import Resizing
        x = Resizing(_input_shape[0], _input_shape[1])(inputs)

        if architecture =="xception":
            base_model = Xception(
                weights="imagenet",  # Load weights pre-trained on ImageNet.
                input_shape=_input_shape,
                include_top=nb_classes is not None, classes=nb_classes
            )

            # Pre-trained Xception weights requires that input be normalized
            # from (0, 1) to a range (-1., +1.), the normalization layer
            # does the following, outputs = (inputs - mean) / sqrt(var)
            from tensorflow.keras.layers.experimental.preprocessing import Normalization
            norm_layer = Normalization()
            mean = np.array([0.5] * 3)
            var = mean ** 2
            # Scale inputs to [-1, +1]

            x = norm_layer(x)
            norm_layer.set_weights([mean, var])

        elif architecture =="resnet101":
            base_model = ResNet101V2(
                weights="imagenet",  # Load weights pre-trained on ImageNet.
                input_shape=_input_shape,
                include_top=nb_classes is not None, classes=nb_classes
            )

        elif architecture =="wide_resnet":
            from keras_contrib.applications.wide_resnet import WideResidualNetwork
            _input_shape = (32, 32, 3)
            x = Resizing(_input_shape[0], _input_shape[1])(inputs)

            base_model = WideResidualNetwork(
                weights=None,
                input_shape=_input_shape,
                include_top=nb_classes is not None,
                classes=nb_classes

            )

        elif architecture == "nasnet":
            from keras_contrib.applications.nasnet import NASNetCIFAR
            _input_shape = (32, 32, 3)
            x = Resizing(_input_shape[0], _input_shape[1])(inputs)

            base_model = NASNetCIFAR(
                weights=None,
                input_shape=_input_shape,
                classes=nb_classes
            )


        # Freeze the base_model
        #base_model.trainable = False



        # The base model contains batchnorm layers. We want to keep them in inference mode
        # when we unfreeze the base model for fine-tuning, so we make sure that the
        # base_model is running in inference mode here.
        x = base_model(x)

        return x


    def build_first_branch(self, inputs, num_classes1=10, x=None, branch_on_flatten=True):
        """
        Used to build the race branch of our multi-class classification network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.
        """

        if x is None:
            x = self.make_default_hidden_layers(inputs, branch_on_flatten=branch_on_flatten)

        if not branch_on_flatten:
            x = Conv2D(32, (3, 3), padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=-1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)
            x = Flatten()(x)

        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes1)(x)
        x = Activation("softmax", name="first_output")(x)
        return x

    def build_second_branch(self, inputs, num_classes2=10, x=None, branch_on_flatten=True):
        """
        Used to build the second branch of our multi-class classification network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.
        """

        #x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(inputs)
        if x is None:
            x = self.make_default_hidden_layers(inputs, branch_on_flatten=branch_on_flatten)

        if not branch_on_flatten:
            x = Conv2D(32, (3, 3), padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=-1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)
            x = Flatten()(x)

        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes2)(x)
        x = Activation("sigmoid", name="second_output")(x)
        return x

    def build_other_branch(self, inputs,num_classes3=10, index=3, x=None):
        """
        Used to build the optional branches of our multi-class classification network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """

        if x is None:
            x = self.make_default_hidden_layers(inputs)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes3)(x)
        name = "{}th_output".format(index)
        x = Activation("linear", name=name)(x)
        return x, name

    def assemble_full_model(self, input_shape, shared=True, branch_on_flatten=True):
        """
        Used to assemble our multi-output model CNN.
        """
        inputs = Input(shape=input_shape)
        x= None

        if self.pre_trained is not None:
            x = self.make_keras_hidden_layers(input_shape, inputs, architecture=self.pre_trained)

        elif shared:
            x = self.make_default_hidden_layers(inputs, branch_on_flatten=branch_on_flatten)

        class1_branch = self.build_first_branch(inputs, self.num_classes[0],x, branch_on_flatten=branch_on_flatten)
        class2_branch = self.build_second_branch(inputs, self.num_classes[1],x, branch_on_flatten=branch_on_flatten)
        outputs = [class1_branch, class2_branch]

        if self.num_classes[2] is not None:
            for i, nb in enumerate(self.num_classes[2]):
                class3_branch, class3_name = self.build_other_branch(inputs,nb,i+3,x)
                self.outputs.append(class3_name)
                outputs.append(class3_branch)

        model = Model(inputs=inputs,
                      outputs=outputs,
                      name="multi_class_net")

        return model, self.outputs


    def assemble_onebranch_model(self,input_shape):
        """
        Used to assemble our one-output model CNN.
        """
        inputs = Input(shape=input_shape)
        x = inputs
        self.outputs = None

        if self.pre_trained is not None:
            x = self.make_keras_hidden_layers(input_shape, inputs, architecture=self.pre_trained, nb_classes=self.num_classes[0])

        if self.pre_trained is None:
            class1_branch = self.build_first_branch(inputs, num_classes1=self.num_classes[0], x=x)

            x = GlobalAveragePooling2D()(class1_branch)
            class1_branch = Dropout(0.2)(x)  # Regularize with dropout

            outputs = [class1_branch]
            self.outputs = ["first_output"]
        else:
            outputs = [x]

        model = Model(inputs=inputs,
                      outputs=outputs,
                      name="multi_class_net")

        return model, self.outputs


    def __init__(self, num_classes1=10, num_classes2=10, num_classes3=None, pre_trained=None):
        self.num_classes = [num_classes1,num_classes2,num_classes3]
        self.pre_trained = pre_trained
        self.outputs = ["first_output","second_output"]
