from art.utils import load_mnist, load_cifar10
import numpy as np
import tensorflow as tf

def load_dataset(dataset, normalize=True):
    if dataset =="mnist":
        (x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()

    elif dataset =="cifar":
        (x_train, y_train), (x_test, y_test), min_, max_ = load_cifar10()

    elif dataset =="cifar100":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
        x_train = np.true_divide(x_train,255)
        x_test = np.true_divide(x_test,255)
        min_, max_ = 0 ,1


    elif dataset =="cifar100_hierarchical":
        (x_train, y_train1), (x_test, y_test1) = tf.keras.datasets.cifar100.load_data(label_mode="coarse")
        (_, y_train2), (_, y_test2) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
        y_train = np.concatenate([y_train1, y_train2], axis=1)
        y_test = np.concatenate([y_test1, y_test2], axis=1)
        x_train = np.true_divide(x_train,255)
        x_test = np.true_divide(x_test,255)
        min_, max_ = 0, 1

    return (x_train, y_train), (x_test, y_test), min_, max_



def get_imagenet_resized(batch_size=128, img_size=32):

    train_gen, test_gen, num_classes =  get_tfds_dataset('imagenet_resized/{}x{}'.format(img_size,img_size) , batch_size)

    return train_gen, test_gen, num_classes, img_size, img_size

    with np.load(path) as data:
        pass


def get_keras_dataset(dataset, batch_size=128, storage_name="default"):
    import os
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
#    http: // image - net.org / small / train_32x32.tar
    remote_paths = {"imagenet_resized/32x32":('train_npz_32x32', 'http://www.image-net.org/image/downsample/Imagenet32_train_npz.zip', (32,32))}
    dataset = remote_paths[dataset]
    dataset_path = tf.keras.utils.get_file(dataset[0],dataset[1]) #, untar=True)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=dataset[2],
        batch_size=batch_size)

    return train_generator, None, 1000

def get_tfds_dataset( dataset, batch_size=128, test_name="validation"):
    import tensorflow_datasets as tfds

    # Construct a tf.data.Dataset
    (ds_train, ds_test), ds_info = tfds.load(dataset, split=['train', test_name], shuffle_files=True,as_supervised=True,
    with_info=True,
    )

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.batch(batch_size)
    #ds_train = ds_train.skip(batch_size*2)
    #ds_train = ds_train.cache()

    #ds_train = ds_train.cache()
    #ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    #ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    return ds_train, ds_test, ds_info.features["label"].num_classes


def get_omniglot_dataset():
    import tensorflow_datasets as tfds

    # Construct a tf.data.Dataset
    ds = tfds.load("omniglot", split=['train', "test"], shuffle_files=True,
                                             with_info=True, as_supervised=True)

    (ds_train, ds_test), ds_info = ds
    return ds


def get_top_k_class(inputs, k=5):
    v = tf.math.top_k(
        inputs, k=k, sorted=True, name=None
    )[1]

    with tf.compat.v1.Session() as sess:
        sess.run(v)
        return v.eval()
