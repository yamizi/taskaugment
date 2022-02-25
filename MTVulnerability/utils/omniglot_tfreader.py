import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from absl import flags, logging
from tqdm.auto import tqdm

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class Omniglot:
    SAVE_PATH = './ressources/Omniglot/'

    def __init__(self,
                 n_train_episode=5000,
                 n_valid_episode=1000,
                 img_size=(28, 28),
                 force_update=False,
                 n_train_class=1200,
                 c_way=5,
                 k_shot=1,
                 batch_size=32):
        self.n_train_episode = n_train_episode
        self.n_valid_episode = n_valid_episode
        self.c_way = c_way #FLAGS.get("c_way",5)
        self.k_shot = k_shot #FLAGS.get("k_shot",1)
        self.bs = batch_size
        self.img_shape = img_size + (1, )
        self.n_train_class = n_train_class
        self.np_dset = self._get_np_dset(img_size, force_update)
        # get mean,std
        self.mean = np.mean(self.np_dset[:self.n_train_class])
        self.std = np.std(self.np_dset[:self.n_train_class])
        LOGGER.info('mean: {}, stddev: {}'.format(self.mean, self.std))
        # normalize
        self.np_dset = (self.np_dset - self.mean) / self.std
        self.data = self._get_tf_dsets(self.np_dset)

    def get_data(self):
        return self.data

    def _get_np_dset(self, img_size, force_update=False):
        if not os.path.exists(Omniglot.SAVE_PATH):
            os.makedirs(Omniglot.SAVE_PATH)

        dset_path = os.path.join(Omniglot.SAVE_PATH,
                                 'dset_{}.npy'.format(img_size))

        if not os.path.exists(dset_path) or force_update:
            [
                dset_all,
            ], info = tfds.load("omniglot:3.0.0",
                                with_info=True,
                                split=['train+test'])

            def resize(x):
                x['image'] = tf.image.resize(x['image'], size=img_size)
                x['image'] = tf.image.rgb_to_grayscale(x['image'])
                return x

            dset_all = dset_all.map(resize)
            np_dset = np.zeros(
                shape=[info.features['label'].num_classes, 20, 28, 28, 1],
                dtype=np.float32)
            counter = np.zeros(shape=[
                info.features['label'].num_classes,
            ],
                               dtype=np.int32)
            for i, v in enumerate(tqdm(dset_all)):
                c_idx = v['label'].numpy()
                np_dset[c_idx, counter[c_idx]] = v['image'].numpy()
                counter[c_idx] += 1

            np.save(dset_path, np_dset)
        else:
            np_dset = np.load(dset_path)

        return np_dset

    def _get_tf_dsets(self, np_dset):
        train_dataset = tf.data.Dataset.from_generator(
            lambda: self._episode_generator(np_dset,
                                            self.n_train_episode,
                                            True,
                                            c_way=self.c_way,
                                            k_shot=self.k_shot,
                                            img_shape=self.img_shape),
            output_types=((np.float32, np.float32),
                          np.float32)).cache().shuffle(10000).batch(
                              self.bs, drop_remainder=True).prefetch(
                                  buffer_size=tf.data.experimental.AUTOTUNE)
        valid_dataset = tf.data.Dataset.from_generator(
            lambda: self._episode_generator(np_dset,
                                            self.n_valid_episode,
                                            False,
                                            c_way=self.c_way,
                                            k_shot=self.k_shot,
                                            img_shape=self.img_shape),
            output_types=((np.float32, np.float32), np.float32)).cache().batch(
                self.bs, drop_remainder=False).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)

        return (train_dataset, valid_dataset)

    def _episode_generator(self,
                           np_dset,
                           n_episode,
                           is_train,
                           c_way=5,
                           k_shot=1,
                           img_shape=[28, 28, 1]):
        n_samples = np_dset.shape[1]  # n_samples=20
        if is_train:
            pool = list(range(0, self.n_train_class))
            n_query_per_c = (n_samples - k_shot)
            n_query = n_query_per_c * c_way  # n_query=19*5
        else:
            pool = list(range(self.n_train_class, np_dset.shape[0]))
            n_query_per_c = 1
            n_query = n_query_per_c * c_way  # n_query=1*5

        for i in range(n_episode):
            support_set = np.zeros(shape=[n_query, c_way, k_shot, *img_shape],
                                   dtype=np.float32)
            query_set = np.zeros(shape=[n_query, c_way, k_shot, *img_shape],
                                 dtype=np.float32)
            y = np.zeros(shape=[n_query, c_way], dtype=np.float32)

            selected_c_idx = np.random.choice(pool, c_way, replace=False)
            selected_c_data = np_dset[selected_c_idx]
            # rotation augment classes
            rotate_degree = np.random.randint(low=0, high=4, size=c_way)
            for i, v in enumerate(rotate_degree):
                selected_c_data[i] = np.rot90(selected_c_data[i],
                                              v,
                                              axes=(-3, -2))

            for c in range(c_way):
                perm_idx = np.random.permutation(n_samples)
                selected_s_idx = perm_idx[:k_shot]
                # support set
                selected_s_per_query = selected_c_data[
                    c, selected_s_idx]  # k_shot data with same class
                support_set[:, c:c + 1] = np.tile(selected_s_per_query,
                                                  [n_query, 1, 1, 1, 1, 1])
                # query set
                selected_q_idx = perm_idx[k_shot:k_shot +
                                          n_query_per_c]  # n_query_per_c
                selected_q = selected_c_data[
                    c, selected_q_idx]  # n_query data with same class
                tmp_q = np.repeat(selected_q, k_shot * c_way, axis=0)
                tmp_q = np.reshape(tmp_q,
                                   [n_query_per_c, c_way, k_shot, *img_shape])
                query_set[c * n_query_per_c:(c + 1) * n_query_per_c] = tmp_q
                y[c * n_query_per_c:(c + 1) * n_query_per_c, c] = np.ones(
                    shape=[
                        n_query_per_c,
                    ], dtype=np.float32)

            yield (support_set, query_set), y


    def vis(self, num_samples=10):
        tfdset = self.data[0]  # train
        v = next(iter(tfdset))
        s = v[0][0][0]  # tuple->support_set->batch
        q = v[0][1][0]  # tuple->query_set->batch
        y = v[1][0]
        sampled_idx = np.random.permutation(q.shape[0])[:num_samples]
        for n in sampled_idx:
            fig = plt.figure(figsize=[6, 1])
            for c in range(q.shape[1]):
                k = 0
                plt.subplot(1, 6, c + 1)
                plt.imshow(s[n, c, k][..., 0], cmap='gray')
            plt.subplot(1, 6, 6)
            plt.title('query:{}'.format(y[n]))
            plt.imshow(q[n, 0, k][..., 0], cmap='gray')



def get_tasks_generator():

    if os.path.exists("ressources/Omniglot/X.npy"):
        X, y = np.load("ressources/Omniglot/X.npy"), np.load("ressources/Omniglot/y.npy")
        alphabets = np.load("ressources/Omniglot/alphabets.npy")

    else:
        import scipy.io
        mat = scipy.io.loadmat('ressources/Omniglot/data_background.mat')

        alphabets = [e[0][0] for e in mat["names"].tolist()]
        images = mat["images"]

        X = []
        y=  []

        for i, imgs in enumerate(images):
            y_ = [i, 0]
            imgs_alphabet = imgs[0]
            for j, letter in enumerate(imgs_alphabet):
                variants = letter[0]
                for k, variant in enumerate(variants):
                    y.append([i, j, k])
                    X.append(variant[0])

        X = np.array(X)
        y = np.array(y)
        alphabets = np.array(alphabets)
        np.save("ressources/Omniglot/X.npy", X)
        np.save("ressources/Omniglot/y.npy", y)
        np.save("ressources/Omniglot/alphabets.npy", alphabets)

    return X, y, alphabets

get_tasks_generator()


