"""
FROM https://towardsdatascience.com/building-a-multi-output-convolutional-neural-network-with-keras-ed24c7bc1178
"""

import numpy as np
from keras.utils import to_categorical
from PIL import Image

import os
import glob
import pandas as pd

dataset_folder_name = './ressources/UTKFace'
TRAIN_TEST_SPLIT = 0.7
IM_WIDTH = IM_HEIGHT = 198
dataset_dict = {
    'race_id': {
        0: 'white',
        1: 'black',
        2: 'asian',
        3: 'indian',
        4: 'others'
    },
    'gender_id': {
        0: 'male',
        1: 'female'
    }
}
dataset_dict['gender_alias'] = dict((g, i) for i, g in dataset_dict['gender_id'].items())
dataset_dict['race_alias'] = dict((r, i) for i, r in dataset_dict['race_id'].items())



def parse_dataset(dataset_path, ext='jpg'):
    """
    Used to extract information about our dataset. It does iterate over all images and return a DataFrame with
    the data (age, gender and sex) of all files.
    """

    def parse_info_from_file(path):
        """
        Parse information from a single file
        """
        try:
            filename = os.path.split(path)[1]
            filename = os.path.splitext(filename)[0]
            age, gender, race, _ = filename.split('_')
            return int(age), dataset_dict['gender_id'][int(gender)], dataset_dict['race_id'][int(race)]
        except Exception as ex:
            return None, None, None

    files = glob.glob(os.path.join(dataset_path, "*.%s" % ext))

    records = []
    for file in files:
        info = parse_info_from_file(file)
        records.append(info)

    df = pd.DataFrame(records)
    df['file'] = files
    df.columns = ['age', 'gender', 'race', 'file']
    df = df.dropna()

    return df


class UtkFaceDataGenerator():
    """
    Data generator for the UTKFace dataset. This class should be used when training our Keras multi-output model.
    """

    def __init__(self, df, outputs = None, combinatorial=False ):
        self.df = df
        self.outputs = outputs
        self.combinatorial = combinatorial

    @staticmethod
    def input_shape():
        return (IM_WIDTH, IM_HEIGHT, 3)

    @staticmethod
    def output_combinations():
        return len(dataset_dict['gender_id'].items()) * len(dataset_dict['race_id'].items())

    def generate_split_indexes(self,max_inputs=None):
        p = np.random.permutation(len(self.df))
        if max_inputs is None:
            max_inputs = int(len(self.df))
        train_up_to =  int(max_inputs * TRAIN_TEST_SPLIT)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:max_inputs]
        train_up_to = int(train_up_to * TRAIN_TEST_SPLIT)
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]

        # converts alias to id
        self.df['gender_id'] = self.df['gender'].map(lambda gender: dataset_dict['gender_alias'][gender])
        self.df['race_id'] = self.df['race'].map(lambda race: dataset_dict['race_alias'][race])
        self.max_age = self.df['age'].max()

        return train_idx, valid_idx, test_idx

    def preprocess_image(self, img_path):
        """
        Used to perform some minor preprocessing on the image before inputting into the network.
        """
        im = Image.open(img_path)
        im = im.resize((IM_WIDTH, IM_HEIGHT))
        im = np.array(im) / 255.0

        return im

    def generate_images(self, image_idx, is_training, batch_size=16):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        """

        # arrays to store our batched data
        images, ages, races, genders, combinations = [], [], [], [], []
        nb_combinations = UtkFaceDataGenerator.output_combinations()

        while True:
            for idx in image_idx:
                person = self.df.iloc[idx]

                age = person['age']
                race = person['race_id']
                gender = person['gender_id']
                file = person['file']

                im = self.preprocess_image(file)
                images.append(im)

                ages.append(age / self.max_age)

                if self.combinatorial:
                    combination = gender * len(dataset_dict['gender_id']) + race
                    combinations.append(to_categorical(combination, nb_combinations))

                else:
                    races.append(to_categorical(race, len(dataset_dict['race_id'])))
                    genders.append(to_categorical(gender, len(dataset_dict['gender_id'])))

                # yielding condition
                if len(images) >= batch_size:

                    if self.combinatorial:
                        yield np.array(images), np.array(combinations)

                    else:

                        if self.outputs is None:
                            yield np.array(images), [np.array(ages), np.array(races), np.array(genders)]
                        else:
                            outs = dict(zip(["age","race","gender"],[np.array(ages), np.array(races), np.array(genders)]))
                            outputs = [outs[e] for e in self.outputs]
                            yield np.array(images), outputs


                    images, ages, races, genders, combinations = [], [], [], [], []

            if not is_training:
                break


    @staticmethod
    def build(folder=None,batch_size = 32, valid_batch_size = 32, outputs=None, max_inputs=None, combinatorial=None):
        if folder is None:
            folder = dataset_folder_name

        df = parse_dataset(folder)

        data_generator = UtkFaceDataGenerator(df, outputs, combinatorial=combinatorial)
        train_idx, valid_idx, test_idx = data_generator.generate_split_indexes(max_inputs)

        train_gen = data_generator.generate_images(train_idx, is_training=True, batch_size=batch_size)
        valid_gen = data_generator.generate_images(valid_idx, is_training=True, batch_size=valid_batch_size)

        train_steps = len(train_idx)// batch_size
        validation_steps = len(valid_idx) // valid_batch_size

        return train_gen,valid_gen, train_steps, validation_steps
