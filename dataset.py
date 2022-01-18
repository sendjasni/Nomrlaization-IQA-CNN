from pathlib import Path
import pandas as pd
import numpy as np
import keras
import math


class Dataset:

    @staticmethod
    def get_dataset_from_csv(csvfilepath: str, database_dir: Path, verbose=False):
        """Images must be in database_dir/dst_imgs/
        with the name format [photo_name].[distorsion_type].[distorsion_level].png"""

        dataframe = pd.read_csv(csvfilepath)

        nparray_img = np.empty((0, 512, 512, 3))
        nparray_label = np.empty((0, 1, 1, 1))

        for i, row in dataframe.iterrows():
            img_path = f"{str(database_dir / 'dst_imgs' / row['image'])}.{row['dst_type']}.{str(row['dst_lev'])}.png"

            img_PIL = keras.preprocessing.image.load_img(img_path)
            img_np = keras.preprocessing.image.img_to_array(img_PIL)
            # print(f"shape img np : {img_np.shape}")
            # print(f"shape nparray_img : {nparray_img.shape}")

            nparray_img = np.append(nparray_img, img_np[np.newaxis, ...], axis=0)
            quality_score = np.array([row['dmos']])
            nparray_label = np.append(nparray_label, quality_score[np.newaxis, np.newaxis, np.newaxis, ...], axis=0)

        if verbose:
            print(f"Shape du tensor d'images : {nparray_img.shape}")
            print(f"shape des labels correspondants : {nparray_label.shape}")

        return nparray_img, nparray_label, dataframe

    @staticmethod
    def convert_permutoriginals_to_permutlabels(permut_originaux: np.ndarray, nb_distor: int):
        """A partir d'une permutation pour les classes uniques et du nombre de distorsions que subira chacune
      de ces classes, retourne la permutation équivalente pour toutes les images distordues qui gardera le
      même ordre de classes uniques
      >>> permutlabels = Dataset.convert_permutoriginals_to_permutlabels(np.array([1, 2, 0]), 3)
      >>> labels = np.array([1,1,1, 2,2,2, 3,3,3])
      >>> assert((labels[permutlabels] == np.array([2,2,2, 3,3,3, 1,1,1])).all())"""

        permutation_tmp = []
        for elmt in permut_originaux:
            permutation_tmp += list(range(elmt * nb_distor, (elmt + 1) * nb_distor))

        return permutation_tmp

    @staticmethod
    def apply_permut_nparray(nparray: np.ndarray, permut: list):
        """Applique une permutation déjà générée à un tableau numpy sous la forme [elmt_de_train, width, height,
        canal_couleurs] """
        return nparray[permut, :, :, :]

    @staticmethod
    def create_base_permut_and_split(originals: np.ndarray, test_percent: int = 0.2, verbose=False):
        """Originals are the names of the src_imgs in a list"""

        permut = np.random.permutation(len(originals))
        originals_permuted = originals[permut]
        split_point = math.ceil(len(originals_permuted) * test_percent)
        originals_test, originals_train = np.split(originals_permuted, [split_point])

        if verbose:
            print(f"Liste des originaux avant shuffle : \n {originals}")
            print(f"Liste des originaux après shuffle : \n {originals_permuted}")
            print(f"Liste des originaux après split point de 20 % min en test : \n {originals_test} ; {originals_train}")

        return permut, originals_permuted

    @staticmethod
    def split_test_and_train(nparray_to_split: np.ndarray, nb_distorsions: int, test_percent: float = 0.2, verbose=False):
        """Splits a ndarray according to the number of distorsions applied to each img"""
        nb_originals_src = nparray_to_split.shape[0] / nb_distorsions
        split_point = round(nb_originals_src * test_percent)
        arr_test, arr_train = np.split(nparray_to_split, [split_point * nb_distorsions], axis=0)

        if verbose:
            print(f"Array to split ({nparray_to_split.size}) : \n {nparray_to_split.flatten()}")
            print(f"Array to split after taking {test_percent * 100} % as Test : ")
            print(f"Test ({arr_test.shape[0]}/{nparray_to_split.shape[0]}) : {arr_test.flatten()} ; ")
            print(f"Train ({arr_train.shape[0]}/{nparray_to_split.shape[0]}) : {arr_train.flatten()}")

        return arr_test, arr_train
