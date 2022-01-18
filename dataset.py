from utiles import *
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
            print(
                f"Liste des originaux après split point de 20 % min en test : \n {originals_test} ; {originals_train}")

        return permut, originals_permuted

    @staticmethod
    def split_test_and_train(nparray_to_split: np.ndarray, nb_distorsions: int, test_percent: float = 0.2,
                             verbose=False):
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

    @staticmethod
    def data_augmentation_patching(dataset_X: np.ndarray, nb_division: int, labels: np.ndarray = None) -> np.ndarray:
        """
        >>> dataset = get_quick_dataset(inRGB=True)
        >>> dataset = Dataset.data_augmentation_patching(dataset[0:1,:,:,:], nb_division=2)
        >>> a = plt.imshow(dataset[0, :, :, :]); plt.show()
        >>> a = plt.imshow(dataset[1, :, :, :]); plt.show()
        """

        imgs_nb, h, w, c = dataset_X.shape
        assert h % nb_division == 0, f"{h} rows is not evenly divisible by {nb_division}"
        assert w % nb_division == 0, f"{w} cols is not evenly divisible by {nb_division}"

        nb_duplicate = nb_division * nb_division
        print(f"[DATA AUGMENT] Processing to get x{nb_duplicate} data by tiling the images...")

        patch_shape = (dataset_X.shape[1] // nb_division, dataset_X.shape[2] // nb_division)
        # new_X = (dataset_X.reshape(h // nb_division, nb_division, -1, nb_division)
        #          .swapaxes(1, 2)
        #          .reshape(-1, nb_division, nb_division))

        new_X = np.zeros((0, patch_shape[0], patch_shape[1], dataset_X.shape[3])).astype(int)

        for i_img in range(dataset_X.shape[0]):
            for x in range(0, dataset_X.shape[1], patch_shape[0]):  # pas certain de si c'est shape[1] ou shape[2]
                for y in range(0, dataset_X.shape[2], patch_shape[1]):
                    new_patch = dataset_X[i_img, y:y + patch_shape[0], x:x + patch_shape[1], :]
                    new_X = np.append(new_X, new_patch[np.newaxis, :, :, :], axis=0)

        print(f"[DATA AUGMENT] Finished.")
        if labels.any():
            labels = np.repeat(labels.flatten(), nb_duplicate)[:, np.newaxis, np.newaxis, np.newaxis]
            return new_X, labels

        return new_X

# from dataset import Dataset
# from utiles import *
# import matplotlib.pyplot as plt
# datasetoriginal = get_quick_dataset(inRGB=True)
# dataset = Dataset.data_augmentation_patching(datasetoriginal[0:1,:,:,:], nb_division=2)
