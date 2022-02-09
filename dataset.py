from utiles import *
from pathlib import Path
import pandas as pd
import numpy as np
import keras
import math
from typing import Tuple
from PIL import Image
import scipy.io


class Dataset:

    @staticmethod
    def get_dataset_from_csv(database_name: str, database_dir: Path, verbose=False, gray=False):

        if database_name == 'CSIQ':
            nparray_img, nparray_label, dataframe = Dataset.get_CSIQdataset_from_csv(database_dir / database_name,
                                                                                     verbose, gray)
        elif database_name == 'TID2013':
            nparray_img, nparray_label, dataframe = Dataset.get_TIDdataset_from_csv(database_dir / database_name,
                                                                                    verbose, gray)
        # elif database_name == 'KADID10K':
        #     nparray_img, nparray_label, dataframe = Dataset.get_KADIDdataset_from_csv(database_dir / database_name,
        #                                                                               verbose, gray)
        elif database_name == 'LIVE2':
            nparray_img, nparray_label, dataframe = Dataset.get_LIVEdataset_from_csv(database_dir / database_name,
                                                                                     verbose, gray)
        else:
            raise Exception(f'database "{database_name}" not found')

        return nparray_img, nparray_label, dataframe

    # @staticmethod
    # def how_many_patches_gives_img(img: np.ndarray, patch_shape: Tuple[int, int] = (64, 64)) -> int:
    #     if img.ndim == 3:
    #         h, w, c = img.shape
    #     elif img.ndim == 2:
    #         h, w = img.shape
    #     else:
    #         raise Exception
    #
    #     nb_patches = (h // patch_shape[0]) * (w // patch_shape[1])
    #     return nb_patches

    @staticmethod
    def crop_any_img_in_patches(img: np.ndarray, patch_shape: Tuple[int, int] = (64, 64), verbose=True):
        """
        >>> one_img = get_quick_dataset(inRGB=False)[0, :, :, :]
        >>> Dataset.crop_any_img_in_patches()
        """
        if img.ndim == 3:
            h, w, c = img.shape
        else:
            raise Exception

        shape_multiple = h - (h % 64), w - (w % 64)
        # if verbose: print(f"img {img.shape}, shape multiple {shape_multiple}")
        img_multiple = img[:shape_multiple[0], :shape_multiple[1], :]

        rangey = range(0, img_multiple.shape[0], patch_shape[0])
        rangex = range(0, img_multiple.shape[1], patch_shape[1])
        x_tot = len(list(rangex))
        y_tot = len(list(rangey))

        nb_patches = (h // patch_shape[0]) * (w // patch_shape[1])
        # if verbose: print("print1", h, w, patch_shape, nb_patches)

        new_patches = np.empty((nb_patches, patch_shape[0], patch_shape[1], c))

        for x in rangex:
            for y in rangey:
                new_patch = img_multiple[y:y + patch_shape[0], x:x + patch_shape[1], :]
                if new_patch.shape != (64, 64, 1):
                    print(f"New patch error shape : {new_patch.shape} x={x}; y={y}")

                i_x = x // patch_shape[1]
                i_y = y // patch_shape[0]
                i_patch = (i_x * y_tot) + i_y
                new_patches[i_patch, :, :, :] = new_patch

        # if verbose: print(f"print2 nb_patches_of_img={nb_patches} size new_patches_of_img {new_patches.shape}")
        return new_patches

    @staticmethod
    def get_LIVEdataset_from_csv(database_dir: Path, verbose=False, gray=False):
        """
        >>> DB = Path('./database/')
        >>> nparray_img, nparray_label, dataframe = Dataset.get_dataset_from_csv('LIVE2', DB, verbose=True, gray=False)
        """
        MOSFILE_NAME = "dmos.csv"
        patch_shape = (64, 64)

        dataframe = pd.read_csv(database_dir / MOSFILE_NAME)
        nb_rows = dataframe.shape[0]

        nb_patches = 0
        for i, row in dataframe.iterrows():
            img_path = database_dir / row['dir'] / row['filename']
            h, w = Image.open(img_path).size
            nb_patches_tmp = ((h // patch_shape[0])*(w // patch_shape[1]))
            nb_patches += nb_patches_tmp

        # nparray_img = np.empty((nb_rows, 512, 768, 3)) if not gray else np.empty((nb_rows, 512, 768, 1))
        nparray_patches = np.empty((nb_patches, 64, 64, 3)) if not gray else np.empty((nb_patches, 64, 64, 1))
        index_nparray = 0
        nparray_label = np.empty((nb_patches, 1, 1, 1))
        # nparray_label = np.empty((nb_rows, 1, 1, 1))

        for i, row in dataframe.iterrows():
            if verbose and i % (nb_rows // 10) == 0: print(f"loading database {i + 1}/{nb_rows}...")
            img_path = database_dir / row['dir'] / row['filename']
            img_PIL = keras.preprocessing.image.load_img(img_path) if not gray else keras.preprocessing.image.load_img(
                img_path, color_mode="grayscale")
            img_np = keras.preprocessing.image.img_to_array(img_PIL)

            if i==0:
                print(f"Ploting : {img_path}")
                plt.imshow(img_np, cmap='gray'); plt.show()

            patches_of_img = Dataset.crop_any_img_in_patches(img_np, patch_shape=patch_shape)
            nb_patches_of_img = patches_of_img.shape[0]
            nparray_patches[index_nparray:index_nparray+nb_patches_of_img, :, :, :] = patches_of_img

            if i==0:
                plt.imshow(patches_of_img[0,:,:,:], cmap='gray'); plt.show()

            # print(f"img[{i}] shape={img_np.shape}")

            # nparray_patches[i, :, :, :] = img_np
            #
            quality_score = np.array([row['dmos']])
            nparray_label[index_nparray:index_nparray+nb_patches_of_img, 0, 0, 0] = np.repeat(quality_score, nb_patches_of_img)

            index_nparray += nb_patches_of_img

        if verbose:
            print(f"Shape du tensor d'images : {nparray_patches.shape}")
            print(f"shape des labels correspondants : {nparray_label.shape}")

        assert np.isfinite(nparray_label).all(), f"{nparray_label[np.isnan(nparray_label)]} value encountered"
        nparray_label = nparray_label / 100  # format du dmos de LIVE2 : bon[0;100]mauvais
        # return nparray_patches, nparray_label, dataframe
        return nparray_patches, nparray_label, dataframe

    @staticmethod
    def get_TIDdataset_from_csv(database_dir: Path, verbose=False, gray=False):
        """TID2013 database
        >>> DB = Path('./database/')
        >>> nparray_img, nparray_label, dataframe = Dataset.get_dataset_from_csv('TID2013', DB, verbose=True, gray=False)
        Shape du tensor d'images : (3000, 384, 512, 3)
        shape des labels correspondants : (3000, 1, 1, 1)
        """
        MOSFILE_NAME = "mos_with_names.csv"
        IMG_FOLDER = "distorted_images"

        dataframe = pd.read_csv(database_dir / MOSFILE_NAME)
        nb_rows = dataframe.shape[0]

        nparray_img = np.empty((nb_rows, 384, 512, 3)) if not gray else np.empty((nb_rows, 384, 512, 1))
        nparray_label = np.empty((nb_rows, 1, 1, 1))

        for i, row in dataframe.iterrows():
            if verbose and i % (nb_rows // 10) == 0: print(f"loading database {i + 1}/{nb_rows}...")
            img_path = database_dir / IMG_FOLDER / row['imgname']
            img_PIL = keras.preprocessing.image.load_img(img_path) if not gray else keras.preprocessing.image.load_img(
                img_path, color_mode="grayscale")
            img_np = keras.preprocessing.image.img_to_array(img_PIL)
            nparray_img[i, :, :, :] = img_np

            quality_score = np.array([row['mos']])
            nparray_label[i, 0, 0, 0] = quality_score

        if verbose:
            print(f"Shape du tensor d'images : {nparray_img.shape}")
            print(f"shape des labels correspondants : {nparray_label.shape}")

        nparray_label = nparray_label / 9  # format du mos de TID2013 : mauvais[0;9]bon
        return nparray_img, nparray_label, dataframe

    @staticmethod
    def get_KADIDdataset_from_csv(database_dir: Path, verbose=False, gray=False):
        MOSFILE_NAME = "CSIQ_DMOS.csv"
        IMG_FOLDER = "images"

        dataframe = pd.read_csv(database_dir / MOSFILE_NAME)
        nb_rows = dataframe.shape[0]
        for i, row in dataframe.iterrows():
            img_path = f"{str(database_dir / row['image'])}.{row['dst_type']}.{str(row['dst_lev'])}.png"

    @staticmethod
    def get_CSIQdataset_from_csv(database_dir: Path, verbose=False, gray=False):
        """Images must be in the name format [photo_name].[distorsion_type].[distorsion_level].png
        >>> DB = Path('./database/CSIQ/')
        >>> nparray_img, nparray_label, dataframe = Dataset.get_dataset_from_csv('CSIQ', DB, verbose=True, gray=False)
        Shape du tensor d'images : (866, 512, 512, 3)
        shape des labels correspondants : (866, 1, 1, 1)
        """
        MOSFILE_NAME = "CSIQ_DMOS.csv"
        IMG_FOLDER = "dst_imgs"

        dataframe = pd.read_csv(database_dir / MOSFILE_NAME)
        nb_rows = dataframe.shape[0]

        nparray_img = np.empty((nb_rows, 512, 512, 3)) if not gray else np.empty((nb_rows, 512, 512, 1))
        nparray_label = np.empty((nb_rows, 1, 1, 1))

        for i, row in dataframe.iterrows():
            img_path = f"{database_dir / IMG_FOLDER / row['image']}.{row['dst_type']}.{str(row['dst_lev'])}.png"

            img_PIL = keras.preprocessing.image.load_img(img_path) if not gray else keras.preprocessing.image.load_img(
                img_path, color_mode="grayscale")
            img_np = keras.preprocessing.image.img_to_array(img_PIL)
            # print(f"shape img np : {img_np.shape}")
            # print(f"shape nparray_img : {nparray_img.shape}")

            nparray_img[i, :, :, :] = img_np

            quality_score = np.array([row['dmos']])
            nparray_label[i, 0, 0, 0] = quality_score

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
    def data_augmentation_patching(dataset_X: np.ndarray, nb_division: Tuple[int, int], labels: np.ndarray = None,
                                   verbose=False):
        """
        >>> dataset = get_quick_dataset(inRGB=False)[0:2, :384, :, :].astype(int)
        >>> print(dataset.shape)
        (2, 384, 512, 1)
        >>> a = plt.imshow(dataset[0, :, :, 0]); plt.show()
        >>> dataset = Dataset.data_augmentation_patching(dataset, nb_division=(6, 8))
        >>> a = plt.imshow(dataset[0, :, :, 0]); plt.show()
        >>> a = plt.imshow(dataset[1, :, :, 0]); plt.show()
        """

        if nb_division == (1, 1):  # rien à faire
            return dataset_X, labels

        imgs_nb, h, w, c = dataset_X.shape
        assert h % nb_division[0] == 0, f"{h} rows is not evenly divisible by {nb_division[0]}"
        assert w % nb_division[1] == 0, f"{w} cols is not evenly divisible by {nb_division[1]}"

        nb_duplicate = nb_division[0] * nb_division[1]
        nb_patch = dataset_X.shape[0] * nb_duplicate
        if verbose: print(f"[DATA AUGMENT] Processing to get x{nb_duplicate}={nb_patch} data by tiling the images...")

        patch_shape = (dataset_X.shape[1] // nb_division[0], dataset_X.shape[2] // nb_division[1])
        # new_X = (dataset_X.reshape(h // nb_division, nb_division, -1, nb_division)
        #          .swapaxes(1, 2)
        #          .reshape(-1, nb_division, nb_division))

        new_X = np.zeros((nb_patch, patch_shape[0], patch_shape[1], dataset_X.shape[3]))
        if verbose: print(f"Creation np.zeros for patchs : {new_X.shape}")

        rangex = range(0, dataset_X.shape[2], patch_shape[1])  # pas certain de si c'est shape[1] ou shape[2]
        rangey = range(0, dataset_X.shape[1], patch_shape[0])
        x_tot = len(list(rangex))
        y_tot = len(list(rangey))

        for i_img in range(dataset_X.shape[0]):
            if verbose and i_img % 20 == 0: print(f"Tiling img {i_img + 1}/{dataset_X.shape[0]}...")
            for x in rangex:
                for y in rangey:
                    new_patch = dataset_X[i_img, y:y + patch_shape[0], x:x + patch_shape[1], :]
                    assert (new_patch.shape != (64, 64, 1) and new_patch.shape != (64, 64, 3)), ...
                    f"New patch error shape : {new_patch.shape} with i_img={i_img}; x={x}; y={y} "

                    i_x = x // patch_shape[1]
                    i_y = y // patch_shape[0]
                    i_patch = (i_img * x_tot * y_tot) + (i_x * y_tot) + i_y
                    new_X[i_patch, :, :, :] = new_patch

        if verbose: print(f"[DATA AUGMENT] Finished.")
        if labels is not None:
            labels = np.repeat(labels.flatten(), nb_duplicate)[:, np.newaxis, np.newaxis, np.newaxis]
            return new_X, labels

        return new_X

# from dataset import Dataset
# from utiles import *
# import matplotlib.pyplot as plt
# datasetoriginal = get_quick_dataset(inRGB=True)
# dataset = Dataset.data_augmentation_patching(datasetoriginal[0:1,:,:,:], nb_division=2)
