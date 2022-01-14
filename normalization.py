import doctest
from numpy import pi, exp, sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import misc, signal
from PIL import Image

"""
This class will be used to gather all the normalization techniques that are finalized (Working)

"""


class Normalization():

    @staticmethod
    def pixelsScalling(dataset: np.ndarray) -> np.ndarray:
        return dataset / 255.0

    @staticmethod
    def get_per_pixel_mean(dataset: np.ndarray) -> np.ndarray:
        mean = np.mean(dataset, axis=0)
        return mean

    @staticmethod
    def get_covariance_from_multiple_images(dataset: np.ndarray) -> np.ndarray:
        cov = np.cov(dataset, rowvar=False)
        return cov

    @staticmethod
    def get_4D_dataset_as_2D(dataset: np.ndarray) -> np.ndarray:
        """For instance, this converts 100 RGB images (100, 216, 216, 3) to (100, 216x216x3)"""
        return dataset.reshape(dataset.shape[0], dataset.shape[1] * dataset.shape[2] * dataset.shape[3])

    @staticmethod
    def get_gaussian_kernel(halfsize, sigma=2.0):
        """
        >>> ker = Normalization.get_gaussian_kernel(halfsize=4) # (i_img, h_img, w_img, colors)
        >>> a = plt.imshow(ker) #doctest:+ELLIPSIS
        >>> plt.show() #doctest:+ELLIPSIS
        """

        probs = [exp(-z * z / (2 * sigma * sigma)) / sqrt(2 * pi * sigma * sigma) for z in
                 range(-halfsize, halfsize + 1)]
        kernel = np.outer(probs, probs)

        return kernel

    @staticmethod
    def center(dataset: np.ndarray) -> np.ndarray:
        mean = Normalization.get_per_pixel_mean(dataset)
        return dataset - mean

    @staticmethod
    def standardize(dataset: np.ndarray) -> np.ndarray:
        """Using per-pixel standard deviation"""
        return Normalization.center(dataset) / np.std(dataset, axis=0)

    @staticmethod
    def ZCA_whiten(dataset_X: np.ndarray, epsilon=0.1) -> np.ndarray:
        """
        >>> asc = misc.ascent()
        >>> asc = asc[0::16, 0::16]
        >>> Image.fromarray(asc).show()
        >>> newasc = asc[np.newaxis, :, :, np.newaxis]
        >>> dataset = np.append(newasc, newasc.T, axis=0)
        >>> res = Normalization.ZCA_whiten(dataset)
        >>> Image.fromarray(res[0, :, :, 0]*255).show()
        """
        original_shape = dataset_X.shape
        X_0to1 = Normalization.pixelsScalling(dataset_X)
        X_flattened = Normalization.get_4D_dataset_as_2D(X_0to1)
        cov = Normalization.get_covariance_from_multiple_images(X_flattened)
        U, S, V = np.linalg.svd(cov)
        X_ZCA = U.dot(np.diag(1.0 / np.sqrt(S + epsilon))).dot(U.T).dot(X_flattened.T).T
        X_ZCA_rescaled_0to1 = (X_ZCA - X_ZCA.min()) / (X_ZCA.max() - X_ZCA.min())
        X_whiten = X_ZCA_rescaled_0to1.reshape(*original_shape)

        # firstimg = X_ZCA_rescaled_0to255[1, :].reshape(*shape_X[1:])
        # plt.figure(figsize=(20, 20))
        # plt.imshow(firstimg)
        # plt.show()
        return X_whiten

    @staticmethod
    def difference_of_gaussian(dataset_X: np.ndarray, sigma1: float = 1, sigma2: float = 2) -> np.ndarray:
        """
        >>> asc = misc.ascent()
        >>> Image.fromarray(asc).show()
        >>> newasc = asc[np.newaxis, :, :, np.newaxis]
        >>> res = Normalization.difference_of_gaussian(newasc)
        >>> Image.fromarray(res[0, :, :, 0]).show()
        """

        for i in range(dataset_X.shape[0]):
            blur1 = gaussian_filter(dataset_X[i, :, :, :], sigma=sigma1)
            blur2 = gaussian_filter(dataset_X[i, :, :, :], sigma=sigma2)
            diff = blur1 - blur2
            dataset_X[i, :, :, :] = diff

        return dataset_X

    @staticmethod
    def local_contrast_norm(dataset_X: np.ndarray, mean_scalar=1.0) -> np.ndarray:
        """
        >>> img = misc.face()
        >>> img1 = img[np.newaxis, :, :, :]
        >>> res = Normalization.local_contrast_norm(img1)
        >>> a = plt.imshow(res[0, :, :, :])# - res.min()/(res.max()-res.min()))
        >>> plt.show()

        """
        ker = Normalization.get_gaussian_kernel(4)

        new_X = np.zeros(dataset_X.shape)

        for i_img in range(dataset_X.shape[0]):
            for i_canal in range(dataset_X.shape[3]):
                X = dataset_X[i_img, :, :, i_canal]
                if X.ndim != 2:
                    raise ValueError("Input must be a 2D matrix.")
                Xh = signal.convolve2d(X, ker, mode='same', boundary='symm')
                locally_centered = X - Xh
                locally_sqrt = np.sqrt(signal.convolve2d(np.power(locally_centered, 2.0),
                                                         ker, mode='same', boundary='symm'))
                thresh = np.exp(np.log(locally_sqrt + np.power(2.0, -5)).mean(axis=-1))
                locally_sqrt = locally_sqrt * np.greater(locally_sqrt - thresh.reshape(-1, 1), 0)
                locally_sqrt += 1.0 * np.equal(locally_sqrt, 0.0)
                new_X[i_img, :, :, i_canal] = locally_centered / locally_sqrt
        return new_X
