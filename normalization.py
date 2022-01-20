import doctest
from numpy import pi, exp, sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import misc, signal
from skimage import data
from skimage import exposure
from utiles import *
from skimage.color import rgb2gray
from PIL import Image

"""
This class will be used to gather all the normalization techniques that are finalized (Working)

"""


class Normalization:

    @staticmethod
    def preprocess_images(dataset_X: np.ndarray, normalization_type: str) -> np.ndarray:

        if normalization_type == 'scale_0_1':
            return Normalization.pixelsScalling(dataset_X)
        elif normalization_type == 'centering':
            return Normalization.center(dataset_X)
        elif normalization_type == 'standardization':
            return Normalization.standardize(dataset_X)
        elif normalization_type == 'LCN':
            return Normalization.local_contrast_norm(dataset_X)
        elif normalization_type == 'LRN':
            return Normalization.intra_channel_local_response_normalization(dataset_X)
        elif normalization_type == 'histogram eq':
            return Normalization.histogram_equalization(dataset_X)
        elif normalization_type == 'CLA hist eq':
            return Normalization.CLA_hist_equalization(dataset_X)
        elif normalization_type == 'ZCA whiten':
            return Normalization.ZCA_whiten(dataset_X)

        print("[preprocess_images()] argument normalization_type not found.")
        return dataset_X

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
    def center(dataset_X: np.ndarray) -> np.ndarray:
        """
        >>> dataset = get_quick_dataset(inRGB=True)
        >>> res = Normalization.center(dataset)
        >>> a = plt.imshow(res[3, :, :, :], cmap="gray"); a = plt.axis('off'); a = plt.colorbar()
        >>> plt.show()
        """
        mean = Normalization.get_per_pixel_mean(dataset_X)
        return dataset_X - mean

    @staticmethod
    def standardize(dataset_X: np.ndarray) -> np.ndarray:
        """Using per-pixel standard deviation
        >>> dataset = get_quick_dataset(inRGB=False)
        >>> res = Normalization.standardize(dataset)
        >>> a = plt.imshow(res[3, :, :, :], cmap="gray"); a = plt.axis('off'); a = plt.colorbar()
        >>> plt.show()
        """
        return Normalization.center(dataset_X) / np.std(dataset_X, axis=0)

    @staticmethod
    def simplified_whitening(dataset_X: np.ndarray) -> np.ndarray:
        """
        Must take in gray img
        Source: https://medium.com/@dibyadas/visualizing-different-normalization-techniques-84ea5cc8c378
        >>> dataset = get_quick_dataset(inRGB=False)
        >>> res = Normalization.simplified_whitening(dataset)
        >>> a = plt.imshow(res[3, :, :, :], cmap="gray"); a = plt.axis('off'); a = plt.colorbar()
        >>> plt.show()
        """

        assert dataset_X.shape[3] == 1, "RGB images are not allowed to be simplified_whiten"

        mean_filter = np.ones((dataset_X.shape[0], 1, 1, 1)) / dataset_X.shape[0]
        squared_images = tf.nn.conv2d(Normalization.center(dataset_X) ** 2, tf.convert_to_tensor(mean_filter), strides=[1, 1, 1, 1], padding="SAME")
        standard_deviation = np.sqrt(squared_images.numpy())
        new_X = Normalization.center(dataset_X) / standard_deviation
        return new_X


    @staticmethod
    def ZCA_whiten(dataset_X: np.ndarray, epsilon=0.1) -> np.ndarray:
        """
        # >>> asc = misc.ascent()
        # >>> asc = asc[0::8, 0::8]
        # >>> Image.fromarray(asc).show()
        # >>> newasc = asc[np.newaxis, :, :, np.newaxis]
        # >>> dataset = np.append(newasc, newasc.T, axis=0)
        # >>> res = Normalization.ZCA_whiten(dataset)
        # >>> Image.fromarray(res[0, :, :, 0]*255).show()

        >>> dataset = get_quick_dataset(inRGB=True)[:, ::8, ::8, :]
        >>> res = Normalization.ZCA_whiten(dataset)
        >>> a = plt.imshow(res[3, :, :, :], cmap="gray"); a = plt.axis('off'); a = plt.colorbar()
        >>> plt.show()
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
        >>> dataset = get_quick_dataset(inRGB=False)
        >>> res = Normalization.difference_of_gaussian(dataset)
        >>> a = plt.imshow(res[1, :, :, :], cmap="gray"); a = plt.axis('off'); a = plt.colorbar()
        >>> plt.show()
        """

        for i in range(dataset_X.shape[0]):
            blur1 = gaussian_filter(dataset_X[i, :, :, :], sigma=sigma1)
            blur2 = gaussian_filter(dataset_X[i, :, :, :], sigma=sigma2)
            diff = blur1 - blur2
            dataset_X[i, :, :, :] = diff

        return dataset_X

    @staticmethod
    def local_contrast_norm(dataset_X: np.ndarray, mean_scalar=1.0, enableRGB=False) -> np.ndarray:
        """
        >>> dataset = get_quick_dataset(inRGB=False)
        >>> res = Normalization.local_contrast_norm(dataset)
        >>> a = plt.imshow(res[3, :, :, :], cmap="gray"); a = plt.axis('off'); a = plt.colorbar()
        >>> plt.show()
        """
        ker = Normalization.get_gaussian_kernel(2)

        new_X = np.zeros(dataset_X.shape)

        for i_img in range(dataset_X.shape[0]):
            # for i_canal in range(dataset_X.shape[3])
            X = dataset_X[i_img, :, :, 0]
            if X.ndim != 2:
                raise ValueError("Input must be a 2D matrix.")
            Xh = signal.convolve2d(X, ker, mode='same', boundary='symm')
            locally_centered = X - Xh
            locally_sqrt = np.sqrt(signal.convolve2d(np.power(locally_centered, 2.0),
                                                     ker, mode='same', boundary='symm'))
            thresh = np.exp(np.log(locally_sqrt + np.power(2.0, -5)).mean(axis=-1))
            locally_sqrt = locally_sqrt * np.greater(locally_sqrt - thresh.reshape(-1, 1), 0)
            locally_sqrt += 1.0 * np.equal(locally_sqrt, 0.0)
            new_X[i_img, :, :, :] = locally_centered / locally_sqrt
        return new_X

    @staticmethod
    def histogram_equalization(dataset_X: np.ndarray, enableRGB=False) -> np.ndarray:
        """
        >>> dataset = get_quick_dataset(inRGB=False)
        >>> a = plt.imshow(dataset[3, :, :, :], cmap="gray"); a = plt.axis('off'); a = plt.colorbar()
        >>> plt.show()
        >>> res = Normalization.histogram_equalization(dataset)
        >>> a = plt.imshow(res[3, :, :, :], cmap="gray"); a = plt.axis('off'); a = plt.colorbar()
        >>> plt.show()
        """
        new_X = np.zeros(dataset_X.shape)

        for i_img in range(dataset_X.shape[0]):
            img = dataset_X[i_img, :, :, 0]
            new_X[i_img, :, :, 0] = exposure.equalize_hist(img)

        return new_X

    @staticmethod
    def CLA_hist_equalization(dataset_X: np.ndarray, enableRGB=False) -> np.ndarray:
        """
        >>> dataset = get_quick_dataset(inRGB=False)
        >>> a = plt.imshow(dataset[3, :, :, :], cmap="gray"); a = plt.axis('off'); a = plt.colorbar()
        >>> plt.show()
        >>> res = Normalization.CLA_hist_equalization(dataset)
        >>> a = plt.imshow(res[3, :, :, :], cmap="gray"); a = plt.axis('off'); a = plt.colorbar()
        >>> plt.show()
        """
        new_X = np.zeros(dataset_X.shape)

        for i_img in range(dataset_X.shape[0]):
            img = dataset_X[i_img, :, :, 0].astype(np.uint8)
            new_X[i_img, :, :, 0] = exposure.equalize_adapthist(img, clip_limit=0.03)

        return new_X

    # @staticmethod
    # def local_response_normalization(dataset_X: np.ndarray) -> np.ndarray:
    #     """
    #     To be precise, this is the inter-channel LRN.
    #     There is another LRN explained in paper which is the intra-channel LRN, which only needs one image
    #     """
    #     tf.nn.local_response_normalization(
    # input, depth_radius=5, bias=1, alpha=1, beta=0.5, name=None
    #     )
    #

    @staticmethod
    def intra_channel_local_response_normalization(dataset_X: np.ndarray, enableRGB=False, halfsize=2, bias=1,
                                                   alpha=0.0001, beta=0.5) -> np.ndarray:
        """
        Intra channel local response normalization
        :param dataset_X:
        :param enableRGB:
        :param halfsize: kernel halfsize
        :param bias: added to avoid any singularities (division by zero)
        :param alpha: normalization constant
        :param beta: contrasting constant
        >>> dataset = get_quick_dataset(inRGB=False)
        >>> res = Normalization.intra_channel_local_response_normalization(dataset)
        >>> a = plt.imshow(res[3, :, :, :], cmap="gray"); a = plt.axis('off'); a = plt.colorbar()
        >>> plt.show()
        """
        new_X = np.zeros(dataset_X.shape)

        probs = [1 for z in range(-halfsize, halfsize + 1)]
        sum_kernel = np.outer(probs, probs)

        for i_img in range(dataset_X.shape[0]):
            img = dataset_X[i_img, :, :, 0].astype(np.uint8)

            imgsquare = img ** 2

            new_X[i_img, :, :, 0] = 1 / (bias + (alpha * signal.convolve2d(imgsquare, sum_kernel, mode='same', boundary='symm')) ** beta)

        return new_X

    @staticmethod
    def intra_channel_local_response_normalization(dataset_X: np.ndarray, enableRGB=False, halfsize=2, bias=1,
                                                   alpha=0.0001, beta=0.5) -> np.ndarray:
        pass
        # dataset_X * dataset_X
        # sizes = input