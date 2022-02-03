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

        if normalization_type == 'nothing':
            return dataset_X
        elif normalization_type == 'scale_0_1':
            return Normalization.pixelsScalling(dataset_X)
        elif normalization_type == 'tight_scale_0_1':
            return Normalization.pixel_tight_scaling(dataset_X)
        elif normalization_type == 'centering':
            return Normalization.center(dataset_X)
        elif normalization_type == 'centering_reduced':
            return Normalization.center_reduced(dataset_X)
        elif normalization_type == 'standardization':
            return Normalization.standardize(dataset_X)
        elif normalization_type == 'DoG':
            return Normalization.difference_of_gaussian(dataset_X)
        elif normalization_type == 'LCN':
            new_X = Normalization.local_contrast_norm(dataset_X)
        elif normalization_type == 'LRN':
            new_X = Normalization.local_response_norm2(dataset_X)
        elif normalization_type == 'histogram_eq':
            new_X = Normalization.histogram_equalization(dataset_X)
        elif normalization_type == 'CLAHE':
            new_X = Normalization.CLA_hist_equalization(dataset_X)
        elif normalization_type == 'ZCAwhiten':
            new_X = Normalization.ZCA_whiten(dataset_X)
        elif normalization_type == 'simplified_whitening':
            new_X = Normalization.simplified_whitening(dataset_X)
        else:
            print("[preprocess_images()] argument normalization_type not found.")
            return dataset_X

        return Normalization.pixel_tight_scaling(new_X)

    @staticmethod
    def pixelsScalling(dataset: np.ndarray) -> np.ndarray:
        return dataset / 255.0

    @staticmethod
    def pixel_tight_scaling(dataset: np.ndarray) -> np.ndarray:
        """Remet les valeurs avec min_Dataset = 0, max_Dataset = 1

        >>> dataset = np.arange(3*8*8).reshape(3,8,8,1)
        >>> dataset = Normalization.pixel_tight_scaling(dataset)
        >>> print(dataset.shape)
        (3, 8, 8, 1)
        >>> print(np.amax(dataset))
        [1. 1. 1.]
        >>> print(np.amin(dataset))
        [0. 0. 0.]
        """

        dataset_max = np.amax(dataset)
        dataset_min = np.amin(dataset)

        tight_scaled = (dataset - dataset_min) / (dataset_max - dataset_min)
        return np.nan_to_num(tight_scaled)

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
        >>> a = plt.imshow(ker)
        >>> plt.show()
        """

        probs = [exp(-z * z / (2 * sigma * sigma)) / sqrt(2 * pi * sigma * sigma) for z in
                 range(-halfsize, halfsize + 1)]
        kernel = np.outer(probs, probs)

        return kernel

    @staticmethod
    def center(dataset_X: np.ndarray, perpixel) -> np.ndarray:
        """
        >>> dataset = get_quick_dataset(inRGB=True)
        >>> res = Normalization.center(dataset)
        >>> a = plt.imshow(res[3, :, :, :]); a = plt.axis('off');
        >>> plt.show()
        """

        if perpixel:
            mean = Normalization.get_per_pixel_mean(dataset_X)
        else:
            mean = np.mean(dataset_X, axis=(0, 1, 2))

        return dataset_X - mean

    @staticmethod
    def center_reduced(dataset_X: np.ndarray) -> np.ndarray:
        """
        Centering data between -1 and 1
        >>> dataset = get_quick_dataset(inRGB=True)
        >>> res = Normalization.center_reduced(dataset)
        >>> a = plt.imshow(res[3, :, :, :].astype(float)); a = plt.axis('off');
        >>> plt.show()
        """
        dataset_X = Normalization.pixel_tight_scaling(dataset_X)
        return dataset_X - np.mean(dataset_X, axis=(0, 1, 2))

    @staticmethod
    def standardize(dataset_X: np.ndarray, perpixel=False) -> np.ndarray:
        """
        >>> dataset = get_quick_dataset(inRGB=False)
        >>> res = Normalization.preprocess_images(dataset, 'standardization')
        >>> a = plt.imshow(res[3, :, :, :], cmap="gray"); a = plt.axis('off'); a = plt.colorbar()
        >>> plt.show()
        """
        centered_data = Normalization.center(dataset_X, perpixel=False)
        new_data = centered_data / np.std(centered_data, axis=(0, 1, 2))

        return new_data

    @staticmethod
    def simplified_whitening(dataset_X: np.ndarray) -> np.ndarray:
        #Source: https://medium.com/@dibyadas/visualizing-different-normalization-techniques-84ea5cc8c378
        """ http://yeephycho.github.io/2016/08/03/normalizations_in_neural_networks/
        Source :
        >>> dataset = get_quick_dataset(inRGB=False)
        >>> res = Normalization.preprocess_images(dataset, 'simplified_whitening')
        >>> a = plt.imshow(res[3, :, :, :], cmap="gray"); a = plt.axis('off'); a = plt.colorbar()
        >>> plt.show()
        """
        centered_data = Normalization.center(dataset_X, perpixel=True)
        new_data = centered_data / np.std(centered_data, axis=0)
        return new_data
        # assert dataset_X.shape[3] == 1, "RGB images are not allowed to be simplified_whiten"
        #
        # mean_filter = np.ones((dataset_X.shape[0], 1, 1, 1)) / dataset_X.shape[0]
        # squared_images = tf.nn.conv2d(Normalization.center(dataset_X, perpixel=True) ** 2, tf.convert_to_tensor(mean_filter), strides=[1, 1, 1, 1], padding="SAME")
        # standard_deviation = np.sqrt(squared_images.numpy())
        # new_X = Normalization.center(dataset_X, perpixel=True) / standard_deviation
        # return new_X


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
            new_X[i_img, :, :, 0] = locally_centered / locally_sqrt  # surement [i_img, :, :, :] si rgb
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
    def local_response_norm2(input, size=3, alpha=1, beta=2, k=1):
        """Applies intra channel LRN

        Est l'exact copié collé de ce code colab par moi
        https://colab.research.google.com/drive/1SiI7jeOu96eewaQCbvS2gyT8LRtSvlQK?usp=sharing
        qui marche beaucoup mieux

        Applies local response normalization over an input signal composed of
        several input planes, where channels occupy the second dimension.
        Applies normalization across channels.
        See :class:`~torch.nn.LocalResponseNorm` for details.
        >>> dataset = get_quick_dataset(inRGB=False)
        >>> res = Normalization.preprocess_images(dataset, 'LRN')
        >>> a = plt.imshow(res[3, :, :, :], cmap="gray"); a = plt.axis('off'); a = plt.colorbar()
        >>> plt.show()
        """

        # input = input.transpose(0, 3, 1, 2)
        dim = len(input.shape)

        input_tensor = tf.convert_to_tensor(input)
        if dim < 3:
            raise ValueError('Expected 3D or higher dimensionality \
                             input (got {} dimensions)'.format(dim))
        div = tf.expand_dims((input_tensor * input_tensor), -1)

        sizes = tf.shape(input_tensor)
        div = tf.cast(div, tf.float32)
        # div = tf.reshape(div, (sizes[0], 1, sizes[1], sizes[2], -1))
        # div = tf.reshape(div)
        # paddings = [[size // 2, (size - 1) // 2], [0, 0], [0, 0]]
        # div = tf.pad(div, paddings)
        div = tf.squeeze(tf.nn.avg_pool3d(div, (1, size, size), strides=(1, 1, 1), padding="SAME"), -1)
        div = tf.cast(div, tf.float64)
        # div = tf.reshape(div, sizes)
        div = (div * alpha + k) ** beta
        res = (tf.cast(input_tensor, tf.float64) / div)  #.transpose(0, 2, 3, 1)
        return res