from pathlib import Path
from typing import List, Optional

import rasterio
from torch.utils.data import Dataset

import glob
import warnings
import torchvision.transforms.functional as TF
import h5py
from datetime import datetime

from typing import List
import torch
import numpy as np


def get_means_stds_missing_values(training_years: List[int]):
    """_summary_ Returns mean and std values as tensor, computed on unaugmented and unstandardized
    data of the indicated training years. We don't clip values, because min/max did not diverge
    much from the 0.1 and 99.9 percentiles. Some variables are not standardized, indicated by mean=0, std=1.
    These are specifically: All variables indicating a direction in degrees
    (wind direction, aspect, forecast wind direction), and the categorical land cover type.

    Args:
        training_years (_type_): _description_

    Returns:
        _type_: _description_
    """

    stats_per_training_year_combo = {
        (2018, 2019): {
        'means': np.array([
            1.95905826e+03,  2.94404070e+03,  1.80315792e+03,  4.18785304e+03,
            2.20147914e+03,  4.75643503e-01,  3.40356332e+00,  5.20851084e-03,
            2.82528452e+02,  2.99264656e+02,  6.81236923e+01,  5.48062173e-03,
            6.72659479e+00,  2.51996541e-03,  1.53416361e+03, -1.13553723e+00,
            8.92881266e+00,  1.84884483e+01,  1.59374234e+00,  9.44323569e-02,
            1.84031356e+01,  5.57368450e-03,  1.48318570e+01], dtype=np.float32),
        'stds': np.array([
            1.13697378e+03, 1.63707682e+03, 1.89291266e+03, 2.21105881e+03,
            1.12833037e+03, 2.15286520e+00, 1.39548951e+00, 7.07312261e-01,
            6.87025186e+00, 7.96616357e+00, 1.89715391e+01, 2.10764239e-03,
            6.75470828e+00, 7.05844663e-01, 7.69800231e+02, 1.88290894e+00,
            3.14455922e+00, 4.32875914e+01, 1.07801499e+00, 5.67901367e-01,
            6.98550201e+00, 1.92365459e-03, 5.52618558e+00], dtype=np.float32),
        'missing_values': np.array([
            0.02567231, 0.0256701, 0.02566863, 0.02243902, 0.02243973,
            0.01035774, 0.01035774, 0.01035774, 0.01035774, 0.01035774,
            0.01035774, 0.01035774, 0.00783755, 0.00767712, 0.00767712,
            0.01404397, 0., 0., 0., 0.,
            0., 0., 0.99896296], dtype=np.float32)},
        (2018, 2020): {'means': np.array([
            1.92527729e+03,  2.90639462e+03,  1.79803198e+03,  4.17896804e+03,
            2.16614586e+03,  4.31776715e-01,  3.53074797e+00,  7.87432855e-03,
            2.82827927e+02,  2.99377397e+02,  7.06674571e+01,  5.32125263e-03,
            6.85055885e+00,  1.44364776e-03,  1.51493429e+03, -1.69149999e+00,
            8.85224324e+00,  1.14009785e+01,  1.64494010e+00,  1.19060594e-01,
            1.84249891e+01,  5.34158917e-03,  1.48273628e+01], dtype=np.float32),
        'stds': np.array([
            1.13771183e+03, 1.61051542e+03, 1.86898895e+03, 2.22301358e+03,
            1.09801944e+03, 2.21079146e+00, 1.52217285e+00, 7.07369022e-01,
            7.47886968e+00, 8.45205917e+00, 1.93343283e+01, 2.17756746e-03,
            6.68841515e+00, 7.06107475e-01, 7.96954308e+02, 1.83049817e+00,
            3.29888042e+00, 3.30864516e+01, 1.19640682e+00, 5.41434581e-01,
            7.37697721e+00, 1.90372161e-03, 5.50970354e+00], dtype=np.float32),
        'missing_values': np.array([
            0.020201, 0.02021083, 0.02020913, 0.04801165, 0.04801215,
            0.01282664, 0.01282664, 0.01282664, 0.01282664, 0.01282664,
            0.01282664, 0.01282664, 0.01241417, 0.01219699, 0.01219699,
            0.01726448, 0., 0., 0., 0.,
            0., 0., 0.99835655], dtype=np.float32)},
        (2018, 2021): {'means': np.array([
            1.89013022e+03,  2.96940208e+03,  1.80725363e+03,  4.50489531e+03,
            2.31480127e+03,  5.02559433e-01,  3.45041794e+00,  1.27139445e-02,
            2.82806380e+02,  2.99358849e+02,  6.84029260e+01,  5.54077946e-03,
            7.71963056e+00,  1.48636202e-03,  1.51864017e+03, -2.87844549e+00,
            8.39885028e+00,  9.27182969e+00,  1.50171912e+00,  6.32564205e-02,
            1.85506226e+01,  5.47539956e-03,  1.47133578e+01], dtype=np.float32),
        'stds': np.array([
            1.17527575e+03, 1.68449116e+03, 1.98719945e+03, 2.24481517e+03,
            1.12442274e+03, 2.26782957e+00, 1.36958104e+00, 7.07592235e-01,
            6.49979053e+00, 7.63064073e+00, 1.89506910e+01, 1.99923623e-03,
            7.10935154e+00, 7.06734174e-01, 7.28691576e+02, 1.61764224e+00,
            3.42863222e+00, 3.32588535e+01, 1.03051749e+00, 4.07503144e-01,
            6.56328211e+00, 1.67899092e-03, 5.49708016e+00], dtype=np.float32),
        'missing_values': np.array([
            0.06689008, 0.06688501, 0.06688439, 0.09876232, 0.09876266,
            0.00874745, 0.00874745, 0.00874745, 0.00874745, 0.00874745,
            0.00874745, 0.00874745, 0.00410403, 0.00401978, 0.00401978,
            0.01257587, 0., 0., 0., 0.,
            0., 0., 0.99811762], dtype=np.float32)},
        (2019, 2020): {'means': np.array([
            1.93210708e+03,  2.91370171e+03,  1.83706324e+03,  4.08953980e+03,
            2.10045986e+03,  4.94984735e-01,  3.53618060e+00,  4.07106577e-03,
            2.83407079e+02,  2.99610979e+02,  6.97037281e+01,  5.45841688e-03,
            6.50879203e+00,  1.75418619e-03,  1.44461686e+03, -8.59165131e-01,
            9.00657070e+00,  8.83954683e+00,  1.67522409e+00,  1.24404141e-01,
            1.86144498e+01,  5.38086557e-03,  1.48645348e+01], dtype=np.float32),
        'stds': np.array([
            1.12339811e+03, 1.63906793e+03, 1.88834784e+03, 2.14790877e+03,
            1.03149639e+03, 2.25637560e+00, 1.53103092e+00, 7.07038206e-01,
            8.01997924e+00, 8.95414067e+00, 2.04390438e+01, 2.30770483e-03,
            6.44091576e+00, 7.05879558e-01, 7.75084680e+02, 1.75367983e+00,
            3.19309720e+00, 1.77037349e+01, 1.21298008e+00, 5.50726106e-01,
            7.87039317e+00, 2.06457111e-03, 5.50900718e+00], dtype=np.float32),
        'missing_values': np.array([
            0.02618726, 0.02620377, 0.02620163, 0.0630706, 0.06307114,
            0.01363767, 0.01363767, 0.01363767, 0.01363767, 0.01363767,
            0.01363767, 0.01363767, 0.01766835, 0.01737764, 0.01737764,
            0.01816856, 0., 0., 0., 0.,
            0., 0., 0.99843441], dtype=np.float32)},
        (2019, 2021): {'means': np.array([
            1.88147144e+03,  3.00731959e+03,  1.85633292e+03,  4.56103236e+03,
            2.31152573e+03,  5.99824717e-01,  3.42193092e+00,  1.07399193e-02,
            2.83410161e+02,  2.99599435e+02,  6.64147507e+01,  5.77996581e-03,
            7.73302100e+00,  1.83252047e-03,  1.44600073e+03, -2.50143930e+00,
            8.36983739e+00,  5.62677766e+00,  1.47282063e+00,  4.51032926e-02,
            1.88067298e+01,  5.57455804e-03,  1.46989234e+01], dtype=np.float32),
        'stds': np.array([
            1.17682375e+03, 1.74877733e+03, 2.06062268e+03, 2.16947243e+03,
            1.06252645e+03, 2.33839504e+00, 1.31010010e+00, 7.07336368e-01,
            6.73165592e+00, 7.86586044e+00, 1.99876580e+01, 2.07340496e-03,
            7.04371793e+00, 7.06763509e-01, 6.70725589e+02, 1.41847911e+00,
            3.37572336e+00, 1.66397698e+01, 9.72962912e-01, 3.50080317e-01,
            6.80142805e+00, 1.77658924e-03, 5.49124080e+00], dtype=np.float32),
        'missing_values': np.array([
            0.09324284, 0.09323855, 0.0932379, 0.13653821, 0.13653853,
            0.00786934, 0.00786934, 0.00786934, 0.00786934, 0.00786934,
            0.00786934, 0.00786934, 0.00616083, 0.00605492, 0.00605492,
            0.01153656, 0., 0., 0., 0.,
            0., 0., 0.9980986], dtype=np.float32)},
        (2020, 2021): {'means': np.array([
            1.87555188e+03,  2.94560341e+03,  1.83152387e+03,  4.41762912e+03,
            2.23468240e+03,  5.14182028e-01,  3.53935693e+00,  1.14206561e-02,
            2.83397348e+02,  2.99592787e+02,  6.94810785e+01,  5.52133545e-03,
            7.50684449e+00,  1.02162830e-03,  1.45767808e+03, -2.57062415e+00,
            8.48965669e+00,  3.19498037e+00,  1.56481648e+00,  8.62800254e-02,
            1.86874840e+01,  5.34801398e-03,  1.47533915e+01], dtype=np.float32),
        'stds': np.array([
            1.16304777e+03, 1.68253524e+03, 1.97741772e+03, 2.19727741e+03,
            1.05329146e+03, 2.32933992e+00, 1.46831775e+00, 7.07383262e-01,
            7.38007815e+00, 8.37759275e+00, 1.99833888e+01, 2.15409146e-03,
            6.88269677e+00, 7.06702488e-01, 7.35189217e+02, 1.53372656e+00,
            3.44095035e+00, 8.56098760e+00, 1.13319484e+00, 4.04395714e-01,
            7.25420162e+00, 1.80867836e-03, 5.49436248e+00], dtype=np.float32),
        'missing_values': np.array([
            0.06439008, 0.06439824, 0.06439709, 0.1217506, 0.12175085,
            0.01114208, 0.01114208, 0.01114208, 0.01114208, 0.01114208,
            0.01114208, 0.01114208, 0.01120559, 0.01102537, 0.01102537,
            0.01554857, 0., 0., 0., 0.,
            0., 0., 0.99780835], dtype=np.float32)}}

    years_tuple = tuple(training_years)
    means = stats_per_training_year_combo[years_tuple]["means"]
    stds = stats_per_training_year_combo[years_tuple]["stds"]
    missing_values = stats_per_training_year_combo[years_tuple]["missing_values"]

    # Zero out means and stds for degree-based features and the categorical land cover type variable
    features_to_not_standardize = get_indices_of_degree_features() + [16]

    means[features_to_not_standardize] = 0
    stds[features_to_not_standardize] = 1

    return means, stds, missing_values


def get_indices_of_degree_features():
    """
    :return: Indices of features that take values in [0,360] and thus will be transformed via sin

    """
    return [7, 13, 19]



class FireSpreadDataset(Dataset):
    def __init__(
            self,
            dataset_path,
            data_list_path=None,   # unused — splits by year
            crop_size=None,
            split: str = "train",
            transforms=None,
            n_leading_observations: int = 1,
            crop_side_length: int = 128,
            remove_duplicate_features: bool = False,
            load_from_hdf5: bool = True,
            n_leading_observations_test_adjustment: Optional[int] = None,
            features_to_keep: Optional[List[int]] = None,
            return_doy: bool = False,
    ):
        super().__init__()

        # Unified interface params (dataset_path replaces data_dir;
        # crop_size and transforms are unused, kept for interface compatibility)
        self.data_dir = dataset_path

        self.return_doy = return_doy
        self.features_to_keep = features_to_keep
        self.remove_duplicate_features = remove_duplicate_features

        if split == "train":
            self.is_train = True
            self.included_fire_years = [2018, 2019]
        elif split == "val":
            self.is_train = False
            self.included_fire_years = [2020]
        elif split == "test":
            self.is_train = False
            self.included_fire_years = [2021]
        else:
            raise ValueError(split)

        self.stats_years = [2018, 2019]

        self.load_from_hdf5 = load_from_hdf5
        self.crop_side_length = crop_side_length
        self.n_leading_observations = n_leading_observations
        self.n_leading_observations_test_adjustment = n_leading_observations_test_adjustment

        self.validate_inputs()

        # Compute how many samples to skip in the test set, to make it look like it would with n_leading_observations set to this value.
        if self.n_leading_observations_test_adjustment is None:
            self.skip_initial_samples = 0
        else:
            self.skip_initial_samples = self.n_leading_observations_test_adjustment - self.n_leading_observations
            if self.skip_initial_samples < 0:
                raise ValueError(
                    f"n_leading_observations_test_adjustment must be greater than or equal to n_leading_observations, but got {self.n_leading_observations_test_adjustment=} and {self.n_leading_observations=}")

        # Create an inventory of all images in the dataset, and how many data points each fire contains. Since we have multiple data points per fire,
        # we need to know how many data points each fire contains, to be able to map a dataset index to a specific fire.
        self.imgs_per_fire = self.read_list_of_images()
        self.datapoints_per_fire = self.compute_datapoints_per_fire()
        self.length = sum([sum(self.datapoints_per_fire[fire_year].values()) for fire_year in self.datapoints_per_fire])

        # Used in preprocessing and normalization. Better to define it once than build/call for every data point
        # The one-hot matrix is used for one-hot encoding of land cover classes
        self.one_hot_matrix = torch.eye(17)
        self.means, self.stds, _ = get_means_stds_missing_values(self.stats_years)
        self.means = self.means[None, :, None, None]
        self.stds = self.stds[None, :, None, None]
        self.indices_of_degree_features = get_indices_of_degree_features()

    def find_image_index_from_dataset_index(self, target_id) -> (int, str, int):
        """_summary_ Given the index of a data point in the dataset, find the corresponding fire that contains it,
        and its index within that fire.

        Args:
            target_id (_type_): _description_ Dataset index of the data point.

        Raises:
            RuntimeError: _description_ Raised if the dataset index is out of range.

        Returns:
            (int, str, int): _description_ Year, name of fire, index of data point within fire.
        """

        # Handle negative indexing, e.g. -1 should be the last item in the dataset
        if target_id < 0:
            target_id = self.length + target_id
        if target_id >= self.length:
            raise RuntimeError(
                f"Tried to access item {target_id}, but maximum index is {self.length - 1}.")

        # The index is relative to the length of the full dataset. However, we need to make sure that we know which
        # specific fire the queried index belongs to. We know how many data points each fire contains from
        # self.datapoints_per_fire.
        first_id_in_current_fire = 0
        found_fire_year = None
        found_fire_name = None
        for fire_year in self.datapoints_per_fire:
            if found_fire_year is None:
                for fire_name, datapoints_in_fire in self.datapoints_per_fire[fire_year].items():
                    if target_id - first_id_in_current_fire < datapoints_in_fire:
                        found_fire_year = fire_year
                        found_fire_name = fire_name
                        break
                    else:
                        first_id_in_current_fire += datapoints_in_fire

        in_fire_index = target_id - first_id_in_current_fire

        return found_fire_year, found_fire_name, in_fire_index

    def load_imgs(self, found_fire_year, found_fire_name, in_fire_index):
        """_summary_ Load the images corresponding to the specified data point from disk.

        Args:
            found_fire_year (_type_): _description_ Year of the fire that contains the data point.
            found_fire_name (_type_): _description_ Name of the fire that contains the data point.
            in_fire_index (_type_): _description_ Index of the data point within the fire.

        Returns:
            _type_: _description_ (x,y) or (x,y,doy) tuple, depending on whether return_doy is True or False.
            x is a tensor of shape (n_leading_observations, n_features, height, width), containing the input data.
            y is a tensor of shape (height, width) containing the binary next day's active fire mask.
            doy is a tensor of shape (n_leading_observations) containing the day of the year for each observation.
        """

        in_fire_index += self.skip_initial_samples
        end_index = (in_fire_index + self.n_leading_observations + 1)

        if self.load_from_hdf5:
            hdf5_path = self.imgs_per_fire[found_fire_year][found_fire_name][0]
            with h5py.File(hdf5_path, 'r') as f:
                imgs = f["data"][in_fire_index:end_index]
                if self.return_doy:
                    doys = f["data"].attrs["img_dates"][in_fire_index:(end_index - 1)]
                    doys = self.img_dates_to_doys(doys)
                    doys = torch.Tensor(doys)
            x, y = np.split(imgs, [-1], axis=0)
            # Last image's active fire mask is used as label, rest is input data
            y = y[0, -1, ...]
        else:
            imgs_to_load = self.imgs_per_fire[found_fire_year][found_fire_name][in_fire_index:end_index]
            imgs = []
            for img_path in imgs_to_load:
                with rasterio.open(img_path, 'r') as ds:
                    imgs.append(ds.read())
            x = np.stack(imgs[:-1], axis=0)
            y = imgs[-1][-1, ...]

        if self.return_doy:
            return x, y, doys
        return x, y

    def __getitem__(self, index):
        found_fire_year, found_fire_name, in_fire_index = self.find_image_index_from_dataset_index(index)
        loaded_imgs = self.load_imgs(found_fire_year, found_fire_name, in_fire_index)

        if self.return_doy:
            x, y, doys = loaded_imgs
        else:
            x, y = loaded_imgs

        x, y = self.preprocess_and_augment(x, y)

        # Remove duplicate static features, which can greatly reduce the number of features, since we use
        # one-hot encoded landcover types. The result would have different amounts of feature channels per
        # time step, therefore, we flatten the temporal dimension.
        if self.remove_duplicate_features and self.n_leading_observations > 1:
            x = self.flatten_and_remove_duplicate_features_(x)

        # Discard features that we don't want to use
        elif self.features_to_keep is not None:
            if len(x.shape) != 4:
                raise NotImplementedError(f"Removing features is only implemented for 4D tensors, but got {x.shape=}.")
            x = x[:, self.features_to_keep, ...]

        if self.return_doy:
            # return x, y, doys
            raise NotImplementedError("")

        pixel_values = x.reshape(-1, x.size(-2), x.size(-1))
        labels = y

        # Unified return format: (image_CHW, label_HW, sample_id)
        sample_id = f"{found_fire_year}_{found_fire_name}_{in_fire_index}"
        return pixel_values.numpy().astype(np.float32), labels.numpy().astype(np.int64), sample_id

    def __len__(self):
        return self.length

    def validate_inputs(self):
        if self.n_leading_observations < 1:
            raise ValueError("Need at least one day of observations.")
        if self.return_doy and not self.load_from_hdf5:
            raise NotImplementedError(
                "Returning day of year is only implemented for hdf5 files.")
        if self.n_leading_observations_test_adjustment is not None:
            if self.n_leading_observations_test_adjustment < self.n_leading_observations:
                raise ValueError(
                    "n_leading_observations_test_adjustment must be greater than or equal to n_leading_observations.")
            if self.n_leading_observations_test_adjustment < 1:
                raise ValueError(
                    "n_leading_observations_test_adjustment must be greater than or equal to 1. Value 1 is used for having a single observation as input.")

    def read_list_of_images(self):
        """_summary_ Create an inventory of all images in the dataset.

        Returns:
            _type_: _description_ Returns a dictionary mapping integer years to dictionaries.
            These dictionaries map names of fires that happened within the respective year to either
            a) the corresponding list of image files (in case hdf5 files are not used) or
            b) the individual hdf5 file for each fire.
        """
        imgs_per_fire = {}
        for fire_year in self.included_fire_years:
            imgs_per_fire[fire_year] = {}

            if not self.load_from_hdf5:
                fires_in_year = glob.glob(f"{self.data_dir}/{fire_year}/*/")
                fires_in_year.sort()
                for fire_dir_path in fires_in_year:
                    fire_name = fire_dir_path.split("/")[-2]
                    fire_img_paths = glob.glob(f"{fire_dir_path}/*.tif")
                    fire_img_paths.sort()

                    imgs_per_fire[fire_year][fire_name] = fire_img_paths

                    if len(fire_img_paths) == 0:
                        warnings.warn(f"In dataset preparation: Fire {fire_year}: {fire_name} contains no images.",
                                      RuntimeWarning)
            else:
                fires_in_year = glob.glob(
                    f"{self.data_dir}/{fire_year}/*.hdf5")
                fires_in_year.sort()
                for fire_hdf5 in fires_in_year:
                    fire_name = Path(fire_hdf5).stem
                    imgs_per_fire[fire_year][fire_name] = [fire_hdf5]

        return imgs_per_fire

    def compute_datapoints_per_fire(self):
        """_summary_ Compute how many data points each fire contains. This is important for mapping a dataset index to a specific fire.

        Returns:
            _type_: _description_ Returns a dictionary mapping integer years to dictionaries.
            The dictionaries map the fire name to the number of data points in that fire.
        """
        datapoints_per_fire = {}
        for fire_year in self.imgs_per_fire:
            datapoints_per_fire[fire_year] = {}
            for fire_name, fire_imgs in self.imgs_per_fire[fire_year].items():
                if not self.load_from_hdf5:
                    n_fire_imgs = len(fire_imgs) - self.skip_initial_samples
                else:
                    # Catch error case that there's no file
                    if not fire_imgs:
                        n_fire_imgs = 0
                    else:
                        with h5py.File(fire_imgs[0], 'r') as f:
                            n_fire_imgs = len(f["data"]) - self.skip_initial_samples
                # If we have two days of observations, and a lead of one day,
                # we can only predict the second day's fire mask, based on the first day's observation
                datapoints_in_fire = n_fire_imgs - self.n_leading_observations
                if datapoints_in_fire <= 0:
                    warnings.warn(
                        f"In dataset preparation: Fire {fire_year}: {fire_name} does not contribute data points. It contains "
                        f"{len(fire_imgs)} images, which is too few for a lead of {self.n_leading_observations} observations.",
                        RuntimeWarning)
                    datapoints_per_fire[fire_year][fire_name] = 0
                else:
                    datapoints_per_fire[fire_year][fire_name] = datapoints_in_fire
        return datapoints_per_fire

    def standardize_features(self, x):
        """_summary_ Standardizes the input data, using the mean and standard deviation of each feature.
        Some features are excluded from this, which are the degree features (e.g. wind direction), and the land cover class.
        The binary active fire mask is also excluded, since it's added after standardization.

        Args:
            x (_type_): _description_ Input data, of shape (time_steps, features, height, width)

        Returns:
            _type_: _description_ Standardized input data, of shape (time_steps, features, height, width)
        """

        x = (x - self.means) / self.stds

        return x

    def preprocess_and_augment(self, x, y):
        """_summary_ Preprocesses and augments the input data.
        This includes:
        1. Slight preprocessing of active fire features, if loading from TIF files.
        2. Geometric data augmentation.
        3. Applying sin to degree features, to ensure that the extreme degree values are close in feature space.
        4. Standardization of features.
        5. Addition of the binary active fire mask, as an addition to the fire mask that indicates the time of detection.
        6. One-hot encoding of land cover classes.

        Args:
            x (_type_): _description_ Input data, of shape (time_steps, features, height, width)
            y (_type_): _description_ Target data, next day's binary active fire mask, of shape (height, width)

        Returns:
            _type_: _description_
        """

        x, y = torch.Tensor(x), torch.Tensor(y)

        # Preprocessing that has been done in HDF files already
        if not self.load_from_hdf5:
            # Active fire masks have nans where no detections occur. In general, we want to replace NaNs with
            # the mean of the respective feature. Since the NaNs here don't represent missing values, we replace
            # them with 0 instead.
            x[:, -1, ...] = torch.nan_to_num(x[:, -1, ...], nan=0)
            y = torch.nan_to_num(y, nan=0.0)

            # Turn active fire detection time from hhmm to hh.
            x[:, -1, ...] = torch.floor_divide(x[:, -1, ...], 100)

        y = (y > 0).long()

        # Augmentation has to come before normalization, because we have to correct the angle features when we change
        # the orientation of the image.
        if self.is_train:
            x, y = self.augment(x, y)
        else:
            x, y = self.center_crop_x32(x, y)

        # Some features take values in [0,360] degrees. By applying sin, we make sure that values near 0 and 360 are
        # close in feature space, since they are also close in reality.
        x[:, self.indices_of_degree_features, ...] = torch.sin(
            torch.deg2rad(x[:, self.indices_of_degree_features, ...]))

        # Compute binary mask of active fire pixels before normalization changes what 0 means.
        binary_af_mask = (x[:, -1:, ...] > 0).float()

        x = self.standardize_features(x)

        # Adds the binary fire mask as an additional channel to the input data.
        x = torch.cat([x, binary_af_mask], axis=1)

        # Replace NaN values with 0, thereby essentially setting them to the mean of the respective feature.
        x = torch.nan_to_num(x, nan=0.0)

        # Create land cover class one-hot encoding, put it where the land cover integer was
        new_shape = (x.shape[0], x.shape[2], x.shape[3],
                     self.one_hot_matrix.shape[0])
        # -1 because land cover classes start at 1
        landcover_classes_flattened = x[:, 16, ...].long().flatten() - 1
        landcover_encoding = self.one_hot_matrix[landcover_classes_flattened].reshape(
            new_shape).permute(0, 3, 1, 2)
        x = torch.concatenate(
            [x[:, :16, ...], landcover_encoding, x[:, 17:, ...]], dim=1)

        return x, y

    def augment(self, x, y):
        """_summary_ Applies geometric transformations:
          1. random square cropping, preferring images with a) fire pixels in the output and b) (with much less weight) fire pixels in the input
          2. rotate by multiples of 90°
          3. flip horizontally and vertically
        Adjustment of angles is done as in https://github.com/google-research/google-research/blob/master/simulation_research/next_day_wildfire_spread/image_utils.py

        Args:
            x (_type_): _description_ Input data, of shape (time_steps, features, height, width)
            y (_type_): _description_ Target data, next day's binary active fire mask, of shape (height, width)

        Returns:
            _type_: _description_
        """

        # Need square crop to prevent rotation from creating/destroying data at the borders, due to uneven side lengths.
        # Try several crops, prefer the ones with most fire pixels in output, followed by most fire_pixels in input
        best_n_fire_pixels = -1
        best_crop = (None, None)

        h_margin = x.shape[-2] - self.crop_side_length
        w_margin = x.shape[-1] - self.crop_side_length

        for i in range(10):
            top = np.random.randint(0, max(h_margin, 0) + 1)
            left = np.random.randint(0, max(w_margin, 0) + 1)
            x_crop = TF.crop(
                x, top, left, self.crop_side_length, self.crop_side_length)
            y_crop = TF.crop(
                y, top, left, self.crop_side_length, self.crop_side_length)

            # We really care about having fire pixels in the target. But if we don't find any there,
            # we care about fire pixels in the input, to learn to predict that no new observations will be made,
            # even though previous days had active fires.
            n_fire_pixels = x_crop[:, -1, ...].mean() + \
                            1000 * y_crop.float().mean()
            if n_fire_pixels > best_n_fire_pixels:
                best_n_fire_pixels = n_fire_pixels
                best_crop = (x_crop, y_crop)

        x, y = best_crop

        hflip = bool(np.random.random() > 0.5)
        vflip = bool(np.random.random() > 0.5)
        rotate = int(np.floor(np.random.random() * 4))
        if hflip:
            x = TF.hflip(x)
            y = TF.hflip(y)
            # Adjust angles
            x[:, self.indices_of_degree_features, ...] = 360 - \
                                                         x[:, self.indices_of_degree_features, ...]

        if vflip:
            x = TF.vflip(x)
            y = TF.vflip(y)
            # Adjust angles
            x[:, self.indices_of_degree_features, ...] = (
                                                                 180 - x[:, self.indices_of_degree_features, ...]) % 360

        if rotate != 0:
            angle = rotate * 90
            x = TF.rotate(x, angle)
            y = torch.unsqueeze(y, 0)
            y = TF.rotate(y, angle)
            y = torch.squeeze(y, 0)

            # Adjust angles
            x[:, self.indices_of_degree_features, ...] = (x[:, self.indices_of_degree_features,
                                                          ...] - 90 * rotate) % 360

        return x, y

    def center_crop_x32(self, x, y):
        """_summary_ Crops the center of the image to side lengths that are a multiple of 32,
        which the ResNet U-net architecture requires. Only used for computing the test performance.

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        T, C, H, W = x.shape
        H_new = H // 32 * 32
        W_new = W // 32 * 32

        x = TF.center_crop(x, (H_new, W_new))
        y = TF.center_crop(y, (H_new, W_new))
        return x, y

    def flatten_and_remove_duplicate_features_(self, x):
        """_summary_ For a simple U-Net, static and forecast features can be removed everywhere but in the last time step
        to reduce the number of features. Since that would result in different numbers of channels for different
        time steps, we flatten the temporal dimension.
        Also discards features that we don't want to use.

        Args:
            x (_type_): _description_ Input tensor data of shape (n_leading_observations, n_features, height, width)

        Returns:
            _type_: _description_
        """
        static_feature_ids, dynamic_feature_ids = self.get_static_and_dynamic_features_to_keep(self.features_to_keep)
        dynamic_feature_ids = torch.tensor(dynamic_feature_ids).int()

        x_dynamic_only = x[:-1, dynamic_feature_ids, :, :].flatten(start_dim=0, end_dim=1)
        x_last_day = x[-1, self.features_to_keep, ...].squeeze(0)

        return torch.cat([x_dynamic_only, x_last_day], axis=0)

    @staticmethod
    def get_static_and_dynamic_feature_ids():
        """_summary_ Returns the indices of static and dynamic features.
        Static features include topographical features and one-hot encoded land cover classes.

        Returns:
            _type_: _description_ Tuple of lists of integers, first list contains static feature indices, second list contains dynamic feature indices.
        """
        static_feature_ids = [12, 13, 14] + list(range(16, 33))
        dynamic_feature_ids = list(range(12)) + [15] + list(range(33, 40))
        return static_feature_ids, dynamic_feature_ids

    @staticmethod
    def get_static_and_dynamic_features_to_keep(features_to_keep: Optional[List[int]]):
        """_summary_ Returns the indices of static and dynamic features that should be kept, based on the input list of feature indices to keep.

        Args:
            features_to_keep (Optional[List[int]]): _description_

        Returns:
            _type_: _description_
        """
        static_features_to_keep, dynamic_features_to_keep = FireSpreadDataset.get_static_and_dynamic_feature_ids()

        if type(features_to_keep) == list:
            dynamic_features_to_keep = list(set(dynamic_features_to_keep) & set(features_to_keep))
            dynamic_features_to_keep.sort()

        if type(features_to_keep) == list:
            static_features_to_keep = list(set(static_features_to_keep) & set(features_to_keep))
            static_features_to_keep.sort()

        return static_features_to_keep, dynamic_features_to_keep

    @staticmethod
    def get_n_features(n_observations: int, features_to_keep: Optional[List[int]], deduplicate_static_features: bool):
        """_summary_ Computes the number of features that the dataset will have after preprocessing,
        considering the number of input observations, which features to keep or discard, and whether to deduplicate static features.

        Args:
            n_observations (int): _description_
            features_to_keep (Optional[List[int]]): _description_
            deduplicate_static_features (bool): _description_

        Returns:
            _type_: _description_ If deduplicate_static_features is True, returns the total number of features, flattened across all time steps.
            Otherwise, returns the number of features per time step.
        """
        static_features_to_keep, dynamic_features_to_keep = FireSpreadDataset.get_static_and_dynamic_features_to_keep(
            features_to_keep)

        n_static_features = len(static_features_to_keep)
        n_dynamic_features = len(dynamic_features_to_keep)
        n_all_features = n_static_features + n_dynamic_features

        # If we deduplicate static features, we remove them from all time steps but the last one.
        # The last day then gets dynamic and static features. All other days only get dynamic features.
        n_features = (int(deduplicate_static_features) * n_dynamic_features) * (n_observations - 1) + n_all_features

        return n_features

    @staticmethod
    def img_dates_to_doys(img_dates):
        """_summary_ Converts a list of date strings to day of year values.

        Args:
            img_dates (_type_): _description_ List of date strings

        Returns:
            _type_: _description_ List of day of year values
        """
        date_format = "%Y-%m-%d"
        # In old preprocessing, the dates still had a TIF file extension, which is also removed here.
        return [datetime.strptime(img_date.replace(".tif", ""), date_format).timetuple().tm_yday for img_date in
                img_dates]

    @staticmethod
    def map_channel_index_to_features(only_base: bool = False):
        """_summary_ Maps the channel index to the feature name.

        Returns:
            _type_: _description_
        """

        # Features before any processing
        base_feature_names = [
            'VIIRS band M11',
            'VIIRS band I2',
            'VIIRS band I1',
            'NDVI',
            'EVI2',
            'Total precipitation',
            'Wind speed',
            'Wind direction',
            'Minimum temperature',
            'Maximum temperature',
            'Energy release component',
            'Specific humidity',
            'Slope',
            'Aspect',
            'Elevation',
            'Palmer drought severity index (PDSI)',
            'Landcover class',
            'Forecast: Total precipitation',
            'Forecast: Wind speed',
            'Forecast: Wind direction',
            'Forecast: Temperature',
            'Forecast: Specific humidity',
            'Active fire']

        # Different land cover classes of feature "Landcover class"
        land_cover_classes = [
            'Land cover: Evergreen Needleleaf Forests',
            'Land cover: Evergreen Broadleaf Forests',
            'Land cover: Deciduous Needleleaf Forests',
            'Land cover: Deciduous Broadleaf Forests',
            'Land cover: Mixed Forests',
            'Land cover: Closed Shrublands',
            'Land cover: Open Shrublands',
            'Land cover: Woody Savannas',
            'Land cover: Savannas',
            'Land cover: Grasslands',
            'Land cover: Permanent Wetlands',
            'Land cover: Croplands',
            'Land cover: Urban and Built-up Lands',
            'Land cover: Cropland/Natural Vegetation Mosaics',
            'Land cover: Permanent Snow and Ice',
            'Land cover: Barren',
            'Land cover: Water Bodies']

        if only_base:
            # Features as in the GeoTIFF files: land cover class not expanded, no binary active fire
            return_features = base_feature_names
        else:
            # Features as used by most experiments
            return_features = base_feature_names[:16] + land_cover_classes + base_feature_names[17:] + [
                "Active fire (binary)"]

        return dict(enumerate(return_features))

    def get_generator_for_hdf5(self):
        """_summary_ Creates a generator that is used to turn the dataset into HDF5 files. It applies a few
        preprocessing steps to the active fire features that need to be applied anyway, to save some computation.

        Yields:
            _type_: _description_ Generator that yields tuples of (year, fire_name, img_dates, lnglat, img_array)
            where img_array contains all images available for the respective fire, preprocessed such
            that active fire detection times are converted to hours. lnglat contains longitude and latitude
            of the center of the image.
        """

        for year, fires_in_year in self.imgs_per_fire.items():
            for fire_name, img_files in fires_in_year.items():
                imgs = []
                lnglat = None
                for img_path in img_files:
                    with rasterio.open(img_path, 'r') as ds:
                        imgs.append(ds.read())
                        if lnglat is None:
                            lnglat = ds.lnglat()
                x = np.stack(imgs, axis=0)

                # Get dates from filenames
                img_dates = [img_path.split("/")[-1].split("_")[0].replace(".tif", "")
                             for img_path in img_files]

                # Active fire masks have nans where no detections occur. In general, we want to replace NaNs with
                # the mean of the respective feature. Since the NaNs here don't represent missing values, we replace
                # them with 0 instead.
                x[:, -1, ...] = np.nan_to_num(x[:, -1, ...], nan=0)

                # Turn active fire detection time from hhmm to hh.
                x[:, -1, ...] = np.floor_divide(x[:, -1, ...], 100)
                yield year, fire_name, img_dates, lnglat, x
