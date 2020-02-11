import os
from typing import List, Tuple, Dict

import numpy as np

import utils_io


S3_URL = 'https://bea-portfolio.s3-us-west-2.amazonaws.com/music-classification/'
SONGS_FOLDER = 'songs'
DATA_FOLDER = 'data'
DATA_SPECTROGRAMS = 'spectrograms.npz'
DATA_FREQUENCIES = 'frequency_list.npz'
DATA_TIME = 'time_list.npz'


def _read_wav_files(folder: str, wav_files: Dict[str, np.ndarray]) -> float:
    """Reads music wav files recursively from disk. Populates a dictionary
    whose keys are the path to each file, and whose values are the 
    corresponding ndarray of audio samples.
    """
    if os.path.exists(folder):
        items = os.listdir(folder)
        for item in items:
            item_path = os.path.join(folder, item)
            if os.path.isdir(item_path):
                sample_rate = _read_wav_files(item_path, wav_files)
            elif os.path.isfile(item_path):
                (sample_rate, wav_files[item_path]) = utils_io.load_wav_file(item_path)

    return sample_rate


def _get_gaussian_filter(b: float, b_list: np.ndarray, 
    sigma: float) -> np.ndarray:
    """
    Returns the values of a Gaussian filter centered at b, with standard 
    deviation sigma.
    """
    a = 1/(2*sigma**2)
    return np.exp(-a*(b_list-b)**2)


def _get_spectrogram_coordinates(sample_rate: float, data: np.ndarray, num_samples: 
    int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gets the time and frequency lists used to construct a spectrogram.
    """
    n = len(data)
    max_time = n/sample_rate

    # Get time steps.
    all_time_list = np.linspace(0, max_time, n)

    # Get frequencies.
    # Angular frequency.
    # omega_list = (2 * np.pi)/max_time * np.linspace(-n/2, n/2, n+1)[0:-1]
    # Frequency in Hz.
    frequency_list = np.linspace(-n/2, n/2, n+1)[0:-1] / max_time

    # Get sampled time steps.
    sampled_time_list = np.linspace(0, max_time, num_samples)

    return (frequency_list, sampled_time_list, all_time_list)


def _normalize(my_array: np.ndarray) -> np.ndarray:
    """Normalizes an ndarray.
    """
    return np.abs(my_array)/np.max(np.abs(my_array))


def _generate_spectrogram(data: np.ndarray, sample_rate: 
    float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns spectrogram for data passed as a parameter.
    """
    # We'll construct the spectrogram using a Gaussian filter.
    sigma = 0.1
    # We'll slide the Gaussian filter num_samples times.
    num_samples = 200
    # The spectrogram is of shape (frequency_list, sampled_time_list), 
    # where sampled_time_list is a sampling of the full list of times, 
    # all_time_list.
    (frequency_list, sampled_time_list, 
        all_time_list) = _get_spectrogram_coordinates(
        sample_rate, data, num_samples)
    spectrogram = np.empty((len(frequency_list), len(sampled_time_list)))

    # For each time step, slide the Gabor filter so that it's centered at 
    # the desired time, apply it to the function in time domain, and 
    # transform the result using FFT.
    for (i, b) in enumerate(sampled_time_list):
        g = _get_gaussian_filter(b, all_time_list, sigma)
        ug = data * g
        ugt = np.fft.fftshift(np.fft.fft(ug))
        spectrogram[:, i] = _normalize(ugt)
       
    return (spectrogram, frequency_list, sampled_time_list)


def _generate_spectrograms() -> None:
    """Reads music wav files, produces spectrograms and saves them.
    """
    # Read wav files.
    print('Reading wav files...')
    wav_files = {}
    sample_rate = _read_wav_files(SONGS_FOLDER, wav_files)
    # Generate spectrograms.
    print('Generating spectrograms...')
    spectrograms = {}
    for wav_file_key in wav_files:
        print(f'Generating spectrogram for {wav_file_key}...')
        (spectrogram, frequency_list, 
            sampled_time_list) = _generate_spectrogram(
            wav_files[wav_file_key], sample_rate)
        spectrograms[wav_file_key] = spectrogram
    # Save spectrograms.
    print('Saving spectrograms...')
    data_path = utils_io.find_or_create_dir(DATA_FOLDER)
    spectrograms_path = os.path.join(DATA_FOLDER, DATA_SPECTROGRAMS)
    frequencies_path = os.path.join(DATA_FOLDER, DATA_FREQUENCIES)
    time_path = os.path.join(DATA_FOLDER, DATA_TIME)
    np.savez(spectrograms_path, *spectrograms)
    np.savez(frequencies_path, frequency_list)
    np.savez(time_path, sampled_time_list)


def _download_spectrograms() -> None:
    """Downloads spectrograms from remote location if they're not yet 
    present locally.
    """
    remote_url_spectrograms = S3_URL + DATA_SPECTROGRAMS
    remote_url_frequencies = S3_URL + DATA_FREQUENCIES
    remote_url_time = S3_URL + DATA_TIME
    local_path_spectrograms = os.path.join(DATA_FOLDER, DATA_SPECTROGRAMS)
    local_path_frequencies = os.path.join(DATA_FOLDER, DATA_FREQUENCIES)
    local_path_time = os.path.join(DATA_FOLDER, DATA_TIME)

    data_path = utils_io.find_or_create_dir(DATA_FOLDER)

    if not os.path.exists(local_path_spectrograms):
        utils_io.download_remote_data_file(DATA_FOLDER, remote_url_spectrograms)
    if not os.path.exists(local_path_frequencies):
        utils_io.download_remote_data_file(DATA_FOLDER, remote_url_frequencies)
    if not os.path.exists(local_path_time):
        utils_io.download_remote_data_file(DATA_FOLDER, remote_url_time)


def get_data(method: str) -> None:
    """Generates spectrograms from wav files or downloads them from a remote
    location.
    """
    if method == 'generate':
        _generate_spectrograms()
    elif method == 'download':
        _download_spectrograms()


def main() -> None:
    """Main program.
    """
    get_data(method='download')
    pass


if __name__ == '__main__':
    main()
