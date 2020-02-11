import os
import shutil
from typing import List, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen
import scipy.io.wavfile as wavfile

import numpy as np


def find_or_create_dir(dir_name: str) -> str:
    """Creates a directory if it doesn't exist.

    Args:
        dir_name (str): The name of the directory to create if it doesn't
        exist.

    Returns:
        The local path of the directory.
    """
    # Get the directory of the current file.
    parent_dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # Create a directory if it doesn't exist.
    dir_path = os.path.join(parent_dir_path, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def download_remote_data_file(local_data_folder: str, 
    data_url: str) -> Tuple[str, bool]:
    """Downloads data from url if it's not saved locally yet.

    Args:
        data_url (str): The url of the data file we want to download.
    
    Returns:
        The path to the local file, and a bool indicating whether the file
        was downloaded or not.
    """
    # Create a data directory if it doesn't exist.
    data_dir_path = find_or_create_dir(local_data_folder)
    
    # Download the data file if it doesn't exist.
    filename = os.path.basename(urlparse(data_url).path)
    data_file_path = os.path.join(data_dir_path, filename)
    downloaded = False
    if not os.path.exists(data_file_path):
        print(f'Downloading data file {data_url}...')
        with urlopen(data_url) as response:
            with open(data_file_path, "wb") as data_file:
                shutil.copyfileobj(response, data_file)
        downloaded = True
        print('Done downloading data file.')

    return (data_file_path, downloaded)


def load_wav_file(local_path: str) -> np.ndarray:
    """Loads a wav file.
    """
    # sample_rate is measurements per second.
    (sample_rate, wav_data) = wavfile.read(local_path)

    # 65536 = 2^16. 
    # 32768 = 2^15.
    # Samples in a wav file have 16 bits, with 1 bit for the sign.
    # We scale the values to be between -1 and 1.
    wav_data = wav_data/32768

    # Throw away one of the stereo channels.
    wav_data = wav_data[:,0]

    return (sample_rate, wav_data)
