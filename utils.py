import tensorflow as tf
import os.path

def download_ngsim(fname="./ngsim_us101.csv"):
    """Downloads NGSIM US Highway 101 dataset locally. Approx 1461MB."""
    URL = "https://data.transportation.gov/api/views/8ect-6jqj/rows.csv?accessType=DOWNLOAD"
    tf.keras.utils.get_file(os.path.abspath(fname), origin=URL)
    return

