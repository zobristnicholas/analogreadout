import numpy as np


def load(file_name, **kwargs):
    try:
        npz_file = np.load(file_name, **kwargs)
    except TypeError:
        kwargs.pop("allow_pickle", None)
        npz_file = np.load(file_name, **kwargs)  # fallback for old numpy that doesn't have allow_pickle
    return npz_file
