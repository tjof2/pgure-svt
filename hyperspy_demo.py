
def prepare_to_denoise(signal, spectral_axis=0):
    """
    Represents `signal` as a spectrum (three dimensions; relevant axis last)
    then returns the data, properly aligned.

    Parameters
    ----------
    signal : hyperspy.signals.Signal
        The HyperSpy signal to prepare_to_denoise; can be of arbitrary
        type/shape.
    spectral_axis : Optional [int]
        The axis along which denoising will occur; for example in a time
        series of images with shape (285|256, 256) the spectral axis is 0.

    Returns
    -------
    signal_data : numpy.ndarray


    """
    signal_3d = signal.as_spectrum(spectral_axis=spectral_axis)
    signal_data = signal_3d._data_aligned_with_axes
    return signal_data


if __name__ == '__main__':
    from hyperspy.api import load
    from pguresvt import SVT

    # Load example dataset
    movie = load("FEI_HAADF_Image_movie_282.dm4")

    # Rearrange into appropriate dimensions
    data = prepare_to_denoise(movie, spectral_axis=0)

    # Check new dimensions
    print(data.shape)
