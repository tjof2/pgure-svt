from hyperspy import signals

try:
    from pguresvt.pguresvt import SVT
except ImportError:
    raise ImportError("It looks like you may be working in the original "
                      "PGURE-SVT directory. Try again in a different "
                      "directory.")


class HSPYSVT(SVT):

    """HyperSpy implementation of SVT.

    Allows arbitrary HyperSpy signals to be passed for SVT denoising.

    """

    def __init__(self, *args, **kwargs):
        super(HSPYSVT, self).__init__(*args, **kwargs)
        self.signal_type = None

    def prepare_to_denoise(self, signal):
        """
        Represents `signal` as a spectrum (three dimensions; relevant axis last)
        then returns the data, properly aligned.

        Parameters
        ----------
        signal : hyperspy.signals.Signal
            The HyperSpy signal to denoise; can be of arbitrary type/shape.

        Returns
        -------
        signal_data : numpy.ndarray

        """
        if signal.axes_manager.signal_dimension == 1:
            signal_data = signal._data_aligned_with_axes
            self.signal_type = "spectrum"
        elif signal.axes_manager.signal_dimension == 2:
            signal.unfold_navigation_space()
            signal_3d = signal.as_signal1D(spectral_axis=0)
            signal_data = signal_3d._data_aligned_with_axes
            signal.fold()
            self.signal_type = "image"
        else:
            raise NotImplementedError("`signal` should be of `Image` or "
                                      "`Spectrum` type.")
        return signal_data

    def denoised_data_to_signal(self):
        signal = signals.BaseSignal(self.Y)
        if self.signal_type == "spectrum":
            return signal.as_signal1D(2)
        if self.signal_type == "image":
            return signal.as_signal2D((1,2))

    def denoise(self, signal):
        """
        Denoises an arbitrary HyperSpy signal.

        Parameters
        ----------
        signal : hyperspy.signals.Signal
            The HyperSpy signal to denoise; can be of arbitrary type/shape.

        Returns
        -------
        hyperspy.signals.Spectrum
            Denoised data as a spectrum.

        """
        x = self.prepare_to_denoise(signal)
        super(HSPYSVT, self).denoise(x)
        return self.denoised_data_to_signal()


if __name__ == '__main__':
    import hyperspy.api as hs
    import matplotlib.pyplot as plt

    # Load example dataset
    movie = hs.load("../test/examplesequence.tif")

    # Truncate to 15 frames
    movie = movie.inav[:15]
    movie.plot()
    plt.show()

    # Denoise
    svt = HSPYSVT(threshold=0.5)
    denoised_movie = svt.denoise(movie)

    # Plot data
    denoised_movie.plot()
    plt.show()
