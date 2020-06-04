# Copyright 2015-2020 Tom Furnival
#
# This file is part of PGURE-SVT.
#
# PGURE-SVT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PGURE-SVT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PGURE-SVT.  If not, see <http://www.gnu.org/licenses/>.

try:
    from hyperspy import signals
except ImportError:
    raise ImportError("Requires HyperSpy to be installed")


try:
    from pguresvt.pguresvt import SVT
except ImportError:
    raise ImportError(
        "It looks like you may be working in the original "
        "PGURE-SVT directory. Try again in a different "
        "directory."
    )


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
            raise NotImplementedError(
                "`signal` should be of `Image` or `Spectrum` type."
            )
        return signal_data

    def denoised_data_to_signal(self):
        signal = signals.BaseSignal(self.Y)
        if self.signal_type == "spectrum":
            return signal.as_signal1D(2)
        if self.signal_type == "image":
            return signal.as_signal2D((1, 2))

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
