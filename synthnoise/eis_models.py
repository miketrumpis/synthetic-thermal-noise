"""
Module to support fitting and manipulating EIS models

"""

import os
import numpy as np
from scipy.fft import ifft, fftshift
from scipy.optimize import minimize


__all__ = ['fit_eis', 'show_fit_eis', 'z_model', 'eis_function', 'thermal_noise', 'thermal_noise_dft']


# Global level cache for models
_model_lut = dict()

# Some default electrode model parameters:
# CPE terms: "cap" strength 26.3991 x 10^6, phase twist: 0.8746
# Faradaic resistance: 13.7172 M-Ohm
# Series resistance: 5.8 k-Ohm


def _get_model(**model_params):
    defaults = {'alpha': 0.8746120571208984,
                'Ck': 3.787473522094292e-08,
                'Rct': 13721951.030366043,
                'Rs': 5799.764041746495}
    params = dict()
    for k in model_params:
        if model_params[k] is not None:
            params[k] = model_params[k]
    for k in defaults:
        params.setdefault(k, defaults[k])
    return params


def fit_eis(eis_freqs: np.ndarray, eis_z: np.ndarray=None, eis_m: np.ndarray=None, eis_p: np.ndarray=None):
    if eis_m is not None:
        Z_obs = eis_m * np.exp(1j * np.pi * eis_p / 180)
    else:
        Z_obs = eis_z
        eis_m = np.abs(eis_z)
        eis_p = np.angle(eis_z) / np.pi * 180

    # Set up initial points
    # Find alpha from the ratios of impedance values separated in
    # frequency by factors of 10
    ratios = [eis_m[x] / eis_m[x + 3] for x in range(len(eis_freqs) - 3)]
    alpha = np.log10(ratios).mean()
    # Now multiply the magnitude spectra by omega ** alpha to "flatten"
    # and find the mean magnitude
    # flattened = eis_m * np.power(2 * np.pi * eis_f, alpha)
    # do log average or straight average (geometric < arithmetic estimator)
    # Ck = 10 ** np.log10((flattened))).mean()
    om = np.abs(eis_freqs - np.pi).argmin()
    # invert Ck for bounds & gradient purposes
    Ck = np.abs(Z_obs[om])

    def cost_fn(p, freq, Z):
        alpha, Ck, Rct, Rs = p
        Ck = np.exp(-Ck)
        Rct = np.exp(Rct)
        Rs = np.exp(Rs)
        # Zm = model_fn(p, om)
        Zr, Zi = z_model(freq, alpha=alpha, Ck=Ck, Rct=Rct, Rs=Rs, realonly=False)
        Zm = Zr + 1j * Zi
        err = np.linalg.norm(np.log(Zm) - np.log(Z))
        return err ** 2

    # optimization
    p0 = (alpha, np.log(Ck), np.log(Z_obs.real.max()), np.log(Z_obs.real.min()))
    bounds = [ (0, 1), (0, None), (0, None), (0, None) ]
    r_opt = minimize(cost_fn, p0, args=(eis_freqs, Z_obs), tol=1e-6, bounds=bounds)
    model = dict(alpha=r_opt.x[0], Ck=np.exp(-r_opt.x[1]), Rct=np.exp(r_opt.x[2]), Rs=np.exp(r_opt.x[3]))
    eis_data = dict(frequency=eis_freqs, magnitude=eis_m, phase=eis_p)
    return model, eis_data


def show_fit_eis(eis_data=None, model=None):
    if eis_data is None:
        dat_path = os.path.join(os.path.split(__file__)[0], 'eis_data.npz')
        z = np.load(dat_path)
        eis_f = z['eis_f']
        eis_m = z['eis_m']
        eis_p = z['eis_p']
        Z_obs = eis_m * np.exp(1j * np.pi * eis_p / 180)
        Z_obs = np.exp(np.log(Z_obs).mean(0))
        model, eis_data = fit_eis(eis_f, eis_z=Z_obs)
    else:
        eis_f = eis_data['frequency']
        eis_m = None
        eis_p = None
        # eis_m = eis_data['magnitude']
        # eis_p = eis_data['phase']
        Z_obs = eis_data['magnitude'] * np.exp(1j * np.pi * eis_data['phase'] / 180)
    import matplotlib.pyplot as plt
    fm = np.logspace(-0.2, 4, 100)
    Zr, Zi = z_model(fm, realonly=False, **model)
    Zm = Zr + 1j * Zi
    f, ax = plt.subplots(2, 1, sharex=True)
    if eis_m is not None:
        ax[0].loglog(eis_f, eis_m.T, lw=.5, alpha=0.5, color='k')
        ax[0].loglog(eis_f, np.abs(Z_obs), color='r', label='Mean of electrodes')
    else:
        ax[0].loglog(eis_f, np.abs(Z_obs), color='r', label='Measured EIS')
    ax[0].loglog(fm, np.abs(Zm), color='c', label='model')
    ax[0].set_ylabel('Magnitude (Ohms)')
    ax[0].legend()
    if eis_p is not None:
        ax[1].semilogx(eis_f, eis_p.T, lw=.5, alpha=0.5, color='k')
    ax[1].semilogx(eis_f, np.angle(Z_obs) * 180 / np.pi, color='r')
    ax[1].semilogx(fm, np.angle(Zm) * 180 / np.pi, color='c')
    ax[1].set_ylabel('Phase (deg.)')
    ax[1].set_xlabel('Frequency (Hz)')
    f.tight_layout()
    return model, eis_data, f


def z_model(f, alpha=None, Ck=None, Rct=None, Rs=None, realonly=True):
    """
    A realistic electrode impedance model (fit for a LCP + gold electrode).

    Parameters
    ----------
    f : float or array
        Frequency vector in Hertz.
    alpha : float
        CPE exponent (0 < alpha <= 1)
    Ck : float
        CPE pseudo-capacitance
    Rct : float
        Shunting faradaic resistance for very low frequency (prevents infinite DC impedance)
    Rs : float
        Solution resistance
    realonly : bool
        If True, return real part (i.e. for thermal noise)

    Returns
    -------
    z_real : float or array
        Real part of the impedance spectrum
    z_imag : float or array
        Imaginary part of the impedance spectrum

    """
    model = _get_model(alpha=alpha, Ck=Ck, Rct=Rct, Rs=Rs)
    alpha, Ck, Rct, Rs = [model[k] for k in ('alpha', 'Ck', 'Rct', 'Rs')]
    om = 2 * np.pi * np.abs(f)
    re_part = 1 + Rct * Ck * np.power(om, alpha) * np.cos(alpha * np.pi / 2)
    im_part = Rct * Ck * np.power(om, alpha) * np.sin(alpha * np.pi / 2)
    denom = re_part ** 2 + im_part ** 2
    z_real = Rct * re_part / denom + Rs
    if not realonly:
        z_imag = -Rct * im_part / denom
    return z_real if realonly else (z_real, z_imag)


class _z_model:
    """
    This curve can be uniformly scaled to a target impedance magnitude at one frequency.
    For example, the entire spectrum can be multiplied to return 500 k-Ohm at 1 kHz.
    """

    def __init__(self, scaleto=None, **kwargs):
        if scaleto is not None:
            if not np.iterable(scaleto):
                f_test = 1000
                z_match = scaleto
            else:
                f_test, z_match = scaleto
            kw = kwargs.copy()
            kw['realonly'] = False
            zr_test, zi_test = z_model(f_test, **kw)
            z_test = np.sqrt(zr_test ** 2 + zi_test ** 2)
            self.scale = z_match / z_test
        else:
            self.scale = 1
        self.kwargs = kwargs

    def __call__(self, f):
        z = z_model(f, **self.kwargs)
        if isinstance(z, tuple):
            return (z[0] * self.scale, z[1] * self.scale)
        return z * self.scale


def eis_function(f, alpha=None, Ck=None, Rct=None, Rs=None, scaleto=None, realonly=True, evaluate=True):
    """
    Retrieve a callable electrode impedance function of frequency and return the value.

    Parameters
    ----------
    f : float or array
        Frequency vector in Hertz.
    alpha : float
        CPE exponent (0 < alpha <= 1)
    Ck : float
        CPE pseudo-capacitance
    Rct : float
        Shunting faradaic resistance for very low frequency (prevents infinite DC impedance)
    Rs : float
        Solution resistance
    scaleto : float, tuple
        If given as (f, Z), scale the spectrum to cross |Z|(f) = Z. If only Z is given, assume f = 1 kHz.
    realonly : bool
        If True, return real part (i.e. for thermal noise)
    evaluate: bool
        By default return the EIS value, otherwise return the model.

    Returns
    -------
    z_real : float or array
        Real part of the impedance spectrum
    z_imag : float or array
        Imaginary part of the impedance spectrum

    """
    model = _get_model(alpha=alpha, Ck=Ck, Rct=Rct, Rs=Rs)
    alpha, Ck, Rct, Rs = [model[k] for k in ('alpha', 'Ck', 'Rct', 'Rs')]
    key = (alpha, Ck, Rct, Rs, scaleto, realonly)
    if key not in _model_lut:
        model = _z_model(alpha=alpha, Ck=Ck, Rct=Rct, Rs=Rs, scaleto=scaleto, realonly=realonly)
        _model_lut[key] = model
    return _model_lut[key](f) if evaluate else _model_lut[key]


def thermal_noise(f, onesided=True, eis_fun=None, **kwargs):
    """
    Compute the thermal noise in micro-Volts squared per Hz.

    Parameters
    ----------
    f : float or array
        Frequency (or ies)
    onesided : bool
        For one sided integration, double the power
    kwargs : dict
        Model parameters for z_model()

    Returns
    -------
    n : float or array
        Noise power density

    """
    T = 37 + 273.15
    K = 1.38e-5
    t_const = 2 * K * T
    if onesided:
        t_const *= 2
    # go back to micro-volts squared
    t_const /= 1e6
    if eis_fun is None:
        kwargs['realonly'] = True
        eis_fun = eis_function(f, evaluate=False, **kwargs)
    return t_const * eis_fun(f)


def thermal_noise_dft(bw: float, nfft: int, full: bool=False, **model_params):
    """

    Parameters
    ----------
    bw :
    nfft :
    bandpass :
    model_params :

    Returns
    -------

    """
    freq = np.arange(nfft) * bw / float(nfft)
    freq -= bw / 2
    t_spectrum = thermal_noise(fftshift(freq), **model_params)
    # IFFT normalized by 1 / N but the actual integral df should be BW / N:
    # Multiply by BW here
    t_acf = ifft(t_spectrum).real * bw
    if full:
        return t_acf
    t_acf = t_acf[:nfft // 2 + 1]
    return t_acf
