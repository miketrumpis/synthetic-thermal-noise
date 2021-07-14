from typing import Union
import numpy as np
from scipy.fft import fft, ifft, fftshift
from scipy.signal.filter_design import butter
from scipy.signal import freqz, cheb1ord, cheby1

from .eis_models import thermal_noise


def _check_bandpass(bandpass):
    if not bandpass:
        return list()
    # make sure bandpass is a sequence [bp1, ...]
    # ... this would potentially confound with an order 2 transfer function :(
    if not np.iterable(bandpass[0]) or len(bandpass[0]) != 2:
        bandpass = [bandpass]
    return list(bandpass)


def simulate_thermal_noise(n: int, bw: float, n_chan: int=1, bandpass: Union[tuple, list]=(),
                           debug: bool=False, sim_filtfilt=True, **model_params):
    # Kolmogorov spectral factorization
    # code based on http://web.cvxr.com/cvx/examples/filter_design/html/spectral_fact.html
    # use this for the final simulation FFT
    nfft_seq = 2 ** int(np.ceil(np.log2(n)))
    # oversample FFT by about 32X
    oversamp = 32
    nfft = nfft_seq * oversamp
    # nfft = 2 ** 16
    # bw = 2e4
    freq = np.arange(nfft) * bw / nfft
    freq -= bw / 2
    # put frequency into the order wanted by scipy fft
    sp_freq = fftshift(freq)
    # This is like the spectrum sampled over the unit circle [0, pi] + (-pi, 0)
    spec_f = thermal_noise(sp_freq, **model_params)
    alpha = np.log(spec_f) / 2
    # This is now in weird fft order
    alpha_ft = fft(alpha)
    # Hilbert transform manually -- negate negative spectrum
    alpha_ft[nfft // 2 + 1:] = -alpha_ft[nfft // 2 + 1:]
    # Zero out DC and Nyquist
    alpha_ft[0] = 0
    alpha_ft[nfft // 2] = 0
    phi = ifft(1j * alpha_ft).real

    # coef_a = spfft.ifft(np.log(spec_f) / 2).real
    spec_mp = np.exp(alpha + 1j * phi)

    bandpass = _check_bandpass(bandpass)
    for filt in bandpass:
        # do a butterworth filter for the lowpass and highpass
        f_lo, f_hi = filt
        # this could be a transfer function polynomial
        if np.iterable(f_lo) or np.iterable(f_hi):
            b, a = f_lo, f_hi
            h = freqz(b, a, worN=sp_freq * 2 * np.pi / bw)[1]
        else:
            if f_lo > 0:
                b, a = butter(3, 2 * f_lo / bw, btype='highpass')
                h = freqz(b, a, worN=sp_freq * 2 * np.pi / bw)[1]
            else:
                h = 1
            if f_hi > 0:
                b, a = butter(3, 2 * f_hi / bw, btype='lowpass')
                h = h * freqz(b, a, worN=sp_freq * 2 * np.pi / bw)[1]
        if sim_filtfilt:
            h = h * h.conj()
        spec_mp = spec_mp * h

    if debug:
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(2, 1, sharex=True)
        sigma = np.trapz(spec_f) * (freq[1] - freq[0])
        ax[0].semilogy(freq, fftshift(np.abs(spec_mp)), label='filter mag')
        ax[0].semilogy(freq, fftshift(np.abs(spec_mp) ** 2), label='mag-sq', ls=':', zorder=2)
        ax[0].semilogy(freq, fftshift(spec_f), label='noise spec (var={:.3f})'.format(sigma), ls='--', zorder=1)
        ax[1].plot(freq, fftshift(np.angle(spec_mp)))
        ax[0].legend()
        ax[1].set_xlabel('Frequency')
        ax[1].set_ylabel('Phase')
        ax[0].set_ylabel('Magnitude/Spectral density')
        f.tight_layout()

    nz_spec = np.zeros((n_chan, nfft_seq), dtype='D')
    # Make the noise density such that the IFFT would give unit variance..
    sigma = (nfft_seq / 2) ** 0.5
    rand_size = (n_chan, nfft_seq // 2)
    nz_spec[:, 1:nfft_seq // 2 + 1] =  (np.random.randn(*rand_size) + 1j * np.random.randn(*rand_size)) * sigma
    nz_spec[:, nfft_seq // 2 + 1:] = nz_spec[:, 1:nfft_seq // 2][:, ::-1].conj()

    # Units adjustment is needed to make the 1 / N in the FFT sum look like df=BW / N in a power spectrum integral.
    # This should ensure that var(nz_seq) ~ integral(noise PSD)
    nz_seq = ifft(nz_spec * spec_mp[::oversamp] * np.sqrt(bw), axis=1).real
    return nz_seq[:, :n].squeeze()


def long_noise_series(n, bw, n_chan, bandpass, min_sub=2 ** 16,
                      resample_bw=None, actually_resample=True,
                      **model_params):
    lo_corner = bandpass[0]
    # suppose that 10 times the period of the highpass corner frequency should be uncorrelated
    sub_length = 2 ** int(np.ceil(np.log2(10 * bw / lo_corner)))
    sub_length = max(min_sub, sub_length)

    if resample_bw:
        # use actually resample to simulate AA filtering
        drop_rate = int(bw / resample_bw) if actually_resample else 1
        # Design an anti-aliasing filter -- same logic as ecogdata downsample
        wp = 2 * 0.4 * resample_bw / bw
        ws = 2 * 0.5 * resample_bw / bw
        ord, wc = cheb1ord(wp, ws, 0.25, 20)
        # fdesign = dict(ripple=0.25, hi=0.5 * wc * fs, Fs=fs, ord=ord)
        b, a = cheby1(ord, 0.25, [wc], btype='lowpass')
        bandpass = _check_bandpass(bandpass)
        bandpass.insert(0, (b, a))
    else:
        drop_rate = 1
    num_segments = int(n // (sub_length / drop_rate))
    num_segments += int(num_segments * (sub_length / drop_rate) < n)
    series = np.empty((n_chan, n))
    t = 0
    # print('sub_length:', sub_length)
    for i in range(num_segments):
        sub_series = simulate_thermal_noise(sub_length, bw, n_chan=n_chan, bandpass=bandpass,
                                            debug=False, **model_params)
        sub_series = sub_series[:, ::drop_rate]
        sub_t = min(n - t, sub_series.shape[1])
        # print('{:03d}: {}-{} of {}'.format(i, t, t + sub_t, n))
        series[:, t:t + sub_t] = sub_series[:, :sub_t]
        t = t + sub_t
    return series

