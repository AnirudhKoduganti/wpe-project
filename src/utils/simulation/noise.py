import numpy as np

def add_noise_to_snr(signal, snr_db, seed):
    """Add white Gaussian noise (per channel) to reach target SNR in dB.
    signal: shape (n_channels, n_samples). snr_db=None returns signal unchanged."""
    if snr_db is None:
        return signal
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, size=signal.shape)
    sig_power = np.mean(signal ** 2, axis=-1, keepdims=True)
    noise_power = np.mean(noise ** 2, axis=-1, keepdims=True)
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / (noise_power + 1e-12))
    return signal + noise * scale