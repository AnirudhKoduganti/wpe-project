import numpy as np
import pyroomacoustics as pra
from scipy.signal import fftconvolve

def generate_room_rir(room_dim, fs, rt60_target, source_distance, mic_spacing, seed):
    np.random.seed(seed)
    e_absorption, max_order = pra.inverse_sabine(rt60_target, room_dim)
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)

    room_center = np.array(room_dim) / 2
    d = mic_spacing / 2
    mic_positions = np.array([
        [room_center[0]-d, room_center[1]-d, room_center[2]],
        [room_center[0]+d, room_center[1]-d, room_center[2]],
        [room_center[0]-d, room_center[1]+d, room_center[2]],
        [room_center[0]+d, room_center[1]+d, room_center[2]],
    ]).T
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))

    source_pos = room_center + np.array([source_distance, 0, 0])
    room.add_source(source_pos.tolist())
    room.compute_rir()

    achieved_rt60 = room.measure_rt60().flatten()

    rir_list = [room.rir[m][0] for m in range(4)]
    max_len = max(len(r) for r in rir_list)
    rir_array = np.zeros((4, max_len))
    for m, r in enumerate(rir_list):
        rir_array[m, :len(r)] = r

    metadata = {
        'seed': seed, 'room_dim_m': room_dim, 'rt60_target_s': rt60_target,
        'rt60_achieved_s': achieved_rt60.tolist(), 'source_distance_m': source_distance,
        'mic_spacing_m': mic_spacing,
    }
    return rir_array, metadata


def convolve_multichannel(clean, rir_array):
    n_mics = rir_array.shape[0]
    channels = [fftconvolve(clean, rir_array[m], mode='full') for m in range(n_mics)]
    max_len = max(len(c) for c in channels)
    out = np.zeros((n_mics, max_len))
    for m, c in enumerate(channels):
        out[m, :len(c)] = c
    peak = np.max(np.abs(out))
    return out / peak * 0.95 if peak > 0 else out