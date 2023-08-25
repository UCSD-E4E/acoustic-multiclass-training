# coding=utf-8
# Copyright 2023 The Chirp Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pandas as pd
import numpy as np
import librosa

import torchaudio
import os
from torchaudio import transforms as audtr
import scipy.signal as scipy_signal

# Large parts of this code comes from https://github.com/google-research/chirp/tree/main

def find_peaks_from_melspec(melspec: np.ndarray, stft_fps: int) -> np.ndarray:
    """Locate peaks inside signal of summed spectral magnitudes.

    Args:
        melspec: input melspectrogram of rank 2 (time, frequency).
        stft_fps: Number of summed magnitude bins per second. Calculated from the
        original sample of the waveform.

    Returns:
        A list of filtered peak indices.

    From: https://github.com/google-research/chirp/blob/719eac91dbc716ec1d6d719b3f39367bf0de5acc/chirp/audio_utils.py#L656
    """
    summed_spectral_magnitudes = np.sum(melspec, axis=1)
    threshold = np.mean(summed_spectral_magnitudes) * 1.5
    min_width = int(round(0.5 * stft_fps))
    max_width = int(round(2 * stft_fps))
    width_step_size = int(round((max_width - min_width) / 10))
    peaks = scipy_signal.find_peaks_cwt(
        summed_spectral_magnitudes,
        np.arange(min_width, max_width, width_step_size),
    )
    margin_frames = int(round(0.3 * stft_fps))
    start_stop = np.clip(
        np.stack([peaks - margin_frames, peaks + margin_frames], axis=-1),
        0,
        summed_spectral_magnitudes.shape[0],
    )
    peaks = [
        p
        for p, (a, b) in zip(peaks, start_stop)
        if summed_spectral_magnitudes[a:b].max() >= threshold
    ]
    return np.asarray(peaks, dtype=np.int32)

def pad_to_length_if_shorter(audio: np.ndarray, target_length: int):
    """Wraps the audio sequence if it's shorter than the target length.

    Args:
        audio: input audio sequence of shape [num_samples].
        target_length: target sequence length.

    Returns:
        The audio sequence, padded through wrapping (if it's shorter than the target
        length).

    From: https://github.com/google-research/chirp/blob/719eac91dbc716ec1d6d719b3f39367bf0de5acc/chirp/audio_utils.py#L539
    """
    #print(audio.shape[0] , target_length)
    if audio.shape[0] < target_length:
        missing = target_length - audio.shape[0]
        pad_left = missing // 2
        pad_right = missing - pad_left
        audio = np.pad(audio, [[pad_left, pad_right]], mode='wrap')
    return audio

def apply_mixture_denoising(
    melspec: np.ndarray, threshold: float
) -> np.ndarray:
    """Denoises the melspectrogram using an estimated Gaussian noise distribution.

    Forms a noise estimate by a) estimating mean+std, b) removing extreme
    values, c) re-estimating mean+std for the noise, and then d) classifying
    values in the spectrogram as 'signal' or 'noise' based on likelihood under
    the revised estimate. We then apply a mask to return the signal values.

    Args:
        melspec: input melspectrogram of rank 2 (time, frequency).
        threshold: z-score theshold for separating signal from noise. On the first
        pass, we use 2 * threshold, and on the second pass we use threshold
        directly.

    Returns:
        The denoised melspectrogram.

    From: https://github.com/google-research/chirp/blob/719eac91dbc716ec1d6d719b3f39367bf0de5acc/chirp/audio_utils.py#L539

    """
    x = melspec
    feature_mean = np.mean(x, axis=0, keepdims=True)
    feature_std = np.std(x, axis=0, keepdims=True)
    is_noise = (x - feature_mean) < 2 * threshold * feature_std

    noise_counts = np.sum(is_noise.astype(x.dtype), axis=0, keepdims=True)
    noise_mean = np.sum(x * is_noise, axis=0, keepdims=True) / (noise_counts + 1)
    noise_var = np.sum(
        is_noise * np.square(x - noise_mean), axis=0, keepdims=True
    )
    noise_std = np.sqrt(noise_var / (noise_counts + 1))

    # Recompute signal/noise separation.
    demeaned = x - noise_mean
    is_signal = demeaned >= threshold * noise_std
    is_signal = is_signal.astype(x.dtype)
    is_noise = 1.0 - is_signal

    signal_part = is_signal * x
    noise_part = is_noise * noise_mean
    reconstructed = signal_part + noise_part - noise_mean
    return reconstructed

def find_peaks_from_audio(
    audio,
    sample_rate_hz: int = 32_000,
    max_peaks: int = 200,
    num_mel_bins: int = 194,
    pcen: bool = True
) -> np.ndarray:
    """Construct melspec and find peaks.

    Args:
        audio: input audio sequence of shape [num_samples].
        sample_rate_hz: sample rate of the audio sequence (Hz).
        max_peaks: upper-bound on the number of peaks to return.
        num_mel_bins: The number of mel-spectrogram bins to use.

    Returns:
        Sequence of scalar indices for the peaks found in the audio sequence.

    From: https://github.com/google-research/chirp/blob/719eac91dbc716ec1d6d719b3f39367bf0de5acc/chirp/audio_utils.py#L597
    """
    #melspec_rate_hz = 100
    #frame_length_s = 0.08
    #nperseg = int(frame_length_s * sample_rate_hz)
    #nstep = sample_rate_hz // melspec_rate_hz
    # TODO REVIEW
    hop_length = int(1024//2)
    mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate_hz,
                n_mels=num_mel_bins,
                n_fft=1024, #TODO NOT HARDCODE
                hop_length=hop_length,
                power=1)

    if (pcen):
        mel = librosa.pcen(
                mel * (2**31),
                sr=32_000,
                hop_length = hop_length,
                gain=0.8,
                power=0.25,
                bias=10,
                time_constant=60 * hop_length / 32_000
                #
            )
    mel = np.swapaxes(mel, -1, -2)

    # BY OLD CODE IN GOOGLE
    # apply_mixture_denoising/find_peaks_from_melspec expect frequency axis last
    # mel = mel
    # magnitude_spectrogram = np.abs(spectrogram)
    # For backwards compatibility, we scale the spectrogram here the same way
    # that the TF spectrogram is scaled. If we don't, the values are too small and
    # end up being clipped by the default configuration of the logarithmic scaling
    # magnitude_spectrogram *= nperseg / 2

    # Construct mel-spectrogram
    # num_spectrogram_bins = magnitude_spectrogram.shape[-1]
    # mel_matrix = signal.linear_to_mel_weight_matrix(
    #     num_mel_bins,
    #     num_spectrogram_bins,
    #     sample_rate_hz,
    #     lower_edge_hertz=60,
    #     upper_edge_hertz=10_000,
    # )
    # mel_spectrograms = magnitude_spectrogram @ mel_matrix
    # melspec = log_scale(mel_spectrograms, floor=1e-2, offset=0.0, scalar=0.1)
    
    melspec = apply_mixture_denoising(mel, 0.75)
    
    # TODO REVIEW
    #print(mel.shape, melspec.shape)
    duration_s = audio.shape[0] / sample_rate_hz
    #print(duration_s, melspec.shape[0])
    melspec_rate_hz = int(melspec.shape[0]/duration_s)
    #print(melspec_rate_hz)

    peaks = find_peaks_from_melspec(melspec, melspec_rate_hz)
    peak_energies = np.sum(melspec, axis=1)[peaks]

    t_mel_to_t_au = lambda tm: 1.0 * tm * sample_rate_hz / melspec_rate_hz
    peaks = [t_mel_to_t_au(p) for p in peaks]

    peak_set = sorted(zip(peak_energies, peaks), reverse=True)
    if max_peaks > 0 and len(peaks) > max_peaks:
        peak_set = peak_set[:max_peaks]
    return np.asarray([p[1] for p in peak_set], dtype=np.int32)

def slice_peaked_audio(
    audio: np.ndarray,
    sample_rate_hz: int,
    interval_length_s: float = 5.0,
    max_intervals: int = 200,
) -> np.ndarray:
    """Extracts audio intervals from melspec peaks.

    Args:
        audio: input audio sequence of shape [num_samples].
        sample_rate_hz: sample rate of the audio sequence (Hz).
        interval_length_s: length each extracted audio interval.
        max_intervals: upper-bound on the number of audio intervals to extract.

    Returns:
        Sequence of extracted audio intervals, each of shape
        [sample_rate_hz * interval_length_s].

    From: https://github.com/google-research/chirp/blob/719eac91dbc716ec1d6d719b3f39367bf0de5acc/chirp/audio_utils.py#L558
    """
    target_length = int(sample_rate_hz * interval_length_s)

    # Wrap audio to the target length if it's shorter than that.
    audio = pad_to_length_if_shorter(audio, target_length)

    peaks = find_peaks_from_audio(audio, sample_rate_hz, max_intervals)
    #print(peaks)
    left_shift = target_length // 2
    right_shift = target_length - left_shift

    # Ensure that the peak locations are such that
    # `audio[peak - left_shift: peak + right_shift]` is a non-truncated slice.
    peaks = np.clip(peaks, left_shift, audio.shape[0] - right_shift)
    # As a result, it's possible that some (start, stop) pairs become identical;
    # eliminate duplicates.
    start_stop = np.unique(
        np.stack([peaks - left_shift, peaks + right_shift], axis=-1), axis=0
    )

    return start_stop


def get_peak_chunks(files, data_path="/", allowed_extensions=["mp3", "flac"], sr=32_000, interval_length_s=5, max_intervals=10):
    file_dfs = []
    for file in files:
        if not file.split(".")[-1] in allowed_extensions:
            continue
        
        # Compute Chunks
        audio, sample_rate = torchaudio.load(       #pyright: ignore [reportGeneralTypeIssues ]
            os.path.join(data_path, file)
        )
        resample = audtr.Resample(sample_rate, sr)
        audio = resample(audio)
        audio = audio.mean(axis=0)

        start_end = slice_peaked_audio(
            audio.detach().numpy(),
            sr,
            interval_length_s=interval_length_s,
            max_intervals = max_intervals,
        )

        # Create Output
        file_df = pd.DataFrame(start_end, columns=["OFFSET", "END"])
        file_df["IN FILE"] = 'test_data/PER_021_S10_20190131_100007Z_18.flac'
        file_df["MANUAL ID"] = 'BIRD'
        file_df["SAMPLE RATE (file)"] = sample_rate
        file_df["OFFSET"] = file_df["OFFSET"] / sr
        file_df["DURATION"] = file_df["END"]/sr - file_df["OFFSET"]
        file_df[["IN FILE", "SAMPLE RATE", "OFFSET", "DURATION", "MANUAL ID"]]
        file_dfs.append(file_df)

    return pd.concat(file_dfs)