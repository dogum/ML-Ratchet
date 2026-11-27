"""Mono clap preprocessing shared by the reference estimator and RoomSense."""
from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.io import wavfile

RATE = 16000
BANDS = (500, 1000, 2000)
FRAMES = 256
HOP = RATE // 100
EARLY_HOP = RATE // 1000
MIN_PRE = RATE // 20
MIN_POST = RATE * 2 // 5
FALLBACK = 0.65
TINY = 1e-20


@dataclass(frozen=True)
class Features:
    curves: np.ndarray  # three early envelopes at 1 ms, three Schroeder curves at 10 ms
    context: np.ndarray  # log T20 and validity for each of three bands
    onset_seconds: float
    snr_db: float
    classical_seconds: float | None
    band_seconds: tuple


def read_wav(path):
    sample_rate, samples = wavfile.read(path)
    if samples.dtype == np.uint8:
        samples = (samples.astype(float) - 128) / 128
    elif np.issubdtype(samples.dtype, np.signedinteger):
        samples = samples.astype(float) / (-int(np.iinfo(samples.dtype).min))
    return samples, sample_rate


def resample(samples, sample_rate):
    """Windowed-sinc converter. Use the same finite kernel in the browser."""
    if sample_rate == RATE:
        return samples
    ratio = sample_rate / RATE
    cutoff = min(1.0, 1.0 / ratio)
    radius = int(np.ceil(12 / cutoff))
    offsets = np.arange(-radius + 1, radius + 1)
    result = np.empty(int(np.floor(len(samples) / ratio)))
    for start in range(0, len(result), 4096):
        position = np.arange(start, min(start + 4096, len(result))) * ratio
        indices = position.astype(int)[:, None] + offsets
        delta = indices - position[:, None]
        w = np.sinc(cutoff * delta) * (0.5 + 0.5 * np.cos(np.pi * delta / radius))
        w[np.abs(delta) >= radius] = 0
        result[start:start + len(w)] = (w * samples[indices.clip(0, len(samples) - 1)]).sum(1) / w.sum(1)
    return result


def band_filter(samples, frequency):
    angle = 2 * np.pi * frequency / RATE
    alpha = np.sin(angle) / (2 * np.sqrt(2))
    return signal.lfilter([alpha, 0, -alpha], [1 + alpha, -2 * np.cos(angle), 1 - alpha], samples)


def frame_power(samples, hop, frames):
    power = np.square(samples[:hop * frames])
    padding = max(0, hop * frames - len(power))
    return np.pad(power, (0, padding)).reshape(frames, hop).mean(1)


def energy_decay(power, noise):
    """Backward integration with expected noise energy subtracted."""
    # A terminal integration artifact can look like a decay. Mark the last
    # frame with meaningful energy and use it to bound the fitted range.
    above = np.flatnonzero(power >= max(4 * noise, power.max() * 1e-8, TINY))
    stop = int(above[-1]) + 1 if len(above) else 0
    energy = np.zeros(len(power))
    energy[:stop] = np.cumsum((power[:stop] - noise)[::-1])[::-1]
    energy = np.minimum.accumulate(np.clip(energy / max(energy[0], TINY), 1e-8, 1))
    return 10 * np.log10(energy), stop


def estimate_t20(decay_db, stop):
    points = np.flatnonzero((decay_db <= -5) & (decay_db >= -25) & (np.arange(len(decay_db)) < stop - 1))
    if len(points) < 4 or decay_db[points[-1]] > -23 or decay_db[points[0]] < -9:
        return None
    x = points * (HOP / RATE)
    y = decay_db[points]
    dx = x - x.mean()
    dy = y - y.mean()
    slope = float(dx @ dy / (dx @ dx))
    if slope >= 0:
        return None
    fit_r2 = float((dx @ dy)**2 / ((dx @ dx) * (dy @ dy)))
    reading = -60 / slope
    return reading if fit_r2 >= 0.85 and 0.08 <= reading <= 6.0 else None


def _onset(samples):
    hop = EARLY_HOP
    power = frame_power(samples, hop, len(samples) // hop)
    peak = int(np.argmax(power))
    start = peak
    # Walk back to the leading edge near the loudest short-time burst.
    while start > max(0, peak - 8) and power[start - 1] > 0.015 * power[peak]:
        start -= 1
    return start * hop, float(power[peak])


def preprocess(samples, sample_rate):
    """Read one unclipped clap with quiet lead-in and at least 0.4 s of tail."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1:
        raise ValueError("RoomSense takes a mono recording.")
    if not 8000 <= sample_rate <= 96000 or not 0.45 * sample_rate <= len(samples) <= 30 * sample_rate:
        raise ValueError("Provide 0.45 to 30 seconds of audio at 8 to 96 kHz.")
    if not np.isfinite(samples).all():
        raise ValueError("The recording contains invalid samples.")
    if np.count_nonzero(np.abs(samples) >= 0.999) >= 3:
        raise ValueError("The recording is clipped.")
    samples = resample(samples - samples.mean(), sample_rate)
    onset, peak_power = _onset(samples)
    if onset < MIN_PRE or len(samples) - onset < MIN_POST:
        raise ValueError("Leave at least 50 ms before the clap and 400 ms after it.")
    pre_end = onset - RATE // 100
    pre_start = max(0, pre_end - RATE // 4)
    noise = float(np.mean(samples[pre_start:pre_end]**2))
    snr = 10 * np.log10(max(peak_power, TINY) / max(noise, TINY))
    if peak_power < 1e-12 or snr < 15:
        raise ValueError("The clap is too quiet relative to the background.")
    segment_start = max(0, pre_start - RATE // 50)
    segment_end = onset + HOP * FRAMES
    curves, readings, decays = [], [], []
    for frequency in BANDS:
        filtered = band_filter(samples[segment_start:segment_end], frequency)
        band_noise = np.mean(filtered[pre_start - segment_start:pre_end - segment_start]**2)
        tail = filtered[onset - segment_start:]
        early = frame_power(tail, EARLY_HOP, FRAMES)
        early_db = 10 * np.log10(np.clip(early / max(early.max(), TINY), 1e-6, 1))
        curves.append(early_db / 60)
        frames = min(FRAMES, len(tail) // HOP)
        decay, stop = energy_decay(frame_power(tail, HOP, frames), band_noise)
        readings.append(estimate_t20(decay, stop))
        decays.append(np.pad(decay, (0, FRAMES - frames), constant_values=-80) / 80)
    valid = [v for v in readings if v is not None]
    context = np.array([np.log(v if v is not None else FALLBACK) for v in readings]
                       + [v is not None for v in readings], dtype=np.float32)
    return Features(np.asarray(curves + decays, dtype=np.float32), context, onset / RATE, snr,
                    float(np.median(valid)) if len(valid) >= 2 else None, tuple(readings))
