"""Image-source arrivals, Eyring reverberation, synthetic claps and a separate instrument."""
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy import signal

from pipeline import BANDS, HOP, RATE, band_filter, energy_decay, estimate_t20, frame_power

PHYSICS_VERSION = '1.0'
DIFFUSE_DENSITY = {'1.0': 0.39, '1.1': 0.3159072888185475}
SOUND_SPEED = 343.0
IMAGE_ORDER = 2
DURATION = 2.7
PRE = 0.16


@dataclass(frozen=True)
class Room:
    dimensions: np.ndarray
    source: np.ndarray
    receiver: np.ndarray
    absorption: np.ndarray  # six faces, three octave bands

    @property
    def volume(self):
        return float(np.prod(self.dimensions))


def sample_room(rng):
    dimensions = np.exp(rng.uniform(np.log([2.3, 2.4, 2.2]), np.log([19, 25, 6.5])))
    source, receiver = rng.uniform(0.13, 0.87, (2, 3)) * dimensions
    center = np.exp(rng.uniform(np.log(0.07), np.log(0.52)))
    # Face-to-face variation and typically more absorption toward high frequencies.
    absorption = np.clip(center * rng.lognormal(0, 0.40, (6, 1))
                         * rng.lognormal(0, 0.16, (6, 3)) * [0.8, 1.0, 1.16], 0.025, 0.9)
    return Room(dimensions, source, receiver, absorption)


def _axis_image(index, length, source):
    if index % 2:
        image = (index + 1) * length - source
    else:
        image = index * length + source
    count = abs(index)
    low = (count + (index < 0)) // 2
    high = count - low
    return image, (low, high)


_IMAGE_IDS = tuple(indices for indices in product(range(-IMAGE_ORDER, IMAGE_ORDER + 1), repeat=3)
                   if sum(map(abs, indices)) <= IMAGE_ORDER)


def image_paths(room):
    """Return path distances, order and pressure reflection coefficients per band."""
    rows = []
    for indices in _IMAGE_IDS:
        images = [_axis_image(i, l, s) for i, l, s in zip(indices, room.dimensions, room.source)]
        distance = np.linalg.norm(np.array([a for a, _ in images]) - room.receiver)
        hits = np.ravel([counts for _, counts in images])
        coeff = np.prod((1.0 - room.absorption)**(hits[:, None] / 2), axis=0)
        rows.append((float(distance), int(hits.sum()), coeff))
    return rows


def eyring(room):
    x, y, z = room.dimensions
    areas = 2 * np.array([y * z, x * z, x * y])
    alpha = (areas[:, None] * room.absorption.reshape(3, 2, 3).mean(1)).sum(0) / areas.sum()
    return 24 * np.log(10) / SOUND_SPEED * room.volume / (-areas.sum() * np.log1p(-alpha))


def clean_response(room, rng, version=PHYSICS_VERSION):
    """A pressure IR. Align direct arrival to sample 32, preserving relative delays."""
    samples = int(DURATION * RATE)
    paths = image_paths(room)
    direct = next(distance for distance, order, _ in paths if order == 0)
    impulses = np.zeros((3, samples))
    for distance, _, pressure in paths:
        position = 32 + (distance - direct) * RATE / SOUND_SPEED
        i = int(position)
        gain = pressure / max(distance, 0.25)
        impulses[:, i] += gain * (1 - (position - i))
        impulses[:, i + 1] += gain * (position - i)
    # Sum a broadband image response and octave corrections relative to 1 kHz.
    response = impulses[1].copy()
    response += band_filter(impulses[0] - impulses[1], BANDS[0])
    response += band_filter(impulses[2] - impulses[1], BANDS[2])
    dimensions = room.dimensions
    mean_free_path = 2 / np.sum(1 / dimensions)
    mix_time = np.clip(4 * mean_free_path / SOUND_SPEED, 0.015, 0.095)
    time = np.arange(samples) / RATE - mix_time
    valid_time = np.maximum(time, 0)
    fade = np.clip(time / 0.012, 0, 1)
    rt = np.clip(eyring(room), 0.11, 3.7)
    # The pressure envelope reaches 0.001 at RT60. The independent octave
    # envelopes use different decay rates, before the microphone and noise.
    wall_decay = np.power(10, -3 * valid_time[None, :] / rt[:, None])
    density = DIFFUSE_DENSITY[version]
    for frequency, decay in zip(BANDS, wall_decay):
        diffuse = band_filter(rng.normal(size=samples), frequency)
        response += density * fade * decay * diffuse / max(1.0, mix_time * SOUND_SPEED)
    return response


def reference_seconds(response):
    """Read the target from the clean response, before clap and recording noise."""
    readings = []
    for frequency in BANDS:
        band = band_filter(response, frequency)
        frames = len(band) // HOP
        decay, stop = energy_decay(frame_power(band, HOP, frames), 0)
        fit = estimate_t20(decay, stop)
        if fit is not None:
            readings.append(fit)
    return float(np.median(readings)) if len(readings) >= 2 else None


def clap(rng):
    length = rng.integers(48, 144)
    time = np.linspace(0, 1, length)
    pulse = rng.normal(size=length) * (1 - np.exp(-time * 22)) * np.exp(-time * 7)
    # A second contact can follow the main clap.
    if rng.random() < 0.35:
        delay = int(rng.integers(8, length // 2))
        pulse[delay:] += rng.uniform(0.1, 0.5) * pulse[:-delay]
    return pulse / np.linalg.norm(pulse)


def record(response, excitation, noise, rng, snr=None, instrument=True):
    pressure = signal.fftconvolve(response, excitation)[:int(DURATION * RATE)]
    wave = np.zeros(int(PRE * RATE) + int(DURATION * RATE))
    if instrument:
        mic = float(rng.uniform(-0.32, 0.50))
        pressure = signal.lfilter([1.0, mic], [1.0], pressure)
    wave[int(PRE * RATE):int(PRE * RATE) + len(pressure)] = pressure
    # Match noise RMS to the strongest 1-ms clap frame. Keep true pre-clap noise.
    peak = np.sqrt(frame_power(wave, RATE // 1000, len(wave) // (RATE // 1000)).max())
    noise = np.asarray(noise, dtype=float)
    if len(noise) < len(wave):
        raise ValueError("The background recording is shorter than the measurement window.")
    offset = rng.integers(len(noise) - len(wave) + 1)
    noise = noise[offset:offset + len(wave)]
    noise = noise - np.mean(noise)
    db = float(rng.uniform(22, 52) if snr is None else snr)
    wave += noise * (peak * 10**(-db / 20) / max(1e-12, np.std(noise)))
    return (wave * (0.8 / max(1e-12, np.max(np.abs(wave))))).astype(np.float32)
