"""RoomSense input checks and the published exam's room isolation."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import mlratchet


ROOT = Path(__file__).resolve().parents[1] / 'models' / 'roomsense'


def test_locked_exam_uses_only_held_out_rooms():
    training = ROOT / 'training'
    split_path = training / 'split.json'
    split = json.loads(split_path.read_text())
    lock = json.loads((training / 'release_lock.json').read_text())
    exam = json.loads((training / 'measured_exam.json').read_text())
    assert lock == exam['locked_release']
    assert lock['split_sha256'] == sha256(split_path.read_bytes()).hexdigest()
    assert lock['weights_sha256'] == sha256((ROOT / 'roomsense_weights.npz').read_bytes()).hexdigest()
    expected = {f"{r['source']}/{r['name']}" for r in split['rooms'] if r['role'] == 'exam'}
    assert {case['room'] for case in exam['cases']} == expected
    assert len({case['case'] for case in exam['cases']}) == 38
    assert len({r['source_id'] for r in split['noise']}) == len(split['noise'])


def test_inference_returns_intervals_and_validates_audio():
    model = mlratchet.load('roomsense').RoomSense()
    with np.load(ROOT / 'examples' / 'room-a-27db.npz', allow_pickle=False) as data:
        audio = data['audio']
    out = model.read(audio, 16000)
    assert out['rt60_interval_s'][0] < out['rt60_s'] < out['rt60_interval_s'][1]
    assert out['volume_interval_m3'][0] < out['volume_m3'] < out['volume_interval_m3'][1]
    with pytest.raises(ValueError, match='mono'):
        model.read(np.column_stack((audio, audio)), 16000)
    clipped = audio.copy()
    clipped[:3] = 1
    with pytest.raises(ValueError, match='clipped'):
        model.read(clipped, 16000)
    start = round(out['onset_s'] * 16000)
    with pytest.raises(ValueError, match='before'):
        model.read(audio[start - 100:], 16000)


def test_room_physics_and_known_decays():
    proc = subprocess.run([sys.executable, str(ROOT / 'training' / 'physics_check.py')],
                          capture_output=True, text=True, check=True)
    assert json.loads(proc.stdout)['max_relative_decay_error'] < 0.02
