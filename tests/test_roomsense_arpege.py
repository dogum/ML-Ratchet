"""End-to-end ARPEGE exam mechanics on invented audio fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from scipy.io import wavfile


TRAINING = Path(__file__).resolve().parents[1] / 'models' / 'roomsense' / 'training'
SCRIPT = TRAINING / 'arpege_exam.py'


def test_arpege_plan_covers_six_independent_rooms():
    plan = json.loads((TRAINING / 'arpege_exam_plan.json').read_text())
    rooms = plan['rooms']
    assert len({r['id'] for r in rooms}) == 6
    pairs = [p for room in rooms for p in room['pairs']]
    assert len({p['claps'] for p in pairs}) == len({p['sweep_response'] for p in pairs}) == 28
    for pair in pairs:
        assert pair['claps'].split('/')[:3] == pair['sweep_response'].split('/')[:3]


@pytest.mark.parametrize('official_archives', [False, True])
def test_frozen_recorded_exam_can_be_scored_and_reproduced(tmp_path, official_archives):
    plan = json.loads((TRAINING / 'arpege_exam_plan.json').read_text())
    raw = tmp_path / 'original'
    rng = np.random.default_rng(991)
    rate = 16000
    tail_time = np.arange(rate) / rate
    for index, room in enumerate(plan['rooms'][:3]):
        pair = room['pairs'][0]
        impulse = 0.04 * rng.normal(size=rate) * 10 ** (-3 * tail_time / (0.5 + 0.1 * index))
        impulse[0] = .3
        clap = rng.normal(0, 0.0002, 3 * rate)
        onset = rate // 2 if index < 2 else 0
        clap[onset:onset + rate] += impulse
        if index == 0:
            clap[2 * rate:] += 0.65 * impulse
        for key, wave in [('claps', clap), ('sweep_response', impulse)]:
            path = raw / pair[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            # A deliberately unusable second channel makes channel selection observable.
            wavfile.write(path, rate, np.column_stack([wave, np.zeros_like(wave)]).astype(np.float32))

    env = dict(os.environ, ML_RATCHET_WORK=str(tmp_path / 'scratch'))

    def run(*args, check=True):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], env=env,
                              text=True, capture_output=True, check=check)

    archive = tmp_path / 'mono.zip'
    if official_archives:
        downloads = tmp_path / 'downloads'
        downloads.mkdir()
        with zipfile.ZipFile(downloads / 'ARPEGE_01.2026_claps.zip', 'w') as claps:
            for i, room in enumerate(plan['rooms'][:3], 1):
                pair = room['pairs'][0]
                name = Path(pair['claps'])
                prefix = Path('RIR_CEREMA_01.2026_claps').joinpath(*name.parts[1:-1])
                clap_member = prefix / (name.stem + f'_{i}.wav')
                claps.write(raw / name, clap_member)
                claps.write(raw / name, str(clap_member).replace('/EM32/', '/EM64/'))
                with zipfile.ZipFile(downloads / f'room{i}_RIR.zip', 'w') as sweeps:
                    ref = Path(pair['sweep_response'])
                    sweeps.write(raw / ref, Path(*ref.parts[1:]))
        run('pack', downloads, archive, '--room', 'room1', '--room', 'room2', '--room', 'room3')
    else:
        run('pack', raw, archive)
    transfer = tmp_path / 'transfer'
    with zipfile.ZipFile(archive) as z:
        z.extractall(transfer)
    run('freeze', transfer)
    completed = run('score', transfer)
    metrics = json.loads(completed.stdout)
    assert metrics['room_count'] == 3
    assert metrics['cases'] == 3 and metrics['accepted'] == 2
    assert metrics['model_rt'] > 0
    result = json.loads((tmp_path / 'scratch/roomsense/logs/arpege_exam.json').read_text())
    assert result['room_target_met']
    assert len(result['missing_pairs']) == 25
    assert list(result['input_abstentions']) == ['arpege/room3/pos1']
    assert run('score', transfer, check=False).returncode != 0
    assert json.loads(run('score', transfer, '--reproduce').stdout) == metrics
    path = transfer / plan['rooms'][0]['pairs'][0]['claps']
    path.write_bytes(path.read_bytes() + b'x')
    assert 'differs from the frozen files' in run('score', transfer, '--reproduce', check=False).stderr
