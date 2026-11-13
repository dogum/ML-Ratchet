"""How TripDetective was built, step by step (scripts in training/; paths assume the ML Ratchet vehicle workspace with
road/roadsim (RoadSim) and road/realdata + road/data (RoadData) next to the training folder). Each step is resumable.
Run `python3 pipeline.py` to print the commands, `python3 pipeline.py run <step>` to run one."""
import subprocess, sys

STEPS = [
    ('routes', 'python3 routes.py', 'route pool: 1,154 real Ann Arbor route fragments (VED train_noise cars, no labels) + 10 light-duty cycles'),
    ('cadence', 'python3 cadence_probe.py', 'per-signal update cadence of the logger, measured on the 16 dev cars\' raw rows'),
    ('twin', 'python3 twin.py 15 v2b v2', 'dev twin test: real dev trips vs their RoadSim twins through the OBD emulator (sets priors.py v2)'),
    ('realbank', 'python3 realbank.py dev 40 && python3 realbank.py exam 40', 'real VED trips -> network channels (at most 40 per car, seed 0)'),
    ('bank_v1', 'python3 simbank.py val_v1 600 99 8 v1 && python3 simbank.py train_v1 1200 1 8 v1', 'simulation bank, generator as published'),
    ('bank_v2', 'python3 simbank.py val_v2 600 99 8 v2 && python3 simbank.py train_v2 2000 2 8 v2', 'bank with dev-twin calibrated priors'),
    ('bank_v3', 'python3 simbank.py val_v3 600 99 8 v3 && python3 simbank.py train_v3 2400 3 8 v3', 'bank with the VED-era mass prior'),
    ('train', 'python3 train.py <bank> <name> 2500 [chunks=..] [feat=2]', 'train one rung (resumable)'),
    ('ratchet', 'python3 ratchet.py <name>.pt <val_tag> "<note>"  |  python3 ratchet.py --rescore <val_tag>', 'score a rung: frozen sim set + dev gate'),
    ('truth', 'python3 exam_truth.py', 'road-load truth for high-confidence VED matches (EPA coast-down targets)'),
    ('exam', 'python3 exam.py <final>.pt', 'final exam on real VED exam cars + baselines + pooling curve (run once)'),
    ('twins', 'python3 twin_ledger.py <final>.pt dev 12', 'twin test of the network + energy ledger on simulated twins'),
    ('epa', 'python3 epa_sim.py <final>.pt 50 4', 'EPA exam vehicles: real parameters, simulated logs'),
    ('export', 'python3 export.py <final>.pt \'{"sigk": {...}}\'', 'numpy weights (BatchNorm folded), numpy == torch check, browser pack'),
    ('web', 'python3 build_lab.py <final>.pt && python3 webdata.py <final>.pt', 'page data'),
]

if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'run':
        cmd = dict((s[0], s[1]) for s in STEPS)[sys.argv[2]]
        sys.exit(subprocess.call(cmd, shell=True, cwd='training'))
    for name, cmd, what in STEPS:
        print(f'{name:9s} {what}\n          {cmd}')
