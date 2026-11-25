"""Run round-2 jobs with at most N processes at once (one thread each). Skips jobs whose done-file exists.

usage: python queue2.py N <job> [<job> ...]
  job "seq:<bank>:<config>:<seed>"   train_seq.py, then eval2.py on its own bank (sim val + dev) and, if the bank is
                                     not p1, the sim val of p1 (tag p1val). Outputs: work/eval/<run>/.
Logs: work/logs/<run>.log.
"""
import os
import subprocess
import sys
import time

import r2

PY = sys.executable


def seq_cmds(bank, cfg, seed):
    run = f'{bank}_{cfg}_s{seed}'
    cand = os.path.join(r2.WORK, 'cand', run)
    out = os.path.join(r2.WORK, 'eval', run)
    cmds = [[PY, '-X', 'utf8', 'train_seq.py', bank, cfg, str(seed)],
            [PY, '-X', 'utf8', 'eval2.py', bank, cand, '--sets', 'simval,dev', '--tag', 'own', '--out', out]]
    if bank != 'p1':
        cmds.append([PY, '-X', 'utf8', 'eval2.py', 'p1', cand, '--sets', 'simval', '--tag', 'p1val', '--out', out])
    return run, cmds, os.path.join(out, 'DONE')


def main():
    n = int(sys.argv[1])
    jobs = []
    for j in sys.argv[2:]:
        kind, *a = j.split(':')
        if kind == 'seq':
            jobs.append(seq_cmds(a[0], a[1], int(a[2])))
    os.makedirs(os.path.join(r2.WORK, 'logs'), exist_ok=True)
    pending = [j for j in jobs if not os.path.exists(j[2])]
    running = []
    env = dict(os.environ, PYTHONWARNINGS='ignore')

    def launch(job):
        run, cmds, done = job
        os.makedirs(os.path.dirname(done), exist_ok=True)
        script = ' && '.join(subprocess.list2cmdline(c) for c in cmds) + f' && echo ok > "{done}"'
        log = open(os.path.join(r2.WORK, 'logs', f'{run}.log'), 'a')
        p = subprocess.Popen(script, shell=True, cwd=r2.ROUND2, stdout=log, stderr=subprocess.STDOUT, env=env)
        print(time.strftime('%H:%M:%S'), 'start', run, flush=True)
        return (run, p)

    while pending or running:
        while pending and len(running) < n:
            running.append(launch(pending.pop(0)))
        time.sleep(5)
        for r in list(running):
            if r[1].poll() is not None:
                print(time.strftime('%H:%M:%S'), 'end', r[0], 'rc', r[1].returncode, flush=True)
                running.remove(r)


if __name__ == '__main__':
    main()
