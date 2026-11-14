"""Build several simulation banks with one worker; run several copies to share the work (chunks are claimed with lock
files, see simbank.build). usage: python bankq.py tag:n:seed:K:prior [tag:n:seed:K:prior ...]"""
from common import *
import simbank

if __name__ == '__main__':
    for job in sys.argv[1:]:
        tag, n, seed, K, prior = job.split(':')
        simbank.build(tag, int(n), int(seed), int(K), prior)
    print('bankq done', flush=True)
