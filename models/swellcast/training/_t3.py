import numpy as np, time, simbank as SB, evaluate as E
BG = SB.Backgrounds(); rng = np.random.default_rng(0); t = time.time(); Y = []
for i in range(60): x, y = SB.make_one(rng, BG, 'v3'); Y.append(y)
Y = np.array(Y); print('s/each', (time.time() - t) / 60); print('D', np.percentile(Y[:, 0], [10, 50, 90]).round(), 'hs', np.percentile(Y[:, 3], [10, 50, 90]).round(1), 'contrast>=.3', np.mean(Y[:, 4] >= 0.3), 'speed', np.nanmedian(Y[:, 8]))
