"""A released numpy TripDetective (its own tripdetective.py + weights) behind evaluate.predict's interface, so an older
release can be re-scored on the same exam with the same code (Phase D: v1.0 vs v2.0).
usage: net = npnet.load('<folder with tripdetective.py and tripdetective_weights.npz>'); evaluate.predict(net, ...)"""
from common import *
import importlib.util
import features as F


class NumpyNet:
    def __init__(self, folder):
        spec = importlib.util.spec_from_file_location(f'td_{abs(hash(folder))}', os.path.join(folder, 'tripdetective.py'))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        self.mod = mod
        self.td = mod.TripDetective(os.path.join(folder, 'tripdetective_weights.npz'))
        self.feat = self.td.feat
        F.FEAT = self.feat

    def np_predict(self, inputs, groups):
        """inputs: (x, mask, sc) from features.net_input (feature version self.feat); groups: pool index per input"""
        groups = np.asarray(groups); G = int(groups.max()) + 1
        E = [self.td.encode(x, m, sc) for x, m, sc in inputs]
        out = dict(arch=[], strat=[], mu=[], sig=[])
        led_mu = np.zeros((len(inputs), 6)); led_sig = np.zeros((len(inputs), 6))
        for g in range(G):
            js = np.where(groups == g)[0]
            try:
                h = self.td.heads([E[j] for j in js], [inputs[j][2] for j in js])
            except TypeError:                                  # v1.0's heads(E) takes no classical numbers
                h = self.td.heads([E[j] for j in js])
            out['arch'].append(h['arch']); out['strat'].append(h['strat']); out['mu'].append(h['mu']); out['sig'].append(h['sig'])
            for i, j in enumerate(js):
                led_mu[j] = h['led'][i][0]; led_sig[j] = h['led'][i][1]
        return {k: np.stack(v) for k, v in out.items()} | dict(led_mu=led_mu, led_sig=led_sig)


def load(folder):
    return NumpyNet(folder)
