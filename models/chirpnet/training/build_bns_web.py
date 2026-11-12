"""Page data for the neutron-star section: GW170817 spectrograms (combined, raw Livingston with its glitch, gated),
the sliding-window score, background, audio, and the BNS rungs."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, base64, numpy as np
sys.path.insert(0, _P.SRC)
import bns_real as R
from viz import cwt_energy

def u8(a): return base64.b64encode(np.asarray(a, np.uint8).tobytes()).decode()
def f16(a): return base64.b64encode(np.asarray(a, '<f2').tobytes()).decode()

T_LO, T_HI, NF, NT = -32.0, 2.0, 96, 544
FREQS = np.geomspace(24, 420, NF)

def spec(x, a, fs=1024, ref=None):
    E = cwt_energy(x[a - 2 * fs: a + int((T_HI - T_LO) * fs) + 2 * fs].astype(np.float64), fs, FREQS, Q=18)[:, 2 * fs:-2 * fs]
    if ref is not None:      # normalise by a clean reference (so a glitch saturates instead of dimming everything)
        E = E * ref
    return E.reshape(NF, NT, -1).mean(-1)

def main(rungs, out=_P.LAB + '/bns.json'):
    best = [r for r in rungs if r.get('accepted')][-1]; ck = best['ckpt']
    real = json.load(open(ck.replace('.pt', '_real.json')))
    D = R.prepare(); t0 = float(D['t0']); fs = 1024; a = int((R.GPS_EVENT + T_LO - t0) * fs)
    gH, _ = R.gate(D['H']); gL, _ = R.gate(D['L'])
    EH = spec(gH, a); EL = spec(gL, a); comb = EH + EL
    # raw Livingston: normalise each frequency row by the clean data's level, so the glitch shows at its true strength
    Er = cwt_energy(D['Lraw'][a - 2 * fs: a + 36 * fs].astype(np.float64), fs, FREQS, Q=18)
    El = cwt_energy(gL[a - 2 * fs: a + 36 * fs].astype(np.float64), fs, FREQS, Q=18)
    sc = np.mean(El[:, 2 * fs:-2 * fs], 1, keepdims=True) / np.maximum(1e-12, np.mean(Er[:, 2 * fs:-2 * fs] * (Er[:, 2 * fs:-2 * fs] < 50 * np.median(Er)), 1, keepdims=True))
    Eraw = (Er[:, 2 * fs:-2 * fs] * sc).reshape(NF, NT, -1).mean(-1)
    gLr, spans = R.gate(D['Lraw'])
    Egated = spec(gLr, a)
    # audio: whitened H1+L1 (L1 shifted by the 3 ms light-travel offset is negligible here), 30-400 Hz, last 14 s
    from scipy.signal import butter, sosfiltfilt
    sos = butter(6, [30, 400], btype='band', fs=fs, output='sos'); e = int((R.GPS_EVENT - t0) * fs)
    aud = sosfiltfilt(sos, (gH + gL)[e - 16 * fs: e + fs])[fs: -int(0.5 * fs)]; aud = aud / np.abs(aud).max()
    # the network's chirp, at its own chirp mass (leading-order phase, whitened-like flat amplitude)
    GM = 4.925490947e-6; mc = real['mc_combined']; t = np.arange(len(aud)) / fs - 15.0; tau = np.maximum(-t, 1e-3)
    ph = -2 * (tau / (5 * GM * mc)) ** 0.625; f = (1 / np.pi) * (5 / (256 * tau)) ** 0.375 * (mc * GM) ** -0.625
    tpl = np.where((t < 0) & (f < 450) & (f > 28), np.cos(ph) * (f / 100) ** (-1 / 6), 0.0)
    tpl *= np.clip((f - 28) / 6, 0, 1); tpl /= np.abs(tpl).max()
    pk = real['peak']
    # the released numpy model's own scan of the two minutes around the event
    sys.path.insert(0, _P.WORK + '/release'); import chirpnet
    bn = chirpnet.ChirpNetBNS(_P.WORK + '/release/chirpnet_bns_weights.npz'); k0 = int((R.GPS_EVENT - 60 - t0) * fs)
    hits = bn.scan(D['H'][k0:k0 + 120 * fs], D['L'][k0:k0 + 120 * fs], gps_start=t0 + k0 / fs, whitened=True)
    rh = dict(hits[0]); rh['dt_ms'] = 1000 * (rh['gps'] - R.GPS_EVENT); rh['threshold'] = bn.meta['threshold']
    out_d = {'freqs': FREQS.round(2).tolist(), 't_lo': T_LO, 't_hi': T_HI, 'nf': NF, 'nt': NT,
             'spec': u8(np.clip(comb * 18, 0, 255)), 'specRaw': u8(np.clip(Eraw * 18, 0, 255)), 'specGated': u8(np.clip(Egated * 18, 0, 255)),
             'gate': [[s + t0 - R.GPS_EVENT for s in sp] for sp in spans],
             'scan_t': (np.array(real['scan_t']) + 16.0).round(3).tolist(), 'scan_s': real['scan_s'], 'scan_raw_nogate': real['raw_nogate']['scan_s'],
             'peak': pk, 'mc_combined': real['mc_combined'], 'mc_ref': R.MC_DET, 'mc_src_ref': 1.186, 'z_ref': 0.0099, 'dl_ref': 40,
             'raw_glitch': {k: v for k, v in real['raw_glitch'].items() if k != 'scan_s'}, 'raw_nogate': real['raw_nogate']['score'],
             'bg_n': real['bg_n'], 'bg_live_s': real['bg_live_s'], 'bg_max': real['bg_max'], 'bg_hist': real['bg_hist'], 'n_bg_above_event': real['n_bg_above_event'],
             'single': real['single'], 'audio': f16(aud), 'tpl': f16(tpl), 'audio_fs': fs, 'audio_t0': -15.0,
             'rungs': rungs, 'release': best['name'], 'hit': rh}
    json.dump(out_d, open(out, 'w'))
    print('bns.json', {k: out_d[k] for k in ('peak', 'mc_combined', 'bg_max', 'bg_n', 'raw_nogate', 'single', 'hit')})

if __name__ == '__main__':
    main(json.load(open(_P.WORK + '/bns_rungs.json')))
