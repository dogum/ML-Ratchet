"""Simulator calibrations measured on the 18 dev stars only (never the exam). Each version adds to the previous one.
v2: (a) amplitude convention: Yu et al. 2018 amplitudes follow Kjeldsen et al. 2008 (a sine of amplitude A puts A^2
        in the power spectrum), i.e. A = sqrt(2) x rms. The dev twin test showed the v1 humps 0.24 dex too high.
    (b) envelope width: the dev stars' published widths are narrower than Mosser 2012's relation at high numax;
        fitted on dev: width = 0.66 numax^0.88 x exp(0.073 - 0.058 ln numax).
"""
import numpy as np

def v2(s, rng):
    s['amp'] /= np.sqrt(2)
    s['fwhm'] *= float(np.exp(0.073 - 0.058 * np.log(s['numax'])))

VERSIONS = {'v2': [v2]}

def make(name):
    fs = VERSIONS[name]
    def calib(s, rng):
        for f in fs: f(s, rng)
    return calib
