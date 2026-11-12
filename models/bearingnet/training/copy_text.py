"""Hand-written page prose (after the exam), with numbers read from exam_r2.json / ratchet.json -> web/copy.json"""
import _paths as _P  # portable paths (see _paths.py)
import json
B = _P.WORK + '/'; E = json.load(open(B + 'exam_r2.json')); f = lambda v: '%.2f' % v
cn, pb, cf, cb = E['cwru_near'], E['pb'], E['cwru_far'], E['cwru_base']; sp, sz = E['speed'], E['size']
C = {}
C['headline'] = dict(answer='partly. On the second machine it beats the classical method; on the rig most papers use, the classical method wins',
    stats=[dict(v='%s vs %s' % (f(pb['net']['bal_acc']), f(pb['classical']['bal_acc'])), k='Paderborn rig (never seen): balanced accuracy, network vs envelope method'),
           dict(v='%s vs %s' % (f(cn['net']['bal_acc']), f(cn['classical']['bal_acc'])), k='CWRU rig, sensor at the faulty bearing: network vs envelope method'),
           dict(v='0', k='real faulty bearings used for training')])
C['kinNote'] = ('Check: our formulas give the multiples CWRU publishes for its SKF 6205 and 6203 bearings to within 0.016% '
                '(checks/kinematics_check.json). In a simulated recording the envelope lines land within 0.16% of theory, '
                'the expected 1-2% cage slip included, with inner-race side-lines at ±1× shaft and ball side-lines at ±1× cage speed (checks/physics_check.json).')
C['examText'] = ('<p>Exam data: %d CWRU recordings (all files not in dev or used as noise) read on each of their sensors, and 23 Paderborn bearings (%d recordings, 4 operating conditions). '
                 'Each reading is one sensor channel read with one bearing\'s geometry; the answer for a reading averages up to eight 2.7 s windows. '
                 'The envelope method is the textbook one (spectral-kurtosis band, squared envelope, lines at BPFO/BPFI/2·BSF), with its one threshold fitted on dev. '
                 'Balanced accuracy averages the per-class hit rates, so always guessing the commonest class scores 1/(number of classes).</p>'
                 '<p><b>On CWRU the envelope method wins</b> at every sensor (near %s vs %s). <b>On the Paderborn rig, which the network never heard, the network wins</b> (%s vs %s): '
                 'the envelope method\'s threshold, set on CWRU-like dev data, does not carry over, while the network\'s comb detector mostly does. '
                 'Neither is good at saying "healthy": the network called the healthy bearing at the other end of a faulty CWRU machine healthy in only %d%% of %d readings (envelope: %d%%).</p>'
                 % (150, pb['n'], f(cn['net']['bal_acc']), f(cn['classical']['bal_acc']), f(pb['net']['bal_acc']), f(pb['classical']['bal_acc']),
                    round(100 * E['other_bearing']['net_says_healthy']), E['other_bearing']['n'], round(100 * E['other_bearing']['classical_says_healthy']))) + '<p class="note">In the table, Network, Envelope and Majority are balanced accuracy; the last two columns are plain accuracy.</p>'
C['examMore'] = ('<h3>Shaft speed and error bars</h3><p>The network is told only a rough speed (CWRU: "about 1750 rpm" for every file, %.1f%% off on average). '
                 'It refines it to %.2f%% mean error on CWRU (envelope method\'s comb search: %.2f%%). Its 1σ bars, widened by a factor %.1f fitted on dev, cover %d%% of CWRU readings at 1σ and %d%% at 2σ. '
                 'On Paderborn the nominal speed is already right to %.2f%%; the network makes it worse (%.2f%%), and its bars cover %d%% / %d%%.</p>'
                 '<h3>Defect size</h3><p>The size output failed. On CWRU its error is %.2f dex, worse than always guessing the median size (%.2f dex); on Paderborn %.2f vs %.2f dex. Do not use it.</p>'
                 '<h3>Other cuts</h3><p>CWRU outer-race defects at 6 o\'clock (in the load zone) are found %d%% of the time, at 3 o\'clock %d%%, at 12 o\'clock %d%%. '
                 'Ball defects are the hardest (CWRU near sensor: %d%% right; the envelope method %d%%). Paderborn bearings with real, run-in damage are read better (%d%%) than ones with machined (artificial) damage (%d%%). '
                 'Bearings with both outer and inner damage (KB23/24/27, not scored above) are called outer race %d%% of the time.</p>'
                 % (sp['cwru']['nominal_mae_pct'], sp['cwru']['net_mae_pct'], sp['cwru']['classical_mae_pct'], E['calib']['speed'], round(100 * sp['cwru']['cover_1sig']), round(100 * sp['cwru']['cover_2sig']),
                    sp['pb']['nominal_mae_pct'], sp['pb']['net_mae_pct'], round(100 * sp['pb']['cover_1sig']), round(100 * sp['pb']['cover_2sig']),
                    sz['cwru_near']['mae_dex'], sz['cwru_near']['median_guess_mae_dex'], sz['pb']['mae_dex'], sz['pb']['median_guess_mae_dex'],
                    round(100 * E['by_pos']['6']['acc']), round(100 * E['by_pos']['3']['acc']), round(100 * E['by_pos']['12']['acc']),
                    round(100 * cn['net']['per_class']['B']), round(100 * cn['classical']['per_class']['B']),
                    round(100 * E['pb_by_kind']['real damage']['acc']), round(100 * E['pb_by_kind']['artificial']['acc']), round(100 * E['pb_combined']['net']['OR'])))
C['steps'] = [
 '<b>Kinematics.</b> Defect click rates from ball count, ball and pitch diameter and contact angle (BPFO, BPFI, BSF, FTF). Checked against the CWRU table: largest difference 0.016%.',
 '<b>Simulator.</b> A train of impacts at the defect rate with 0-2% cage slip and 0.5-2% timing jitter; load-zone amplitude modulation (Stribeck, at shaft speed for inner race, at cage speed for balls, which strike both races alternately); each strike is a soft entry and a sharp exit; each rings 1-3 random structural resonances between 0.7 and 5.6 kHz; plus shaft lines, impact trains from somewhere else (decoys) and random knocks.',
 '<b>Real noise.</b> Every simulated defect is added into a real healthy recording: CWRU file 98 and Paderborn bearings K001-K003. These files are never in dev or exam. The background is time-stretched ±15% so its own shaft lines follow the new speed.',
 '<b>Features.</b> Envelope spectra in 5 bands (4 fixed, 1 chosen by spectral kurtosis), each read on a log-frequency axis lined up with each fault\'s click rate. On that axis a fault always makes the same comb, whatever the bearing.',
 '<b>Network.</b> A 1-D residual CNN, 438k parameters, trained on 20,000 simulations. Outputs: condition, shaft speed with an error bar, defect size with an error bar.',
 '<b>Dev calibration.</b> Twin test on the 10 CWRU dev files and 3 Paderborn dev bearings: the first network got 100% of simulated twins right and 22% of the real CWRU recordings. Real faults were 10-25 dB louder than the simulations allowed and rang longer (3.5 ms vs 2.3 ms); healthy machines showed shaft-rate modulation. Physics v2 built these in; dev balanced accuracy went from 0.31 to 0.68.',
 '<b>Ratchet.</b> Two more rungs (one shared trunk per fault hypothesis; band dropout) did not beat v2 on the frozen simulations by 0.01, so v2 (rung r2) is the release.',
 '<b>Exam.</b> Scored once, on files never used for any choice, against the envelope method and the majority guess.']
C['limits'] = ('<ul>'
 '<li><b>On the CWRU rig the classical envelope method is better</b> (balanced accuracy %s vs %s at the near sensor). Ball defects are mostly missed (%d%%) and often called outer race.</li>'
 '<li><b>"Healthy" is the weak answer.</b> A fault anywhere in the machine leaks into every sensor; the network often blames the wrong bearing (healthy bearing next to a faulty one called healthy %d%% of the time; Paderborn healthy bearings %d%%).</li>'
 '<li><b>The size output does not work</b> (worse than guessing the median). The speed refinement helps when the nominal speed is rough (CWRU), and hurts when it is already exact (Paderborn).</li>'
 '<li><b>The dev set was small</b> (10 CWRU files, 3 Paderborn bearings) and too kind: Paderborn dev balanced accuracy was 0.78, the exam 0.45.</li>'
 '<li>CWRU has only 4 healthy readings at the near sensor in the exam, so its balanced accuracy for "healthy" rests on 4 readings. Some CWRU fault files are known to be hard or ambiguous for any method (Smith & Randall 2015 benchmark).</li>'
 '<li>One machine type per dataset, small deep-groove ball bearings, constant speed. Variable speed, gears, rolling-element bearings of other types and different sensor mounts are untested.</li></ul>'
 % (f(cn['net']['bal_acc']), f(cn['classical']['bal_acc']), round(100 * cn['net']['per_class']['B']), round(100 * E['other_bearing']['net_says_healthy']), round(100 * pb['net']['per_class']['N'])))
C['credits'] = ('<b>Data.</b> Case Western Reserve University Bearing Data Center (engineering.case.edu/bearingdatacenter): 12 and 48 kHz drive-end, fan-end and normal-baseline files. '
 'Paderborn University KAt-DataCenter bearing data (C. Lessmeier, J. K. Kimotho, D. Zimmer, W. Sextro, "Condition monitoring of bearing damage in electromechanical drive systems by using motor current signals of electric motors: a benchmark data set for data-driven classification", PHM Society European Conference 2016); used under the terms on the KAt-DataCenter page. '
 'MFPT fault data was checked as a third source; the MFPT site no longer offers it for download. '
 '<b>Physics.</b> P. D. McFadden & J. D. Smith (1984), J. Sound Vib. 96; R. B. Randall & J. Antoni (2011), "Rolling element bearing diagnostics - a tutorial", MSSP 25; N. Sawalhi & R. B. Randall (2011) on entry/exit events; W. A. Smith & R. B. Randall (2015), CWRU benchmark study, MSSP 64-65. '
 '<b>Built</b> by Claude for Greg\'s physics + ML collection. Fonts: Big Shoulders Display, Public Sans, IBM Plex Mono (Google Fonts).')
json.dump(C, open(B + 'web/copy.json', 'w'), indent=1); print('ok')
