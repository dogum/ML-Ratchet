#!/bin/bash
# The exam-mode pipeline end to end (lock, guard, workers, aggregation, EXAM_DONE) with the DEV records standing in for
# the exam (config.MOCK): no exam data is read. Results go to tmp/mock_results (deleted after); the summary is kept in
# results/mock_exam_check.json.
cd /home/claude/bakeoff/vehicle/harness && source env.sh
export BAKEOFF_MOCK_EXAM=1
rm -rf tmp/mock_results
python3 exam.py --check || exit 1
python3 exam.py --run || exit 1
python3 exam.py --run; echo "second --run exit $? (must be 3: refused, runs once)"
python3 - <<'PY'
import json, glob, os
os.environ['BAKEOFF_MOCK_EXAM'] = '1'
import config as C
done = json.load(open(os.path.join(C.RESULTS, 'EXAM_DONE.json')))
r = json.load(open(os.path.join(C.HARNESS, done['result'])))
t = r['boards']['table']['rows']
out = {'mock': True, 'run_id': r['run_id'], 'mode': r['mode'], 'arms': {k: v['unchanged_after'] for k, v in r['arms'].items()},
       'enet_v21_VED_ICE_MdAPE_on_dev': t['enet_v21']['VED_ICE']['MdAPE'], 'enet_v21_EPA_conv_FTP_on_dev': t['enet_v21']['EPA_conv_FTP']['MdAPE'],
       'reproduction_keys': sorted(r['reproduction_enet_v21'])[:5], 'reproduction_refs_keys': sorted(r.get('reproduction_refs', {}))[:5],
       'boards': list(r['boards']), 'n_sets_table': len(t['enet_v21']), 'exam_access_log_lines': sum(1 for _ in open(os.path.join(C.RESULTS, 'exam_access.log'))),
       'lock_removed': not os.path.exists(os.path.join(C.RESULTS, 'EXAM_RUNNING.json'))}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(C.HARNESS, 'results', 'mock_exam_check.json'), 'w'), indent=1)
PY
test ! -e results/EXAM_DONE.json && test ! -e real/exam && echo "real results/ and real/exam untouched"
rm -rf tmp/mock_results
