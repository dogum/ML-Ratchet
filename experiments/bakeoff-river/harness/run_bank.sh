#!/bin/bash
# builds val then train then the manifest; resumable (re-run after a restart)
cd /home/claude/bakeoff/river/harness
source env.sh
nice -n 5 python3 build_bank.py val && nice -n 5 python3 build_bank.py train && python3 build_bank.py finalize
