"""Readout-seed sensitivity on the frozen v2 bank and dev, after fixing the release."""
from _paths import work
from noise import RESULTS, read_result, write_json
from simbank import load_bank
from split import dev_cases
from train import RUNS, train
from ratchet import error_ratio
from stormnet import StormNet

if __name__ == "__main__":
    cal = read_result("calibration.json")
    datasets = {"val2": load_bank("val2"), **dev_cases()}
    out = {}
    for seed in (103, 104, 105):
        out[seed] = {}
        for rid, mode, banks, _ in RUNS:
            folder = "ckpt" if seed == 103 else f"seed{seed}"
            if seed != 103:
                train(rid, mode, banks, cal, seed, folder)
            net = StormNet(work(folder, rid + ".npz"))
            out[seed][rid] = {key: 1 - error_ratio(d, net.predict(d["wind"], d["dst"])["min_dst"])
                              for key, d in datasets.items()}
    write_json(RESULTS / "seed_check.json", out)
