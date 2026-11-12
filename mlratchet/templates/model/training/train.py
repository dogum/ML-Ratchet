"""Train one rung on simulations only, with resumable checkpoints (the cloud container can restart).

    python train.py --name r1 --bank train1 --steps 5000
    python train.py --name r2 --bank train2 --init r1 --steps 3000 --lr 3e-4

Use any framework (the collection used PyTorch). Keep it small: one thread, ≤ ~500k parameters,
a checkpoint every ~500 steps, and a loss that learns its own error bars (β-NLL, von Mises NLL).
"""
import argparse

import _paths as _P  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--init")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    raise NotImplementedError("TODO: model, loss, loop; save ckpt to _P.work('ckpt', f'{args.name}.pt')")


if __name__ == "__main__":
    main()
