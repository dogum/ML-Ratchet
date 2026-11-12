"""BearingNet: a small 1-D residual CNN over the hypothesis-aligned envelope strips (21 channels x 295 log-frequency cells)
plus 8 scalars. Outputs: 4 class logits (N, OR, IR, B), speed (log2 f_true/f_nom: mean, log sigma), size (log10 mm: mean, log sigma)."""
import _paths as _P  # portable paths (see _paths.py)
import torch, torch.nn as nn, sys
sys.path.insert(0, _P.SRC); import features as FE
NOUT = 8; SP_SCALE = 0.05; SZ_MU, SZ_SD = -0.11, 0.53       # targets standardised: speed / 0.05 ; (log10 size - mu) / sd

class Block(nn.Module):
    def __init__(s, ci, co, k=5):
        super().__init__(); s.c1 = nn.Conv1d(ci, co, k, 2, k // 2); s.b1 = nn.BatchNorm1d(co); s.c2 = nn.Conv1d(co, co, k, 1, k // 2); s.b2 = nn.BatchNorm1d(co)
        s.sk = nn.Conv1d(ci, co, 1, 2); s.bs = nn.BatchNorm1d(co); s.a = nn.GELU()
    def forward(s, x): return s.a(s.b2(s.c2(s.a(s.b1(s.c1(x))))) + s.bs(s.sk(x)))

class BearingNet(nn.Module):
    def __init__(s, ch=(40, 56, 80, 112, 128), hid=160):
        super().__init__(); s.stem = nn.Conv1d(FE.NCH, ch[0], 7, 2, 3); s.bst = nn.BatchNorm1d(ch[0]); s.a = nn.GELU()
        s.body = nn.ModuleList([Block(ch[i], ch[i + 1]) for i in range(len(ch) - 1)])
        s.head = nn.Sequential(nn.Linear(2 * ch[-1] + FE.NSC, hid), nn.GELU(), nn.Linear(hid, hid // 2), nn.GELU(), nn.Linear(hid // 2, NOUT))
    def forward(s, x, sc):
        h = s.a(s.bst(s.stem(x)))
        for b in s.body: h = b(h)
        return s.head(torch.cat([h.mean(-1), h.amax(-1), sc], 1))

class HypNet(nn.Module):
    """Hypothesis-shared network (rung r3). ONE trunk reads the 5 demodulation-band strips of each hypothesis axis
    (shaft, BPFO, BPFI, BSF) with the same weights, so a fault class can only win by showing a comb at ITS OWN
    defect frequency; a small learned code per hypothesis lets the head read class-specific sidebands.
    Healthy logit, speed and size come from all trunk outputs + the raw-spectrum strip + scalars."""
    def __init__(s, ch=(16, 24, 48, 64, 96), emb=4, hid=96):
        super().__init__(); s.nh = len(FE.HYP); s.nb = FE.NB
        s.stem = nn.Conv1d(s.nb, ch[0], 7, 2, 3); s.bst = nn.BatchNorm1d(ch[0]); s.a = nn.GELU()
        s.body = nn.ModuleList([Block(ch[i], ch[i + 1]) for i in range(len(ch) - 1)])
        s.rstem = nn.Conv1d(1, 16, 7, 2, 3); s.rbst = nn.BatchNorm1d(16); s.rbody = nn.ModuleList([Block(16, 24), Block(24, 32)])
        g = 2 * ch[-1]; s.emb = nn.Parameter(torch.zeros(3, emb)); nn.init.normal_(s.emb, 0, 0.5)
        s.fault = nn.Sequential(nn.Linear(2 * g + emb + FE.NSC, hid), nn.GELU(), nn.Linear(hid, hid // 2), nn.GELU(), nn.Linear(hid // 2, 1))
        s.rest = nn.Sequential(nn.Linear(4 * g + 64 + FE.NSC, hid * 2), nn.GELU(), nn.Linear(hid * 2, hid), nn.GELU(), nn.Linear(hid, 5))
    def trunk(s, x):
        h = s.a(s.bst(s.stem(x)))
        for b in s.body: h = b(h)
        return torch.cat([h.mean(-1), h.amax(-1)], 1)
    def forward(s, x, sc):
        B = x.shape[0]; e = x[:, :s.nb * s.nh].reshape(B, s.nb, s.nh, -1).transpose(1, 2).reshape(B * s.nh, s.nb, -1)
        g = s.trunk(e).reshape(B, s.nh, -1)                                       # (B, 4 hypotheses, g)
        r = s.a(s.rbst(s.rstem(x[:, -1:])))
        for b in s.rbody: r = b(r)
        r = torch.cat([r.mean(-1), r.amax(-1)], 1)
        lf = torch.cat([s.fault(torch.cat([g[:, k], g[:, 0], s.emb[k - 1].expand(B, -1), sc], 1)) for k in (1, 2, 3)], 1)   # OR, IR, B
        o = s.rest(torch.cat([g.reshape(B, -1), r, sc], 1))                         # N logit, speed mu, log sig, size mu, log sig
        return torch.cat([o[:, :1], lf, o[:, 1:]], 1)

ARCH = {'base': BearingNet, 'hyp': HypNet}

if __name__ == '__main__':
    n = HypNet(); print(sum(p.numel() for p in n.parameters()), n(torch.zeros(2, FE.NCH, len(FE.V)), torch.zeros(2, FE.NSC)).shape)
    n = BearingNet(); print(sum(p.numel() for p in n.parameters()), n(torch.zeros(2, FE.NCH, len(FE.V)), torch.zeros(2, FE.NSC)).shape)
