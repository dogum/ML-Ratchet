"""A region = hourly arrays over a span of hours (real BA or simulated), and the sample builder shared by sim and real.

Region attributes (arrays over the span, local index 0 = global HOURS index `off`):
  cal       daytype, dst, doy (civil calendar)
  wx        T [C], q [g/kg], RH [%], ghi, ghi_clr [W/m2], btm_cf  (population-weighted)
  W         per-point arrays (pT, pRH, pws10, pws50, pps, pghi, pcf_track, pcf_fixed)
  pdesc     building-stock params used for the PRIOR (descriptor); pact = prior components with actual weather
  yd, yw, ys  observed demand / wind / solar MW (NaN = missing)
  wg, wcap, wk  wind groups, capacity [G, n], scale (dev-calibrated k); sg, scap, sk for solar
"""
import numpy as np, pandas as pd
import buildings as B, renew as R, features as F

WARM = 720


class Region:
    def prior_demand(self, a, b, wx=None):
        """Demand prior components over local [a, b) (warm-up from a - WARM)."""
        a0 = max(0, a - WARM)
        wx = self.wx if wx is None else wx
        sl = slice(a0, b)
        cal = {k: v[sl] for k, v in self.cal.items()}
        comp = B.demand({k: wx[k][sl] for k in ('T', 'q', 'ghi', 'btm_cf')}, cal, self.pdesc,
                        t_ref=self.pdesc.get('t_ref', 2020.0))
        return {k: v[a - a0:] for k, v in comp.items()}

    def wind(self, a, b, vmult=None):
        sub = {k: self.W[k] for k in ('pws10', 'pws50', 'pT', 'pps', 'pRH')}
        return self.wk * R.wind_mw(sub, self.wg, self.wcap, slice(a, b), vmult=vmult)

    def solar(self, a, b, cfmult=None):
        sub = {k: self.W[k] for k in ('pcf_track', 'pcf_fixed')}
        return self.sk * R.solar_mw(sub, self.sg, self.scap, slice(a, b), cfmult=cfmult)

    def wind_ws(self, a, b, vmult=None):
        capg = self.wcap[:, a]
        if capg.sum() <= 0: return np.zeros(b - a)
        v = self.W['pws50'][self.wg[:, 0].astype(int), a:b] * (1 if vmult is None else vmult[self.wg[:, 0].astype(int)])
        return (capg[:, None] * v).sum(0) / capg.sum()

    def finalize(self):
        """Prior with actual weather over the whole span + wind/solar priors."""
        self.pact = B.demand({k: self.wx[k] for k in ('T', 'q', 'ghi', 'btm_cf')}, self.cal, self.pdesc,
                             t_ref=self.pdesc.get('t_ref', 2020.0))
        n = len(self.wx['T'])
        self.wprior = self.wind(0, n) if len(self.wg) else np.zeros(n)
        self.sprior = self.solar(0, n) if len(self.sg) else np.zeros(n)
        self.wcap_tot = self.wcap.sum(0) if len(self.wg) else np.zeros(n)
        self.scap_tot = self.scap.sum(0) if len(self.sg) else np.zeros(n)

    def sample(self, i0, fc=None):
        """Features and targets for target day starting at local index i0. fc: None (actual weather) or dict of
        forecast errors over the 38 h [issue, end of D): T (K, add), rh (relative, mult-1), ghi (mult, BA-level [38]
        or per point [n_pts, 38]), v (mult per point [n_pts, 38])."""
        ih, iss, ie = i0 - 182, i0 - 14, i0 + 24
        n_pts = self.W['pws50'].shape[0]
        # ---- demand
        ph = {k: v[ih:iss] for k, v in self.pact.items()}
        ph['T'] = self.wx['T'][ih:iss]
        if fc is None:
            pt = {k: v[i0:ie] for k, v in self.pact.items()}
            wt = {k: self.wx[k][i0:ie] for k in ('T', 'q', 'ghi')}
        else:
            a0 = max(0, iss - WARM)
            wxf = {k: self.wx[k][a0:ie].copy() for k in ('T', 'q', 'ghi', 'btm_cf', 'ghi_clr')}
            j = iss - a0
            gm = fc['ghi'] if np.ndim(fc['ghi']) == 1 else np.mean(fc['ghi'], 0)
            wxf['T'][j:] += fc['T']
            wxf['q'][j:] *= np.exp(0.065 * fc['T']) * (1 + fc['rh'])
            g = wxf['ghi'][j:] * gm
            wxf['ghi'][j:] = np.minimum(g, np.maximum(wxf['ghi_clr'][j:] * 1.05, wxf['ghi'][j:]))
            wxf['btm_cf'][j:] = np.clip(wxf['btm_cf'][j:] * gm, 0, 1)
            full = wxf
            tmp = Region(); tmp.cal = {k: v[a0:ie] for k, v in self.cal.items()}; tmp.wx = full; tmp.pdesc = self.pdesc
            comp = tmp.prior_demand(j, len(full['T']))
            pt = {k: v[14:] for k, v in comp.items()}
            wt = {k: full[k][j + 14:] for k in ('T', 'q', 'ghi')}
        cal = dict(daytype=self.cal['daytype'][i0 + 12], dst=self.cal['dst'][i0 + 12], doy=self.cal['doy'][i0 + 12])
        xd = F.demand_x(self.yd[ih:iss], ph, pt, wt, cal, self.desc, 0.0 if fc is None else 1.0)
        yd = F.demand_y(self.yd[i0:ie], pt['total'])
        out = dict(xd=xd, yd=yd, pd_total=pt['total'].astype(np.float32), d_true=self.yd[i0:ie].astype(np.float32),
                   comp=np.stack([pt[c] for c in F.COMP]).astype(np.float32))
        # ---- wind
        doy = cal['doy']
        if len(self.wg) and self.wcap_tot[iss] > 0:
            vm = None if fc is None else fc['v'][:, 14:]
            base = np.ones((n_pts, 24)) if vm is None else vm
            pt_w = self.wind(i0, ie, base); lo = self.wind(i0, ie, base * 0.9); hi = self.wind(i0, ie, base * 1.1)
            ws = self.wind_ws(i0, ie, base)
            cw = self.wcap_tot[iss]
            out['xw'] = F.wind_x(self.yw[ih:iss], self.wprior[ih:iss], pt_w, lo, hi, ws, cw, doy, 0.0 if fc is None else 1.0)
            out['yw'] = (self.yw[i0:ie] / cw).astype(np.float32); out['pw'] = (pt_w / cw).astype(np.float32); out['capw'] = cw
        # ---- solar
        if len(self.sg) and self.scap_tot[iss] > 0 and self.has_solar:
            if fc is None:
                cm = np.ones((n_pts, 24))
            else:
                cm = fc['ghi'][:, 14:] if np.ndim(fc['ghi']) == 2 else np.repeat(fc['ghi'][None, 14:], n_pts, 0)
            pt_s = self.solar(i0, ie, cm); lo = self.solar(i0, ie, cm * 0.8); hi = self.solar(i0, ie, cm * 1.2)
            clr = self.wx['ghi_clr'][i0:ie] / 1000.0
            gm = 1.0 if fc is None else (fc['ghi'][14:] if np.ndim(fc['ghi']) == 1 else np.mean(fc['ghi'][:, 14:], 0))
            kt = np.where(clr > 0.02, np.clip(self.wx['ghi'][i0:ie] * gm / 1000.0 / np.maximum(clr, 1e-3), 0, 1.2), 0)
            cs = self.scap_tot[iss]
            out['xs'] = F.solar_x(self.ys[ih:iss], self.sprior[ih:iss], pt_s, lo, hi, clr, kt, cs, doy, 0.0 if fc is None else 1.0)
            out['ys'] = (self.ys[i0:ie] / cs).astype(np.float32); out['ps'] = (pt_s / cs).astype(np.float32); out['caps'] = cs
            out['night'] = (clr <= 0.0).astype(np.float32)
        return out


def calendar_arrays(times, tz, std_h):
    c = B.calendar(times, tz)
    loc = pd.DatetimeIndex(times).tz_convert(tz)
    off = (loc.tz_localize(None) - pd.DatetimeIndex(times).tz_localize(None)) / pd.Timedelta('1h')
    c['dst'] = (np.asarray(off) > std_h + 0.5).astype(float)
    return c
