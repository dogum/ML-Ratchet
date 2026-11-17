"""Equipment performance curves, each with its source. All temperatures in deg C.

Biquadratic: f(x, y) = c0 + c1 x + c2 x^2 + c3 y + c4 y^2 + c5 x y (EnergyPlus Curve:Biquadratic).
Every curve is normalised to 1.0 at its rating point (checked in checks/check_curves.py).
"""

# --- DX cooling (single speed), EnergyPlus example-file curves used in the DOE prototypes' packaged units
# ("HPACCoolCapFT", "HPACCoolEIRFT", "HPACCoolCapFFF", "HPACEIRFFF", "HPACCOOLPLFFPLR"), x = entering wet bulb,
# y = outdoor dry bulb. Source: EnergyPlus DataSets / example files (e.g. 5ZoneAutoDXVAV.idf), US DOE.
DX_COOL_CAP_FT = (0.942587793, 0.009543347, 0.000683770, -0.011042676, 0.000005249, -0.000009720)
DX_COOL_EIR_FT = (0.342414409, 0.034885008, -0.000623700, 0.004977216, 0.000437951, -0.000728028)
DX_COOL_RANGE = dict(x=(12.78, 23.89), y=(18.0, 46.11))
DX_CAP_FFF = (0.8, 0.2, 0.0)          # quadratic in flow fraction
DX_EIR_FFF = (1.1552, -0.1808, 0.0256)
DX_PLF = (0.85, 0.15, 0.0)            # part-load fraction vs PLR (cycling losses), single speed
DX_PLF_TWO = (0.92, 0.08, 0.0)        # assumption: two-stage units cycle less (stage 1 = 2/3 capacity)
RATED_COOL = dict(T_wb_in=19.44, T_odb=35.0)   # AHRI 210/240 A test (80/67 F indoor, 95 F outdoor)

# --- DX heating (heat pump), EnergyPlus example curves "HPACHeatCapFT" / "HPACHeatEIRFT" (cubic in outdoor dry
# bulb, -20..20 C), "HPACHeatPLFFPLR" = cooling PLF. Rated at 8.33 C outdoor (AHRI 47 F).
HP_HEAT_CAP_FT = (0.758746, 0.027626, 0.000148716, 0.0000034992)
HP_HEAT_EIR_FT = (1.19248, -0.0300438, 0.00103745, -0.000023328)
HP_HEAT_RANGE = (-20.0, 20.0)
RATED_HEAT_T = 8.33

# --- Electric EIR chiller, EnergyPlus DataSets/Chillers.idf "DOE-2 Centrifugal/5.50COP" curves (from DOE-2.1E):
# x = leaving chilled-water temperature, y = entering condenser fluid temperature. Rated 6.67 / 29.44 C.
CH_CAP_FT = (0.257896, 0.0389016, -0.00021708, 0.0468684, -0.00094284, -0.00034344)
CH_EIR_FT = (0.933884, -0.058212, 0.00450036, 0.00243, 0.000486, -0.001215)
CH_EIR_FPLR = (0.222903, 0.313387, 0.46371)   # quadratic in PLR
CH_RATED = dict(T_chw=6.67, T_cond_water=29.44, T_cond_air=35.0)
CH_MIN_PLR = 0.15

# --- VAV fan part-load power, ASHRAE 90.1 Appendix G (G3.1.3.15, "VSD with static pressure reset" uses the
# 90.1 User's Manual curve): P/P_design = 0.0013 + 0.1470 x + 0.9506 x^2 - 0.0998 x^3, x = flow fraction.
VAV_FAN = (0.0013, 0.1470, 0.9506, -0.0998)
VAV_FAN_NO_RESET = (0.0408, 0.0880, -0.0729, 0.9437)   # constant static (EnergyPlus prototype "VSD fan" curve)

# --- Variable-speed pumps, 90.1 Appendix G / EnergyPlus prototype pump curve: P/P_d = 0 + 0.0205 x + 0.4101 x^2
# + 0.5753 x^3.
VS_PUMP = (0.0, 0.0205, 0.4101, 0.5753)

# --- Boilers. Non-condensing: EnergyPlus example "BoilerEfficiency" quadratic in PLR (normalised).
BOILER_PLR = (0.97, 0.0633, -0.0333)


def condensing_eff_factor(T_return):
    """Condensing boiler efficiency vs return-water temperature, linearised from typical published curves
    (ASHRAE HVAC Systems & Equipment 2016 ch. 32): ~0.98 at 25 C return, ~0.92 at 50 C, ~0.87 above the
    ~55 C flue dew point. Returns absolute efficiency (HHV)."""
    if T_return <= 25.0:
        return 0.98
    if T_return <= 55.0:
        return 0.98 - (T_return - 25.0) * 0.0035
    return max(0.86, 0.875 - (T_return - 55.0) * 0.0005)


def biquad(c, x, y):
    return c[0] + c[1] * x + c[2] * x * x + c[3] * y + c[4] * y * y + c[5] * x * y


def cubic(c, x):
    return c[0] + x * (c[1] + x * (c[2] + x * c[3]))


def quad(c, x):
    return c[0] + x * (c[1] + x * c[2])


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def dx_cool_capft(twb, todb):
    r = DX_COOL_RANGE
    return biquad(DX_COOL_CAP_FT, clamp(twb, *r['x']), clamp(todb, *r['y']))


def dx_cool_eirft(twb, todb):
    r = DX_COOL_RANGE
    return biquad(DX_COOL_EIR_FT, clamp(twb, *r['x']), clamp(todb, *r['y']))


def hp_heat_capft(todb):
    return cubic(HP_HEAT_CAP_FT, clamp(todb, *HP_HEAT_RANGE))


def hp_heat_eirft(todb):
    return cubic(HP_HEAT_EIR_FT, clamp(todb, *HP_HEAT_RANGE))


def ch_capft(tchw, tcond):
    return biquad(CH_CAP_FT, clamp(tchw, 4.0, 12.0), clamp(tcond, 15.0, 40.0))


def ch_eirft(tchw, tcond):
    return biquad(CH_EIR_FT, clamp(tchw, 4.0, 12.0), clamp(tcond, 15.0, 40.0))


# Shape modifiers for equipment types without a published curve of their own (documented assumptions):
def cold_climate_capft(todb):
    """Cold-climate / variable-speed heat pumps hold capacity better: half the slope of the single-speed curve
    (gives ~80 % of rated capacity at -8.3 C, typical of NEEP cold-climate listed units)."""
    return 1.0 + 0.5 * (hp_heat_capft(todb) - hp_heat_capft(RATED_HEAT_T))


def gshp_heat(ewt):
    """Water-to-air heat pump heating capacity and EIR multipliers vs entering water temperature, rated at 0 C
    (ISO 13256-1 ground-loop rating). Slopes typical of manufacturer catalogues: capacity +2.2 %/K, EIR -1.8 %/K."""
    return 1.0 + 0.022 * (ewt - 0.0), max(0.5, 1.0 - 0.018 * (ewt - 0.0))


def gshp_cool(ewt):
    """Rated at 25 C EWT (ISO 13256-1): capacity -0.8 %/K, EIR +2.0 %/K."""
    return 1.0 - 0.008 * (ewt - 25.0), max(0.5, 1.0 + 0.020 * (ewt - 25.0))


def vs_power_frac(plr):
    """Inverter-driven compressor power fraction at part load (minimum speed 30 %): P/P_full = PLR (0.75 +
    0.25 PLR) above 0.3, i.e. COP ~14 % above rated at half load. Assumption informed by AHRI/NEEP part-load data."""
    return plr * (0.75 + 0.25 * plr)
