"""Physical constants and unit helpers used across WaterSim (SI unless stated)."""

SECONDS_PER_DAY = 86400.0
RHO_W = 1000.0            # kg/m3, liquid water
G = 9.80665               # m/s2
LAMBDA_F = 0.334          # MJ/kg, latent heat of fusion
LAMBDA_V = 2.501          # MJ/kg, latent heat of vaporisation at 0 C
LAMBDA_S = LAMBDA_F + LAMBDA_V  # MJ/kg, sublimation
C_ICE = 2.102e-3          # MJ/kg/K
C_WATER = 4.186e-3        # MJ/kg/K
CP_AIR = 1.013e-3         # MJ/kg/K
R_DRY = 287.05            # J/kg/K
SIGMA_MJ_DAY = 4.903e-9   # Stefan-Boltzmann, MJ/m2/K4/day (FAO-56)
VON_KARMAN = 0.41
GSC = 0.0820              # solar constant, MJ/m2/min (FAO-56)
W_TO_MJ_DAY = 0.0864      # 1 W/m2 sustained over a day = 0.0864 MJ/m2/day


def mm_day_to_m3s(q_mm_day, area_km2):
    """Streamflow depth over the basin (mm/day) -> discharge (m3/s). 1 mm/day over 1 km2 = 1000 m3/day."""
    return q_mm_day * area_km2 / 86.4


def m3s_to_mm_day(q_m3s, area_km2):
    return q_m3s * 86.4 / area_km2


def hm3_to_mm(v_hm3, area_km2):
    """Volume (hm3 = 1e6 m3) as an equivalent depth (mm) over area_km2."""
    return v_hm3 * 1000.0 / area_km2


def mm_to_hm3(d_mm, area_km2):
    return d_mm * area_km2 / 1000.0
