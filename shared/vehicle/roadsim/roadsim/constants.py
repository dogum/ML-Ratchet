"""Physical constants and fuel properties used across roadsim (SI units)."""
G = 9.80665                 # m/s^2
R_AIR = 287.05              # J/(kg K), dry air
MPH = 0.44704               # m/s per mph
MILE = 1609.344             # m
GAL = 3.785411784           # L per US gallon
LBF = 4.4482216             # N per lbf
LB = 0.45359237             # kg per lb
KWH = 3.6e6                 # J per kWh
EPA_KWH_PER_GAL = 33.705    # kWh per gallon-equivalent (EPA MPGe definition)

# Fuels. Density per the RoadSim/RoadData shared contract; lower heating values from typical certification fuels
# (EPA ALPHA fuel tags: Tier 2 cert gasoline 42.9 MJ/kg, Tier 3 E10 41.95 MJ/kg; ULSD ~42.9-43.1 MJ/kg).
FUELS = {
    'gasoline': dict(lhv=42.6e6, density=745.0),   # J/kg, g/L  (between E0 and E10 cert fuels)
    'diesel': dict(lhv=42.9e6, density=832.0),
    'none': dict(lhv=42.6e6, density=745.0),
}

T0K = 273.15
