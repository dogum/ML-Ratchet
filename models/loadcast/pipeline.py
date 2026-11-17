"""LoadCast's input pipeline, the same for simulated and real meters (see loadcast.py, which holds the code):
weather -> hourly features (weather_features), local calendar -> calendar_features (holidays for US/CA/UK/IE),
metadata -> static_features, own meter history -> history_encode / history_hourly (ridge TOWT, change point,
time-of-week profile, scalars). Re-exported here for convenience."""
from loadcast import (weather_features, calendar_features, static_features, history_encode, history_hourly,  # noqa: F401
                      annual_climate, is_holiday, towt_fit, towt_predict, cp_fit, g14, LoadCast)
