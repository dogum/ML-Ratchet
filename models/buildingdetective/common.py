"""Constants shared by the BuildingDetective inference code (release copy; no simulator imports)."""
USES = ['office', 'education', 'lodging', 'retail', 'healthcare', 'assembly', 'warehouse', 'residential_single',
        'residential_multi', 'food', 'lab', 'other']
HEAT = ['heat_pump', 'resistance', 'fuel', 'district', 'none']
COOL = ['none', 'dx', 'chiller', 'district']
REG = ['logA', 'year', 'logUA', 'logUAm2', 'logtau', 'Tb_heat', 'Tb_cool', 'logS_heat', 'logS_cool', 'base_share',
       'op_hours', 'weekend']
CHANNELS = ['elec', 'fuel', 'dheat', 'dcool']   # electricity, on-site fuel (gas/oil/propane), district heat, chilled water
