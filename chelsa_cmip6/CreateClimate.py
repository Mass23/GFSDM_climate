import glob
import subprocess
from multiprocessing import Pool
import os

import numpy as np
import pandas as pd
from netCDF4 import Dataset


def GetValues(file, gps_coords, var, months):
    nc = Dataset(file,'r')
    latitude= nc.variables['lat'][:]
    longitude = nc.variables['lon'][:]
    
    out_vals = []
    
    for i, coords in enumerate(gps_coords):
        lat = coords[0]
        lon = coords[1]
        if var.startswith('bio'):
            vals = nc.variables[var][:]
            vals = np.reshape(vals, (latitude.shape[0],longitude.shape[0]))
        elif var == 'gdd':
            continue
        else:
            vals = nc.variables[var][:]
            vals = vals[int(months[i])-1]
            vals = np.reshape(vals, (latitude.shape[0],longitude.shape[0]))

        lat_idx = (np.abs(latitude - lat)).argmin()
        lon_idx = (np.abs(longitude - lon)).argmin()
        out_vals.append(vals[lat_idx, lon_idx])
    nc.close()
    return(out_vals)

scenarios = ['ssp126','ssp245','ssp370','ssp585']
# institution_id = [0]	source_id = [1]
institutions = [['NOAA-GFDL','GFDL-ESM4'],['MOHC','UKESM1-0-LL'],['DKRZ','MPI-ESM1-2-HR'],['IPSL','IPSL-CM6A-LR'],['MRI','MRI-ESM2-0']]
bioclims = [f'bio{i}' for i in range(1,20)]
clims = ['pr','tas','tasmin','tasmax']
dates = [['2019-01-01','2021-12-31'],['2100-01-01','2100-12-31']] # ['2040-01-01','2040-12-31'],['2070-01-01','2070-12-31']

# Get GPS coordinates in list of list, and the sampling months
metadata = pd.read_csv('../nomis-20220426-1541-db.csv')

expeditions = list(set(metadata['Expedition']))

def GetData(func_in):
    expedition = func_in[0]
    inst = func_in[1]
    scen = func_in[2]
    date = func_in[3]
    os.mkdir(f'../data/{expedition}/{inst}')

    exp_meta = metadata.loc[metadata['Expedition'] == expedition,]
    gps_coords = []
    lats = []
    lons = []
    samp_months = []
    gl_ids = []
    for i,r in exp_meta.iterrows():
        gps_coords.append([r['lat_sp [DD]'], r['lon_sp [DD]']])
        lats.append(r['lat_sp [DD]'])
        lons.append(r['lon_sp [DD]'])
        samp_months.append(r['date [DD.MM.YYYY]'].split('.')[1])
        gl_ids.append(r['patch'])

    # Download the data using the chelsa_cmip6.py command
    if inst[0] == 'MOHC':
        member = 'r1i1p1f2'
    else:
        member = 'r1i1p1f1'
    cmd = f"""python3 chelsa_cmip6.py --activity_id 'ScenarioMIP' --table_id 'Amon' --experiment_id '{scen}' --institution_id '{inst[0]}' --source_id '{inst[1]}' --member_id '{member}' --refps '1981-01-01' --refpe '2010-12-31' --fefps '{date[0]}' --fefpe '{date[1]}' --xmin {np.min(lons)-1} --xmax {np.max(lons)+1} --ymin {np.min(lats)-1} --ymax {np.max(lats)+1} --output '../data/{expedition}/{inst}/'"""
    subprocess.call(cmd, shell=True)

list_for_workers = []
for exp in expeditions:
    for scen in scenarios:
        for inst in institutions:
            for date in dates:
                list_for_workers.append([exp, inst, scen, date])
                
with Pool(48) as p:
    p.map(GetData, list_for_workers)



# out_df = pd.DataFrame()
# files = glob.glob(f'../data/{expedition}/*.nc')
#                for f in files:
#                    var = f.split('_')[3]
#                    if var != 'gdd':
#                        var_data = GetValues(f, gps_coords, var, samp_months)
#
#                        var_df = pd.DataFrame({'Value':var_data,
#                                            'Glacier':gl_ids,
#                                            'Institution': ['_'.join(inst) for i in range(0,len(var_data))],
#                                            'Scenario': [scen for i in range(0,len(var_data))],
#                                            'Date': ['_'.join(date) for i in range(0,len(var_data))],
#                                            'Member': [member for i in range(0,len(var_data))],
#                                            'Var': [var for i in range(0,len(var_data))],
#                                            'Expedition': [expedition for i in range(0,len(var_data))]})
#
#                        out_df = pd.concat([out_df, var_df])
#out_df.write(f'../GFSDM_{expedition}_climate.csv')
