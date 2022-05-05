#!/usr/bin/env python

#This file is part of chelsa_cmip6.
#
#chelsa_cmip6 is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#chelsa_cmip6 is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with chelsa_cmip6.  If not, see <https://www.gnu.org/licenses/>.


import requests
import xml.etree.ElementTree as ET
import numpy as np
import xarray as xr
import rasterio
import pandas as pd
import zarr
import gcsfs
import datetime
from classes.BioClim import BioClim
from classes.BioClim import growing_degree_days

def _esgf_search(server="https://esgf-node.llnl.gov/esg-search/search",
                  files_type="OPENDAP", local_node=True, project="CMIP6",
                  verbose=False, format="application%2Fsolr%2Bjson",
                  use_csrf=False, **search):
    client = requests.session()
    payload = search
    payload["project"] = project
    payload["type"] = "File"
    if local_node:
        payload["distrib"] = "false"
    if use_csrf:
        client.get(server)
        if 'csrftoken' in client.cookies:
            # Django 1.6 and up
            csrftoken = client.cookies['csrftoken']
        else:
            # older versions
            csrftoken = client.cookies['csrf']
        payload["csrfmiddlewaretoken"] = csrftoken
    payload["format"] = format
    offset = 0
    numFound = 10000
    all_files = []
    files_type = files_type.upper()
    while offset < numFound:
        payload["offset"] = offset
        url_keys = []
        for k in payload:
            url_keys += ["{}={}".format(k, payload[k])]

        url = "{}/?{}".format(server, "&".join(url_keys))
        print(url)
        r = client.get(url)
        r.raise_for_status()
        resp = r.json()["response"]
        numFound = int(resp["numFound"])
        resp = resp["docs"]
        offset += len(resp)
        for d in resp:
            if verbose:
                for k in d:
                    print("{}: {}".format(k, d[k]))
            url = d["url"]
            for f in d["url"]:
                sp = f.split("|")
                if sp[-1] == files_type:
                    all_files.append(sp[0].split(".html")[0])
    return sorted(all_files)


def _get_cmip(activity_id, table_id, variable_id, experiment_id, instituion_id, source_id, member_id):
    """Get CMIP model from Google"""
    gcs = gcsfs.GCSFileSystem(token='anon')
    df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
    search_string = "activity_id == '" + activity_id + "' & table_id == '" + table_id + "' & variable_id == '" + variable_id + "' & experiment_id == '" + experiment_id + "' & institution_id == '" + instituion_id + "' & source_id == '" + source_id + "' & member_id == '" + member_id + "'"
    df_ta = df.query(search_string)
    # get the path to a specific zarr store (the first one from the dataframe above)
    zstore = df_ta.zstore.values[-1]
    # create a mutable-mapping-style interface to the store
    mapper = gcs.get_mapper(zstore)
    # open it using xarray and zarr
    ds = xr.open_zarr(mapper, consolidated=True)
    try:
        ds['time'] = np.sort(ds['time'].values)
    except Exception:
        pass

    return ds


class interpol:
    """Interpolation class"""
    def __init__(self, ds, template):
        """ Create a set of baseline clims """
        self.ds = ds
        self.template = template

    def interpolate(self):
        res = self.ds.interp(lat=self.template["y"], lon=self.template["x"])
        return res


class chelsaV2:
    """ get and clip CHELSA climatologies """
    def __init__(self, xmin, xmax, ymin, ymax, variable_id):
        """ Create a set of baseline clims """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.variable_id = variable_id

    def _crop_ds_(self, ds):
        """clip xarray"""
        mask_lon = (ds.x >= self.xmin) & (ds.x <= self.xmax)
        mask_lat = (ds.y >= self.ymin) & (ds.y <= self.ymax)
        cropped_ds = ds.where(mask_lon & mask_lat, drop=True)
        return cropped_ds

    def get_chelsa(self):
        """download chelsa"""
        a = []
        for month in range(1, 13):
            url = 'https://envicloud.os.zhdk.cloud.switch.ch/chelsa/chelsa_V2/GLOBAL/climatologies/1981-2010/' + self.variable_id + '/CHELSA_' + self.variable_id + '_' + '%02d' % (
                month,) + '_1981-2010_V.2.1.tif'
            a.append(url)

        ds = self._crop_ds_(xr.concat([xr.open_rasterio(i) for i in a], 'time'))
        if self.variable_id == "tas" or self.variable_id == 'tasmin' or self.variable_id == 'tasmax':
            res = ds / 10 - 273.15
        if self.variable_id == 'pr':
            res = ds / 10
        return res


class cmip6_clim:
    """ climatology class for monthly climatologies """
    def __init__(self, activity_id, table_id,
                 variable_id, experiment_id,
                 institution_id, source_id,
                 member_id, ref_startdate,
                 ref_enddate, fut_startdate,
                 fut_enddate):
        """ Create a set of baseline clims """
        self.activity_id = activity_id
        self.table_id = table_id
        self.variable_id = variable_id
        self.experiment_id = experiment_id
        self.institution_id = institution_id
        self.source_id = source_id
        self.member_id = member_id
        self.refps = ref_startdate
        self.refpe = ref_enddate
        self.fefps = fut_startdate
        self.fefpe = fut_enddate
        self.future_period = _get_cmip(self.activity_id, self.table_id, self.variable_id, self.experiment_id, self.institution_id, self.source_id, self.member_id).sel(time=slice(self.fefps, self.fefpe)).groupby("time.month").mean("time")
        print("future data loaded... ")
        self.historical_period = _get_cmip('CMIP', self.table_id, self.variable_id, 'historical', self.institution_id, self.source_id, self.member_id).sel(time=slice(self.refps, self.refpe)).groupby("time.month").mean("time")
        print("historical period set... ")
        self.reference_period = _get_cmip('CMIP', self.table_id, self.variable_id, 'historical', self.institution_id, self.source_id, self.member_id).sel(time=slice('1981-01-15', '2010-12-15')).groupby("time.month").mean("time")
        print("reference period set... done")

    def get_anomaly(self, period):
        """Get climatological anomaly"""
        if period == 'futr':
            if self.variable_id == "tas" or self.variable_id == 'tasmin' or self.variable_id == 'tasmax':
                res = self.future_period - self.reference_period # additive anomaly
            if self.variable_id == 'pr':
                res = (self.future_period * 86400 + 0.01) / (self.reference_period * 86400 + 0.01)   # multiplicative anomaly

        if period == 'hist':
            if self.variable_id == "tas" or self.variable_id == 'tasmin' or self.variable_id == 'tasmax':
                res = self.historical_period - self.reference_period # additive anomaly
            if self.variable_id == 'pr':
                res = (self.historical_period * 86400 + 0.01) / (self.reference_period * 86400 + 0.01)   # multiplicative anomaly

        res1 = res.assign_coords({"lon": (((res.lon) % 360) - 180)})
        return res1


class ChelsaClimat:
    """chelsa class"""
    def __init__(self, xmin, xmax, ymin, ymax):
        """ Create a set of baseline clims """
        for var in ['pr', 'tas', 'tasmax', 'tasmin']:
            setattr(self, var, chelsaV2(xmin, xmax, ymin, ymax, var).get_chelsa())


class CmipClimat:
    """ climatology class for monthly cmip 6 climatologies """

    def __init__(self, activity_id, table_id,
                 experiment_id,
                 institution_id, source_id,
                 member_id, ref_startdate,
                 ref_enddate, fut_startdate,
                 fut_enddate):
        """ Create a set of baseline clims """
        for var in ['pr', 'tas', 'tasmax', 'tasmin']:
            setattr(self, var, cmip6_clim(activity_id, table_id,
                             var, experiment_id,
                             institution_id, source_id,
                             member_id, ref_startdate,
                             ref_enddate, fut_startdate,
                             fut_enddate))


class DeltaChangeClim:
    """Delta change class"""
    def __init__(self, ChelsaClimat, CmipClimat, refps, refpe, fefps, fefpe, output=False):
        """ Create a set of baseline clims """
        self.output = output
        self.refps = refps
        self.refpe = refpe
        self.fefps = fefps
        self.fefpe = fefpe
        self.hist_year = np.mean([int(datetime.datetime.strptime(refps, '%Y-%m-%d').year), int(datetime.datetime.strptime(refpe, '%Y-%m-%d').year)]).__round__()
        self.futr_year = np.mean([int(datetime.datetime.strptime(fefps, '%Y-%m-%d').year), int(datetime.datetime.strptime(fefpe, '%Y-%m-%d').year)]).__round__()

        for per in ['futr', 'hist']:
            setattr(self, str(per + '_pr'), getattr(ChelsaClimat, 'pr').to_dataset(name='pr').rename({'time': 'month'}).drop('band') * interpol(
                    getattr(CmipClimat, 'pr').get_anomaly(per), getattr(ChelsaClimat, 'pr')).interpolate())
            for var in ['tas', 'tasmax', 'tasmin']:
                setattr(self, str(per + '_' + var), getattr(ChelsaClimat, var).to_dataset(name=var).rename({'time': 'month'}).drop('band') + interpol(
                        getattr(CmipClimat, var).get_anomaly(per), getattr(ChelsaClimat, var)).interpolate())

        for var in ['tas', 'tasmax', 'tasmin', 'pr']:
            getattr(self, str('futr_' + var))['month'] = [datetime.datetime(self.futr_year, month, 15) for month in getattr(self, str(per + '_' + var))['month'].values]
            getattr(self, str('hist_' + var))['month'] = [datetime.datetime(self.hist_year, month, 15) for month in getattr(self, str(per + '_' + var))['month'].values]

        if output:
            print('saving files to :' + output)
            for var in ['hist_tas', 'hist_tasmax', 'hist_tasmin',
                        'hist_pr']:
                getattr(self, var).to_netcdf(self.output
                                             + 'CHELSA_'
                                             + CmipClimat.tas.institution_id
                                             + '_' + CmipClimat.tas.source_id
                                             + '_' + var.replace('hist_', '')
                                             + '_' + CmipClimat.tas.experiment_id
                                             + '_' + CmipClimat.tas.member_id
                                             + '_' + CmipClimat.tas.refps
                                             + '_' + CmipClimat.tas.refpe
                                             + '.nc')
            for var in ['futr_tas', 'futr_tasmax',
                        'futr_tasmin', 'futr_pr']:
                getattr(self, var).to_netcdf(self.output
                                             + 'CHELSA_'
                                             + CmipClimat.tas.institution_id
                                             + '_' + CmipClimat.tas.source_id
                                             + '_' + var.replace('futr_', '')
                                             + '_' + CmipClimat.tas.experiment_id
                                             + '_' + CmipClimat.tas.member_id
                                             + '_' + CmipClimat.tas.fefps
                                             + '_' + CmipClimat.tas.fefpe
                                             + '.nc')



def chelsa_cmip6(source_id, institution_id, table_id, activity_id, experiment_id, member_id, refps, refpe, fefps, fefpe, xmin, xmax, ymin, ymax, output):
    print('starting downloading CMIP data:')
    cm_climat = CmipClimat(activity_id, table_id,
                           experiment_id,
                           institution_id, source_id,
                           member_id, refps,
                           refpe, fefps,
                           fefpe)

    print('starting downloading CHELSA data:')
    ch_climat = ChelsaClimat(xmin, xmax, ymin, ymax)

    dc = DeltaChangeClim(ch_climat, cm_climat, refps,
                         refpe, fefps,
                         fefpe, output)

    print('starting building climatologies data:')
    biohist = BioClim(dc.hist_pr, dc.hist_tas, dc.hist_tasmax, dc.hist_tasmin)
    biofutr = BioClim(dc.futr_pr, dc.futr_tas, dc.futr_tasmax, dc.futr_tasmin)

    print('saving bioclims:')
    for n in range(1, 20):
        name = output + 'CHELSA' + '_' + cm_climat.tas.institution_id + '_' \
               + cm_climat.tas.source_id + '_' + str('bio' + str(n)) + '_' \
               + cm_climat.tas.experiment_id + '_' + cm_climat.tas.member_id \
               + '_' + cm_climat.tas.refps + '_' + cm_climat.tas.refpe + '.nc'
        getattr(biohist, 'bio' + str(n))().to_netcdf(name)

    for n in ['gdd']:
        name = output + 'CHELSA' + '_' + cm_climat.tas.institution_id + '_' \
               + cm_climat.tas.source_id + '_'  + str(n) + '_' \
               + cm_climat.tas.experiment_id + '_' + cm_climat.tas.member_id \
               + '_' + cm_climat.tas.refps + '_' + cm_climat.tas.refpe + '.nc'
        getattr(biohist, str(n))().to_netcdf(name)

    for n in range(1, 20):
        name = output + 'CHELSA' + '_' + cm_climat.tas.institution_id + '_' \
               + cm_climat.tas.source_id + '_' + str('bio' + str(n)) + '_' \
               + cm_climat.tas.experiment_id + '_' + cm_climat.tas.member_id \
               + '_' + cm_climat.tas.fefps + '_' + cm_climat.tas.fefpe + '.nc'
        getattr(biofutr, 'bio' + str(n))().to_netcdf(name)

    for n in ['gdd']:
        name = output + 'CHELSA' + '_' + cm_climat.tas.institution_id + '_' \
               + cm_climat.tas.source_id + '_' + str(n) + '_' \
               + cm_climat.tas.experiment_id + '_' + cm_climat.tas.member_id \
               + '_' + cm_climat.tas.fefps + '_' + cm_climat.tas.fefpe + '.nc'
        getattr(biofutr, str(n))().to_netcdf(name)










