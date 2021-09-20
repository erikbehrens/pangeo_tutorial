#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 03:34:39 2021

@author: behrense
"""

# needed packages
import intake # access the data store
from cmip6_preprocessing.preprocessing import combined_preprocessing # does some CMIP6 preprocessing
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# set the location of the data store
col = intake.open_esm_datastore('https://storage.googleapis.com/cmip6/pangeo-cmip6.json')

#%% show the content of the data frame
col.df

#%% show the columns
col.df.columns

#%% that is what can be selected and searched for
#Index(['activity_id', 'institution_id', 'source_id', 'experiment_id',
#       'member_id', 'table_id', 'variable_id', 'grid_label', 'zstore',
#       'dcpp_init_year', 'version'],
#      dtype='object')


#%% Find all data from HighResMIP and snowfall
query=dict(activity_id='HighResMIP',variable_id='prsn')
col_subset = col.search(**query)


#%% refined search from above to just get  timestep 100
query=dict(activity_id='HighResMIP',variable_id='prsn',institution_id='MPI-M',source_id=['MPI-ESM1-2-XR'])
col_subset = col.search(**query)
#get your xarray
dsets = col_subset.to_dataset_dict(zarr_kwargs={'consolidated':True, 'decode_times':False},aggregate=True , preprocess=combined_preprocessing                                  )
ds=dsets['HighResMIP.MPI-M.MPI-ESM1-2-XR.highresSST-present.Amon.gn']
precip=ds.isel(time=100)

#%% quick plot, now the data gets sourced, that is why it take a short moment
plt.close('all')
fig = plt.figure(figsize=(12,7))
xr.plot.contourf(precip.prsn[0,:,:],levels=20)

#%% first 10 months
precip=ds.isel(time=range(10))
#%%
plt.close('all')
fig = plt.figure(figsize=(12,7))
xr.plot.contourf(np.mean(precip.prsn[0,:,:,:],axis=0),levels=20)



#%% want to compute barotropic streamfunction
query = dict(source_id=['GFDL-CM4'],experiment_id='historical',variable_id='uo')
col_subset = col.search(**query)
dsets = col_subset.to_dataset_dict(zarr_kwargs={'consolidated':True, 'decode_times':False},aggregate=True , preprocess=combined_preprocessing                                  )

ds=dsets['CMIP.NOAA-GFDL.GFDL-CM4.historical.Omon.gn']
u=ds.isel(time=range(1),y=range(0,400)) # frist 12 months
#%%
uuo=u.uo.values
lat=u.lat.values
lon=u.lon.values
#%%

def calc_distance(lo1,la1,lo2,la2):
    from math import sin, cos, sqrt, atan2, radians
    R = 6373.0

    lat1 = radians(la1)
    lon1 = radians(lo1)
    lat2 = radians(la2)
    lon2 = radians(lo2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

#%% calculate hoizontal grid

e1t=np.zeros(lat.shape)
e2t=np.zeros(lat.shape)

for i in range(lat.shape[1]-1):
    for j in range(lat.shape[0]-1):
      e1t[j,i]= calc_distance(lon[j,i],lat[j,i],lon[j,i+1],lat[j,i+1]) 
      e2t[j,i]= calc_distance(lon[j,i],lat[j,i],lon[j+1,i],lat[j+1,i]) 
      

e1t[:,-1]=e1t[:,0]
e1t[-1,:]=e1t[-2,:]
e2t[:,-1]=e2t[:,0]
e2t[-1,:]=e2t[-2,:]


e2t=e2t*1000 #convert to m


#%% generating dz
depth=u.coords['lev'].values
dz=np.diff(depth)


#%% calculate streamfunction
tmp=np.zeros((uuo.shape[1],uuo.shape[3],uuo.shape[4]))
psi=np.zeros((uuo.shape[1],uuo.shape[3],uuo.shape[4]))

for t in range(uuo.shape[1]):
    print(t)
    for j in range(e1t.shape[0]):
       
        tmp[t,j,:]=np.nansum(-uuo[0,t,:34,j,:]*np.repeat(e2t[np.newaxis,j,:],34,axis=0)*np.repeat(dz[:,np.newaxis],e1t.shape[1],axis=1),axis=0)
#%%

psi=np.nancumsum(tmp,axis=1)/1e6

#%%
import matplotlib.pyplot as plt

plt.close('all')
fig = plt.figure(figsize=(12,7))
plt.contourf(psi.mean(axis=0),levels=np.arange(0,60,10))
plt.colorbar()

