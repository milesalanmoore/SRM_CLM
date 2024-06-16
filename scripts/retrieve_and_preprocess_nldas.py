"""
This script reads in the nldas forcing data and extracts PRECIP and TPQWL data, aggregating
to a user specified frequency, monthly by default.
"""
import os
import pandas as pd
import numpy as np
import xarray as xr
from functools import partial
from distributed import wait


## Pathing ### ----------------
cesm_forcing_dir = "/glade/campaign/cesm/cesmdata/inputdata/"
nldas_dir = os.path.join(cesm_forcing_dir,
                         "atm/datm7/atm_forcing.datm7.NLDAS2.0.125d.v1")

ppt_dir = os.path.join(nldas_dir, "Precip")
ppt_fns = [os.path.join(ppt_dir, fn) for fn in os.listdir(ppt_dir)]
ppt_fns.sort()

tp_dir = os.path.join(nldas_dir, "TPQWL")
tp_fns = [os.path.join(tp_dir, fn) for fn in os.listdir(tp_dir)]
tp_fns.sort()

path_to_samples = '../data/tundra_cells.csv'

# Data output directory
output_dir = "/glade/u/home/milesmoore/SRM_CLM/data/nldas_processed/"

### Settings ### ---------------
extract_samples = False  # Should this script read in the tundra points and sample precip data?

if clip_and_resample_nldas:
lon_bnds, lat_bnds = (-112, -102), (33, 45) #BBox to clip dataset to
freq = 'ME'  # Resample Frequency e.g. ME for Month End, see xarray.cftime_range docs

# ------------------------------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------------------------------

### Define a function that sets up a dask cluster.
# By default gets 1 core w/ 16 GB memory
def get_ClusterClient(ncores=1, nmem='16GB'):
    import dask
    from dask_jobqueue import PBSCluster
    from dask.distributed import Client
    ncores = ncores
    nmem = nmem

    cluster = PBSCluster(
        cores=ncores, # The number of cores you want
        memory=nmem, # Amount of memory
        processes=ncores, # How many processes
        queue='casper', # The type of queue to utilize (/glade/u/apps/dav/opt/usr/bin/execcasper)
        resource_spec='select=1:ncpus='+str(ncores)+':mem='+nmem, # Specify resources
        account='P93300641', # Input your project ID here
        walltime='2:30:00', # Amount of wall time
        interface='ext', # Interface to use 'lo' provided a cluster window, below.
    )

    dask.config.set({
        'distributed.dashboard.link':
        'https://jupyterhub.hpc.ucar.edu/stable/user/{USER}/proxy/{port}/status'
    })

    client = Client(cluster)
    return cluster, client

### Define a function that preprocess xarray data.
# - Clipes data to bbox
# - scales lat/lon to be -180 to 180 intstead of 360
# - renames coordinate dimensions
# - Parses times
# - Aggregates to user specified frequency 

def _preprocess(ds, lon_bnds, lat_bnds, freq):
    # Adjust coordinates from 0 - 360 lon to regular dd lon to be more intuitive
    ds = ds.assign_coords(longitude=('lon', (360 - ds['LONGXY'].data[0, :]) * -1),
                          latitude=('lat', ds['LATIXY'].data[:, 0])) 

    # Rename old dims to avoid confusion
    ds = ds.rename({'lon': 'x', 'lat': 'y'})  

    # Swamp more intuitive coords into ds coords
    ds = ds.swap_dims({'x': 'longitude', 'y': 'latitude'})

    # Decode times to Climate and Forceast Standards as np datetimen64
    ds['time'] = xr.decode_cf(ds).time
    
    # Subset Dataset for region of interest.
    ds = ds.sel(longitude=slice(*lon_bnds), latitude=slice(*lat_bnds))

    # Compute resampled average along time domain.
    ds = ds.resample({'time':freq}).mean()
    
    return ds

### Define a function to read in tundra_pts and build into an xr dataarray for sampling.
def build_sample_locations():
    # Read in a df of sample locations
    points_df = pd.read_csv(path_to_samples)

    # Extract the coordinates as numpy arrays
    lons = points_df['longitude'].values
    lats = points_df['latitude'].values

    # Create an xarray.DataArray with the points
    points_da = xr.DataArray(
        np.stack([lons, lats], axis=1),
        dims=['points', 'coords'],
        coords={'points': points_df.index, 'coords': ['lon', 'lat']}
    )

    return points_da
    

# ------------------------------------------------------------------------------------------------------
# Initiate Cluster
# ------------------------------------------------------------------------------------------------------
cluster, client = get_ClusterClient(nmem='2GB')
cluster.scale(60) 
# cluster

# ------------------------------------------------------------------------------------------------------
# Open all files and (optionally) preprocess nldas data to optimize memory 
# ------------------------------------------------------------------------------------------------------

# Partially define function w lat lon bounds to subset by
preprocess = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds, freq=freq)

# Reading in and Preprocessing data ---

print(f'Initializing datasets, clipping to {lon_bnds}, {lat_bnds} and resampling to {freq}...')

precip_ds = xr.open_mfdataset(
    ppt_fns, preprocess=preprocess, data_vars='minimal', parallel=True
).chunk({'time': -1, 'latitude': 1, 'longitude': 1}) #.persist()
# _ = wait(precip_ds)

tp_ds = xr.open_mfdataset(
    tp_fns, preprocess=preprocess, data_vars='minimal', parallel=True
).chunk({'time': -1, 'latitude': 1, 'longitude': 1}) #.persist()
_ = wait(tp_ds)

print('Finished loading datasets!')

print(precip_ds)
print(tp_ds)

print(f'Saving data to {output_dir} ...')
precip_ds.to_netcdf(os.path.join(output_dir, "nldas_srm_precip.nc"))
# tp_ds.to_netcdf(os.path.join(output_dir, "nldas_srm_tpqwl.nc"))

print('Shutting down Dask Cluster...')
client.shutdown()
del(precip_ds)
del(tp_ds)
print('Done!')

client.shutdown()

# ------------------------------------------------------------------------------------------------------
# Otionally get a space-time long data frame of pixels that overlap the tundra region.
# ------------------------------------------------------------------------------------------------------

if extract_samples:
    points_da = build_sample_locations()
    sampled_precip = precip_ds['PRECTmms'].sel(
        longitude=points_da.sel(coords='lon'),
        latitude=points_da.sel(coords='lat'),
        method='nearest'
    )
    sampled_tbot = tp_ds['TBOT'].sel(
    longitude=points_da.sel(coords='lon'),
    latitude=points_da.sel(coords='lat'),
    method='nearest'
)
    
    # Convert to a DataFrame
    ppt_samples = sampled_precip.to_dataframe().reset_index()
    tbot_samples = sampled_tbot.to_dataframe().reset_index()
    
    # Save sampled data
    ppt_samples.to_csv(f"../data/ppt_samples_{freq}.csv", index = False)
    tbot_samples.to_csv(f"../data/tbot_samples_{freq}.csv", index = False)

# End of Script #