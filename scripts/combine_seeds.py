import glob
import os
import dask
import numpy as np
import xarray as xr

# Combining a handful of small NetCDFs is trivial compute, but the default
# dask "threads" scheduler serialises every worker read and the final write
# through HDF5's global lock. That lock ordering occasionally inverts and
# to_netcdf() deadlocks with zero CPU and no error (the "combine seeds" step
# in presto-LMR run 26911314285 hung for 45+ min while a byte-identical run
# finished in 20 s). The synchronous scheduler runs one chunk at a time in the
# main thread — no lock contention, and peak memory is still bounded by
# CHUNK_TIME — so the deadlock cannot occur.
dask.config.set(scheduler='synchronous')

RECON_DIR  = '/recons'
CHUNK_TIME = 50   # time steps materialised at once during write (~few MB/chunk)
OUT_PATH   = os.path.join(RECON_DIR, 'combined_recon.nc')

files = sorted(glob.glob(os.path.join(RECON_DIR, 'job_r*_recon.nc')))
if not files:
    raise RuntimeError(f'No job_r*_recon.nc files found in {RECON_DIR}')
print(f'Combining {len(files)} seed file(s): {[os.path.basename(f) for f in files]}', flush=True)

tas_list    = []
tas_gm_list = []
for f in files:
    ds = xr.open_dataset(f, chunks={'time': CHUNK_TIME})
    tas_list.append(ds['tas'])     # (time, lat, lon) — ensemble mean, no ens dim

    # Each file's tas_gm has ens=[0..nens-1].  Drop the coordinate so xarray
    # builds a clean RangeIndex when concatenating across seeds.
    tas_gm_list.append(ds['tas_gm'].drop_vars('ens'))

# tas: one ensemble-mean field per seed → concat along 'seed' dim, not 'ens'
# (ens is reserved for the full ensemble members in tas_gm)
tas    = xr.concat(tas_list,    dim='seed').transpose('time', 'seed', 'lat', 'lon')

# tas_gm: all ensemble members across all seeds → concat along 'ens' dim
tas_gm = xr.concat(tas_gm_list, dim='ens')

# int16 quantization keeps combined_recon.nc under GitHub's 100 MB file limit.
# Per-variable scale_factor sized to each field's dynamic range:
#   tas (full field, wide range ~+/-100 K):  0.01  -> 0.005 K precision
#   tas_gm (global mean, ~+/-5 K):           0.0001 -> 0.00005 K precision
def _encoding(da):
    rng = float(max(abs(da.min()), abs(da.max())))
    scale = 0.0001 if rng < 5 else 0.01
    return {
        'zlib': True, 'complevel': 5, 'shuffle': True,
        'dtype': 'int16', 'scale_factor': scale, 'add_offset': 0.0,
        '_FillValue': np.int16(-32768),
    }

out = xr.Dataset({'tas': tas, 'tas_gm': tas_gm})
encoding = {v: _encoding(out[v]) for v in out.data_vars}
out.to_netcdf(OUT_PATH, encoding=encoding)

print(f'combined_recon.nc written: {OUT_PATH}', flush=True)
print(f'  tas    {dict(tas.sizes)}', flush=True)
print(f'  tas_gm {dict(tas_gm.sizes)}', flush=True)
