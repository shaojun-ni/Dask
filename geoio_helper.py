import os
import numpy as np
import dask.array as da
from geoio.geoio import GeoIoVolume, SpVerticalTraceHeaderArray, SpVerticalTraceCheckArray
from dask import delayed, compute


# load vt data into meemory
def load_vt(filename):
    intercept = GeoIoVolume(filename)
    return intercept.get_float()


# write data into vt.
# @delayed
def write_vt(data, filename, original_vt):
    vt = GeoIoVolume(original_vt)
    hd, ck = vt.get_header_info()
    hd.min_clip_amp = float(da.min(data))
    hd.max_clip_amp = float(da.max(data))

    if ck.track_dir == ord("H"):  # switch bin and tracks if track_dir = V
        ck.num_bins = data.shape[0]
        ck.num_tracks = data.shape[1]
        ck.num_samples = data.shape[2]
    else:
        ck.num_bins = data.shape[1]
        ck.num_tracks = data.shape[0]
        ck.num_samples = data.shape[2]

    if os.path.exists(filename):
        os.remove(filename)
    output_vt = GeoIoVolume(filename, hd, ck)
    output_vt.put(np.array(data))

