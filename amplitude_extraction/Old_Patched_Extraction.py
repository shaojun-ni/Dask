import numpy as np
from geoio.geoio import GeoIoVolume, GeoIoHorizonCollection
from IPython.display import clear_output
import os
import dask.array as da

# from numba import njit, jit

# @jit(parallel=True)
def read_horizon(horizons, event,q,r,k,l):
    event_data = horizons.get_event(event).get_picks((q,r),(k-1,l-1))
    return event_data

# @njit
def calculate_min_max_index(event_data,first_sample,delta_sample,top_window,bottom_window,num_sample):
    skip = 0
    min_time_index = first_sample
    max_time_index = num_sample 
    # if all the elements of the patch are NaN, skip it
    if da.isnan(event_data).all():
        skip = 1
        return min_time_index, max_time_index, skip, top_window, bottom_window

    min_time = da.nanmin(event_data)
    max_time = da.nanmax(event_data)

    # If horizon minimum time is larger than seismic maximum time (if horizon is below seismic limit), skip the patch
    if min_time >= ((first_sample+num_sample-1) * delta_sample):
        skip = 1
        return min_time_index, max_time_index, skip, top_window, bottom_window
    # If horizon maximum time is smaller than seismic minimum time (if horizon is above seismic limit), skip the patch
    if max_time <= (first_sample * delta_sample):
        skip = 1
        return min_time_index, max_time_index, skip, top_window, bottom_window

    # Convert time to index
    min_time_index = int(da.round((min_time / delta_sample)  - first_sample))
    if min_time_index<0: min_time_index = first_sample
    max_time_index = int(da.round((max_time / delta_sample)  - first_sample))
    if max_time_index>num_sample: max_time_index = num_sample - 1

    # Make sure we don't go beyond top and bottom of seismic data when there is a window
    if  (min_time_index >= first_sample)and(top_window > (min_time_index - first_sample)):
        top_window = first_sample - min_time_index
    if (max_time_index < num_sample)and((max_time_index + bottom_window) >= num_sample):
        bottom_window = num_sample - max_time_index - 1

    return min_time_index, max_time_index, skip, top_window, bottom_window

# @njit
def calculate_boundaries(i,j,n,m, num_i, num_j):
    q = i * n 
    r = j * m 
    k = (i+1) * n 
    l = (j+1) * m 
    if q > num_i: q = num_i - 1
    if r > num_j: r = num_j - 1
    if k > num_i: k = num_i 
    if l > num_j: l = num_j 
    return q,r,k,l 

# @njit
def get_picks_data(event_data,first_sample,delta_sample,min_time_index):
    picks_data = (da.round(event_data / delta_sample - first_sample)).astype(int)
    picks_data -= min_time_index 
    # Bring back NaNs
    picks_data = (picks_data * (event_data / event_data)).astype(int)
    return picks_data

# @njit
def create_grid(cube_shape):
    ii,jj = da.meshgrid(range(cube_shape[0]), range(cube_shape[1]), sparse=False, indexing='ij')
    kk = da.arange(cube_shape[2])
    ii = ii.reshape(ii.shape[0],ii.shape[1],1)
    jj = jj.reshape(jj.shape[0],jj.shape[1],1)
    kk = kk.reshape(1,1,kk.shape[0]) 
    return ii,jj,kk

# @njit
def calcualte_cube_sigma(picks_data,ii,jj,kk,top_window, bottom_window):
    cube_sigma = da.where((kk<=picks_data[ii,jj]+bottom_window) & (kk>=picks_data[ii,jj]-top_window), 1, 0)
    return cube_sigma

# @njit
def calculate_cube_nan(cube_sigma, vtdata):
    cube_nan = da.where(cube_sigma ==0, da.nan, vtdata)
    return cube_nan

def read_info(cube, horizons):
    survey = cube.get_survey()
    events = horizons.get_event_names()
    first_sample = survey.first_sample()
    delta_sample = survey.delta_sample()
    num_sample = survey.num_sample()

    num_i = survey.num_i()
    num_j = survey.num_j()
    return events, first_sample, delta_sample, num_sample, num_i, num_j

# @njit
def calculate_rms(cube_nan):
    amp = da.sqrt(da.nanmean(da.square(cube_nan), axis = 2))
    return amp

# @njit
def calculate_avg(cube_nan):
    amp = da.nanmean(cube_nan, axis=2)
    return amp

# @njit
def calculate_spa(cube_nan):
    # cube_pa = [(i > 0)*i for i in cube_nan]
    cube_pa = da.where(cube_nan>0, cube_nan, 0)
    amp = da.nansum(cube_pa, axis=2)
    del cube_pa
    return amp

# @njit
def calculate_sna(cube_nan):
    # cube_na = [(i < 0)*i for i in cube_nan]
    cube_na = da.where(cube_nan<0, cube_nan, 0)
    amp = da.nansum(cube_na, axis=2)
    del cube_na
    return amp

# @njit
def calculate_saa(cube_nan):
    cube_aa = [da.abs(i) for i in cube_nan]
    amp = da.nansum(cube_aa, axis=2)
    del cube_aa
    return amp

def PExtract (cube: GeoIoVolume, horizons: GeoIoHorizonCollection, 
              extraction_method, top_window = 5, bottom_window = 5,
              number_of_i_patches = 5, number_of_j_patches = 5):

    events, first_sample, delta_sample, num_sample, num_i, num_j = read_info(cube,horizons)

    n = int(da.round(num_i/number_of_i_patches))
    m = int(da.round(num_j/number_of_j_patches))
    event_counter = 0
    extracted_horizons = da.empty((len(events),num_i,num_j))

    method = extraction_method.lower()
    if method in ['inst','instant','ins', 'instantaneous']:
        top_window = 0
        bottom_window = 0

    for e in range(len(events)):

        patched_horizon = da.empty((num_i,num_j))
        nan_horizon = da.ones((num_i,num_j))
        min_time_index = 0
        max_time_index = 0

        for i in range(number_of_i_patches):
            for j in range(number_of_j_patches):

                update_progress((e*number_of_i_patches*number_of_j_patches)+(i*number_of_j_patches)+j+1,number_of_i_patches*number_of_j_patches * len(events))
                                
                q,r,k,l = calculate_boundaries(i,j,n,m, num_i, num_j)
  
                event_data = read_horizon(horizons, events[e],q,r,k,l)
                
                skip = 0
                min_time_index, max_time_index, skip, top_window, bottom_window = calculate_min_max_index(event_data,first_sample,delta_sample,top_window,bottom_window,num_sample)
                if skip == 1: 
                    nan_horizon[q:k,r:l] = da.nan
                    patched_horizon[q:k,r:l] = da.nan
                    continue

                # Create a patch of 1s and NaNs
                nan_horizon[q:k,r:l] = event_data / event_data

                # This is for the case that min and max index are too close to each other. If there is less than 2 numbers between them, extracted seismic will not be 3D.
                if max_time_index<=(min_time_index+1): 
                    if (max_time_index+1) < num_sample:
                        max_time_index += 1
                        min_time_index -= 1 
                    else:
                        min_time_index -= 2 

                # Get picks_data which is index numbers caluclated from time values
                picks_data = get_picks_data(event_data,first_sample,delta_sample,min_time_index)

                # Extract subcube of seismic data
                vtdata = cube.get_float((q, r, min_time_index - top_window),(k-1, l-1, max_time_index + bottom_window))
                    
                #create a grid to host amplitude values
                cube_shape = vtdata.shape
                ii,jj,kk = create_grid(cube_shape)

                picks_data[picks_data<0] = 0
                picks_data[picks_data >= cube_shape[2]] = cube_shape[2] - 1

                if method in ['inst','instant','ins', 'instantaneous']:
                    amp = da.squeeze(vtdata[ii,jj,picks_data[ii,jj]], axis = 2)
                    patched_horizon[q:k,r:l] = amp
                elif method in ['rms']:
                    patched_horizon[q:k,r:l] = calculate_rms(calculate_cube_nan(calcualte_cube_sigma(picks_data,ii,jj,kk,top_window,bottom_window), vtdata))
                elif method in ['average','avg','ave','mean']:
                    patched_horizon[q:k,r:l] = calculate_avg(calculate_cube_nan(calcualte_cube_sigma(picks_data,ii,jj,kk,top_window,bottom_window), vtdata))
                elif method in ['spa']:
                    patched_horizon[q:k,r:l] = calculate_spa(calculate_cube_nan(calcualte_cube_sigma(picks_data,ii,jj,kk,top_window,bottom_window), vtdata))
                elif method in ['sna']:
                    patched_horizon[q:k,r:l] = calculate_sna(calculate_cube_nan(calcualte_cube_sigma(picks_data,ii,jj,kk,top_window,bottom_window), vtdata))
                elif method in ['saa']:
                    patched_horizon[q:k,r:l] = calculate_saa(calculate_cube_nan(calcualte_cube_sigma(picks_data,ii,jj,kk,top_window,bottom_window), vtdata))


        # Make sure all NaNs are considered
        patched_horizon = patched_horizon * nan_horizon
        extracted_horizons[e] = patched_horizon
        event_counter = event_counter + 1

    return extracted_horizons


# Extract instantaneous amplitude slice by slice. It is slower than patched extraction, but easier to implement 
def S_INS (vt: GeoIoVolume, int_file: GeoIoHorizonCollection):
    # vt = GeoIoVolume(cube)
    # int_file = GeoIoHorizonCollection(horizons)
    event_names = int_file.get_event_names()
    vt_survey = vt.get_survey()
    num_i = vt_survey.num_i()
    num_j = vt_survey.num_j()
    num_k = vt_survey.num_k()
    first_sample = vt_survey.first_sample()
    digi = vt_survey.delta_sample()



    nan_horizon = da.ones((num_i,num_j))
    new_picks = da.empty((len(event_names), num_i, num_j))
    io_orient = vt.get_io_orientation()
    for e in range(len(event_names)):
        event = int_file.get_event(event_names[e])
        picks = event.get_picks()
        nan_horizon = picks / picks

        if io_orient[0] < io_orient[1]:
            for i in range(num_i):
                update_progress( (e*num_i)+i+1, num_i*len(event_names))
                i_slice = vt.get2dSeismicSlice((i, 0, 0), (i, num_j-1, num_k-1))
                for j in range(num_j):
                    event_val = picks[i, j]
                    if (not da.isnan(event_val)):
                        # just getting nearest k and not checking limits...
                        k_val = da.round(event_val/digi - first_sample)
                        if k_val<0: k_val = 0
                        if k_val>=num_k: k_val = num_k - 1
                        new_event_val = i_slice[j, int(k_val)]
                        new_picks[e, i, j] = new_event_val
        elif io_orient[1] < io_orient[0]:
            for j in range(num_j):
                update_progress( (e*num_j)+j+1, num_j*len(event_names))
                j_slice = vt.get2dSeismicSlice((0, j, 0), (num_i-1, j, num_k-1))
                for i in range(num_i):
                    event_val = picks[i, j]
                    if (not da.isnan(event_val)):
                        # just getting nearest k and not checking limits...
                        k_val = da.round(event_val/digi - first_sample)
                        if k_val<0: k_val = 0
                        if k_val>=num_k: 
                            new_picks[e, i, j] = np.nan
                            continue
                        # if k_val>=num_k: k_val = num_k - 1
                        new_event_val = j_slice[i, int(k_val)]
                        new_picks[e, i, j] = new_event_val            

        new_picks[e] = new_picks[e] * nan_horizon


    # save_horizon(int_file, event_names[0], new_picks[0].astype(float),'./test5.int')

    return new_picks

#Progress Bar
def update_progress(i,j):
    bar_length = 50
    progress = i/j
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))
    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)


