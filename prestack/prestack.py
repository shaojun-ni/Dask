from geoio_helper import load_vt, write_vt
import numpy as np
import dask.array as da
from dask import delayed, compute
import dask


@delayed
def intercep_f(epsilon, near, gamma, mid, delta, far):
    return epsilon * near - gamma * mid + delta * far


@delayed
def curv_f(zeta, ufar, far, eta, mid, near):
    return (zeta * (ufar - far) - eta * (mid - near) + iota * (far - near))


@delayed
def grad_f(alpha, mid, near, beta, far):
    return 0.5 * (alpha*(mid-near) + beta*(far-near))


@delayed
def is_r_f(vpvs, curv, grad, intercept):
    return 0.25 * (vpvs**2) * (curv-grad) + intercept - curv, intercept, curv, grad

#=============================================================================
#  Input data ( Angle stacks and their central angle, low frequency models) All data is entered as RUNSUM
#=============================================================================
# near_vt = "/glb/data/CDIS2/users/uslji0/Attributes/Prestack/seismicCubes_Runsum_noise_free__0_degrees_2019.79273404.vt"
# mid_vt = "/glb/data/CDIS2/users/uslji0/Attributes/Prestack/seismicCubes_Runsum_noise_free__10_degrees_2019.79273404.vt"
# far_vt = "/glb/data/CDIS2/users/uslji0/Attributes/Prestack/seismicCubes_Runsum_noise_free__18_degrees_2019.79273404.vt"
# ufar_vt = "/glb/data/CDIS2/users/uslji0/Attributes/Prestack/seismicCubes_Runsum_noise_free__26_degrees_2019.79273404.vt"
near_vt = "/glb/data/cdis_projects/users/uslji0/Long/RQI_Master/nDI/seismic/2015_Freeman_0-12deg_INV_T_trimmed.vt"
mid_vt = "/glb/data/cdis_projects/users/uslji0/Long/RQI_Master/nDI/seismic/2015_Freeman_8-16deg_INV_T_trimmed.vt"
far_vt = "/glb/data/cdis_projects/users/uslji0/Long/RQI_Master/nDI/seismic/2015_Freeman_16-24deg_INV_T_trimmed.vt"
ufar_vt = "/glb/data/cdis_projects/users/uslji0/Long/RQI_Master/nDI/seismic/2015_Freeman_24-32deg_INV_T_trimmed.vt"
n, m, f, uf = np.deg2rad((0, 8, 16, 24))
near = da.from_array(load_vt(near_vt))
mid = da.from_array(load_vt(mid_vt))
far = da.from_array(load_vt(far_vt))
ufar = da.from_array(load_vt(ufar_vt))

# the amplitude level helps normalising the amplitudes
amp_level = 2000 * near # temp_scalar * near
# Parameters for relative fluid and litho cubes
vpvs = 1.74
vpvs_slope = 1.23
p_slowness = 0.38
ipden_slope = 14.
#=============================================================================
#  Calculate some constants based on the angle of incidence
#=============================================================================
alpha = 3 / (2 * (np.sin(m)**2 - np.sin(n)**2))
beta = 1 / (2 * (np.sin(f)**2 - np.sin(n)**2))
gamma = alpha * np.sin(n)**2
delta = np.sin(n)**2 / (2 * (np.sin(f)**2 - np.sin(n)**2))
epsilon = 1 + gamma - delta
zeta = 1 / (np.sin(uf)**2 * np.tan(uf)**2 - np.sin(f)**2 * np.tan(f)**2)
eta = alpha * ((np.sin(uf)**2 - np.sin(f)**2)/(np.sin(uf)**2*np.tan(uf)**2 - np.sin(f)**2 * np.tan(f)**2))
iota = beta * (eta / alpha)
#=============================================================================
# Use the above constants and angle stacks to calculate Relative Impedances from AVA
#=============================================================================
intercept = intercep_f(epsilon, near, gamma, mid, delta, far) #epsilon*near - gamma*mid + delta*far
curv = curv_f(zeta, ufar, far, eta, mid, near) #(zeta*(ufar-far) - eta*(mid-near) + iota*(far-near))
grad = grad_f(alpha, mid, near, beta, far)  # 0.5 * (alpha*(mid-near) + beta*(far-near))
is_r, intercept, curv, grad = is_r_f(vpvs, curv, grad, intercept).compute()  # 0.25 * (vpvs**2) * (curv-grad) + intercept - curv
ip_r = 2 * intercept
den_r = 2 * (intercept - curv)
#=============================================================================
# Calculate fluid and lithology cubes (Relative)
#=============================================================================
FF = ip_r - is_r * (vpvs_slope / vpvs)
LF = ip_r - p_slowness * ipden_slope * den_r
#=============================================================================
# Write Outputs
#=============================================================================
# Relative Properties
r = []
r.append(write_vt(intercept, "output/intercept_freeman.vt", near_vt))
r.append(write_vt(grad, "output/grad_freeman.vt", near_vt))
r.append(write_vt(curv, "output/curv_freeman.vt", near_vt))
r.append(write_vt(FF, "output/ff_freeeman.vt", near_vt))
r.append(write_vt(LF, "output/lf_freeman.vt", near_vt))
dask.compute(*r)
#=============================================================================
