# A set of analysis functions for QGPV analysis of CMIP data
# Author: Ryan Eagan - May 2025

# To speed caclulations and take advantage of under the hood optimization, 
# this uses xarray's differentiate method which acts on a 1-D coordinate and 
# calculates the 2nd order accurate central differences, similar to numpy's gradient feature.
# https://docs.xarray.dev/en/stable/generated/xarray.Dataset.differentiate.html

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA

def open_datasets(u_path, v_path, z_path):
    """
    Open the individual CMIP datasets for U, V, and Z and return a single dataset
    with just the Southern Hemisphere.
    
    Parameters:
    -----------
    u_path : text
        A text string to the location of the netCDF for the U component of wind
    
    v_path : text
        A text string to the location of the netCDF for the Z component of wind
        
    z_path : text
        A text string to the location of the netCDF for the geopotential height
    
    Returns:
    --------
    ds : xarray dataset
    """
    
    uwind = xr.open_dataset(u_path)
    vwind = xr.open_dataset(v_path)
    gp = xr.open_dataset(z_path)
    
    # Merge all the datasets into one dataset
    ds_raw = xr.merge([gp, uwind, vwind])

    # We just want the southern hemisphere from 45S to the southern pole
    ds = ds_raw.sel(lat=slice(-90, -45))
    
    return ds

def calc_field_anom(da):
    """
    Calculates the spatial anomalies over the given timeseries in the data
    
    Parameters:
    -----------
    da : xarray.DataArray
        3D or 4D data array with time and spatial dimensions (and optionally level)
    
    Returns:
    --------
    d_anom : xarray.DataArray
        DataArray of anomalies with time, lev, and spatial dimensions
    """
    da_mean = da.mean(dim=['lat','lon'])
    da_anom = da - da_mean
    
    return da_anom

def calc_trend(da):
    """
    Calculates the spatial trend over the given timeseries in the data
    
    Parameters:
    -----------
    da : xarray.DataArray
        3D or 4D data array with time and spatial dimensions (and optionally level)
    
    Returns:
    --------
    d_anom : xarray.DataArray
        DataArray of trend with lev, and spatial dimensions
    """
    
    def compute_slope(y, x):
    # Fit 1st-degree polynomial and return the slope
        return np.polyfit(x, y, deg=1)[0]

    time_numeric = da['time'].astype('datetime64[Y]').astype(int)

    # Apply over (time) dimension for each (lat, lon)
    trend = xr.apply_ufunc(
        compute_slope,
        da, time_numeric,
        input_core_dims=[['time'], ['time']],
        vectorize=True,
        dask='parallelized',  # Optional: if using dask
        output_dtypes=[float]
    )

def calc_psi(Z,lev,lat):
    """
    Calculates the geostrophic streamfunction from geopotential height Z, on a given level.
    Requires latitude to calculate f0 (Coriolis param)
    
    Parameters:
    -----------
    Z : xarray.DataArray
        3D or 4D data array with time and spatial dimensions (and optionally level)
    
    lev : xarray.DataArray
        A text string to the location of the netCDF for the Z component of wind
        
    lat : xarray.DataArray
        A text string to the location of the netCDF for the geopotential height
    
    Returns:
    --------
    da : xarray.DataArray
        DataArray with time, lev, and spatial dimensions
    """
    # Define constants

    g = 9.81  # Gravity [m/s^2]
    
    omega = 7.2921e-5  # Earth's rotation rate [s^-1]
    f0 = 2 * omega * np.sin(np.deg2rad(lat))  # Coriolis parameter

    # Compute streamfunction ψ
    psi = (Z * g) * (lev / f0)
    
    return psi

def calc_laplacian_psi(lat,lon,psi):
    """
    Calculates the laplacian of the stream function returned from calc_psi()
    
    Parameters:
    -----------
    lat : xarray.DataArray
        DataArray of latitude coordiantes
    
    lon : xarray.DataArray
        DataArray of longitude coordiantes
        
    psi : xarray.DataArray
        The psi DataArray returned from calc_psi()
    
    Returns:
    --------
    da : xarray.DataArray
        DataArray with time, lev, and spatial dimensions
    """
    
    # Assuming lat and lon are 1D arrays, and psi is a 2D DataArray with dims ('lat', 'lon')
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Convert degrees to radians and compute dx, dy as 2D arrays
    R = 6371000  # Earth radius [m]
    dx = np.gradient(lon) * (np.pi / 180) * R * np.cos(np.deg2rad(lat2d))  # shape: (lat, lon)
    dy = np.gradient(lat) * (np.pi / 180) * R  # shape: (lat,)

    # Broadcast dy to 2D to match shape
    dy2d = np.broadcast_to(dy[:, np.newaxis], dx.shape)

    # Compute second derivatives using xarray DataArray operations (assuming psi is a DataArray)
    d2psi_dx2 = (psi.shift(lon=-1) - 2 * psi + psi.shift(lon=1)) / (dx ** 2)
    d2psi_dy2 = (psi.shift(lat=-1) - 2 * psi + psi.shift(lat=1)) / (dy2d ** 2)

    laplacian_psi = d2psi_dx2 + d2psi_dy2
    
    return laplacian_psi

def calc_qgvp_vert(lev):
    """
    Calculates the laplacian of the stream function returned from calc_psi()
    
    Parameters:
    -----------
    lat : xarray.DataArray
        DataArray of latitude coordiantes
    
    lon : xarray.DataArray
        DataArray of longitude coordiantes
        
    psi : xarray.DataArray
        The psi DataArray returned from calc_psi()
    
    Returns:
    --------
    da : xarray.DataArray
        DataArray with time, lev, and spatial dimensions
    """
    
    #########################################
    # THIS IS CURRENTLY BROKEN, NEEDS WORK  #
    #########################################
    
    omega = 7.2921e-5  # Earth's rotation rate [s^-1]
    f0 = 2 * omega * np.sin(np.deg2rad(lat))  # Coriolis parameter
    
    R = 287.
    # Ensure lev is a coordinate
    T = T.assign_coords(lev=lev)
    psi = psi.assign_coords(lev=lev)

    # Compute potential temperature
    theta = T * (100000 / T.lev) ** (R / 1004)

    # Compute d(log(theta))/dp along 'lev' (pressure) dimension
    log_theta = np.log(theta)
    dtheta_dp = log_theta.differentiate('lev')

    # Static stability parameter sigma
    sigma = -(R * T / T.lev) * dtheta_dp

    # Vertical derivatives of psi
    dpsi_dp = psi.differentiate('lev')
    d2psi_dp2 = dpsi_dp.differentiate('lev')

    # QGPV vertical term
    QGPV_vertical = ((f0**2 / sigma) * dpsi_dp).differentiate('lev')
    
    return QGPV_vertical

def calc_q(laplacian_psi, QGPV_vertical, lat):
    """
    Calculates QGPV (q) using the laplacian_psi and QGPV_vertical function's output.
    
    Parameters:
    -----------
    laplacian_psi : xarray.DataArray
        Laplacian DataArray from calc_laplacian_psi()
        
    QGPV_vertical : xarray.DataArray
        QGVP vertical DataArray from calc_qgvp_vert()
        
    lat : xarray.DataArray
        DataArray of latitude coordiantes
    
    Returns:
    --------
    da : xarray.DataArray
        DataArray with time, lev, and spatial dimensions
    """
    
    omega = 7.2921e-5  # Earth's rotation rate [s^-1]
    f0 = 2 * omega * np.sin(np.deg2rad(lat))  # Coriolis parameter
    
    # Compute full QGPV
    QGPV = laplacian_psi + f0 + QGPV_vertical # vertical broken
        
    return QGPV

def calc_geostrophic_vort(lat, lon, Z):
    """
    Calculates the geostrophic vorticity from geopotential height Z
    
    Parameters:
    -----------
    lat : xarray.DataArray
        DataArray of latitude coordiantes
    
    lon : xarray.DataArray
        DataArray of longitude coordiantes
        
    Z : xarray.DataArray
        The geopotential height in meters
    
    Returns:
    --------
    ug : xarray.DataArray
        DataArray of zonal wind with time, lev, and spatial dimensions
        
    vg : xarray.DataArray
        DataArray of meridional wind with time, lev, and spatial dimensions
        
    zeta_g : xarray.DataArray
        DataArray of relative vorticity with time, lev, and spatial dimensions
    """
    
    omega = 7.2921e-5  # Earth's rotation rate [s^-1]
    f0 = 2 * omega * np.sin(np.deg2rad(lat))  # Coriolis parameter

    # Compute geostrophic winds using derivatives
    ug = - (1 / f0) * Z.differentiate('lat') * (np.pi / 180) * 6371000  # dΦ/dy
    vg = (1 / f0) * Z.differentiate('lon') * (np.pi / 180) * 6371000 * np.cos(np.deg2rad(Z.lat))

    # Compute Geostrophic Vorticity
    dvg_dx = vg.differentiate('lon') * (np.pi / 180) * 6371000 * np.cos(np.deg2rad(Z.lat))
    dug_dy = ug.differentiate('lat') * (np.pi / 180) * 6371000
    zeta_g = dvg_dx - dug_dy
    
    return ug, vg, zeta_g

def calc_rossby_source(U, V, vg, lat, Z, zeta_g):
    """
    Calculates the Rossby Wave Source.
    
    Parameters:
    -----------
    U : xarray.DataArray
        DataArray of U component of wind from original netCDF
        
    V : xarray.DataArray
        DataArray of V component of wind from original netCDF
        
    vg : xarray.DataArray
        DataArray of geostrophic meridional pertubation wind from
        
        calc_geostrophic_vort() function
    lat : xarray.DataArray
        DataArray of latitude coordiantes
    
    Z : xarray.DataArray
        DataArray of geopotential height
        
    zeta_g : xarray.DataArray
        The relative vorticity calculated by calc_geostrophic_vort
    
    Returns:
    --------
    rws : xarray.DataArray
        DataArray of Rossby Wave Source with time, lev, and spatial dimensions
    """
    
    omega = 7.2921e-5  # Earth's rotation rate [s^-1]
    f0 = 2 * omega * np.sin(np.deg2rad(lat))  # Coriolis parameter
    
    # Compute the Rossby Wave Source

    dU_dx = U.differentiate('lon') * (np.pi / 180) * 6371000 * np.cos(np.deg2rad(Z.lat))
    dV_dy = V.differentiate('lat') * (np.pi / 180) * 6371000
    dvg_dy = vg.differentiate('lat') * (np.pi / 180) * 6371000

    rws = - (zeta_g * dU_dx + dV_dy) - f0 * dvg_dy
    
    return rws

def compute_eofs_general(
    da,
    time_dim='time',
    spatial_dims=('lat', 'lon'),
    level_dim=None,
    level_value=None,
    n_modes=3,
    remove_mean=True
):
    """
    Generalized EOF analysis for xarray.DataArray with optional level selection.

    Parameters:
    -----------
    da : xarray.DataArray
        3D or 4D data array with time and spatial dimensions (and optionally level).
        
    time_dim : str
        Name of the time-like dimension.
        
    spatial_dims : tuple of str
        Names of the two spatial dimensions (e.g., ('lat', 'lon')).
        
    level_dim : str or None
        Optional name of the level dimension (e.g., 'lev' or 'plev').
        
    level_value : float or int or None
        Optional value to select from level_dim (e.g., 50000 for 500 hPa).
        
    n_modes : int
        Number of EOF modes to compute.
        
    remove_mean : bool
        If True, removes the mean over the time dimension before computing EOFs.

    Returns:
    --------
    pca : sklearn PCA object
    
    eof_maps : list of xarray.DataArray, each with dimensions spatial_dims
    
    pcs : xarray.DataArray with dims (time_dim, 'mode')
    """

    # Subset level if specified
    if level_dim is not None and level_value is not None:
        if level_dim not in da.dims:
            raise ValueError(f"Level dimension '{level_dim}' not found in DataArray.")
        da = da.sel({level_dim: level_value}, method='nearest')

    # Check required dimensions
    dims_required = {time_dim, *spatial_dims}
    if not dims_required.issubset(set(da.dims)):
        raise ValueError(f"DataArray must have dimensions: {dims_required}")

    # Anomalies
    da_anom = da - da.mean(dim=time_dim) if remove_mean else da.copy()

    # Stack space and ensure consistent ordering
    da_stacked = da_anom.stack(space=spatial_dims).transpose(time_dim, 'space')
    data_matrix = da_stacked.values

    # Mask NaNs
    valid_mask = ~np.any(np.isnan(data_matrix), axis=0)
    data_matrix_valid = data_matrix[:, valid_mask]

    # PCA
    pca = PCA(n_components=n_modes)
    pcs = pca.fit_transform(data_matrix_valid)

    # Reconstruct EOF maps
    eof_maps = []
    for i in range(n_modes):
        eof_1d = np.full(valid_mask.shape, np.nan)
        eof_1d[valid_mask] = pca.components_[i]
        eof_2d = eof_1d.reshape([da.sizes[d] for d in spatial_dims])
        eof_da = xr.DataArray(
            eof_2d,
            coords={d: da[d] for d in spatial_dims},
            dims=spatial_dims,
            name=f'EOF{i+1}'
        )
        eof_maps.append(eof_da)

    # Convert PCs to xarray
    pc_da = xr.DataArray(
        pcs,
        coords={time_dim: da[time_dim], 'mode': np.arange(1, n_modes + 1)},
        dims=(time_dim, 'mode'),
        name='PCs'
    )

    return pca, eof_maps, pc_da

def compute_rws(Z, U, V, f0=1e-4):
    """
    Compute Rossby Wave Source (RWS) from geopotential (Z), and wind components (U, V).
    
    Parameters:
    -----------
    Z : xarray.DataArray
        Geopotential height (m^2/s^2), with dimensions ('time', 'lev', 'lat', 'lon')
    U, V : xarray.DataArray
        Zonal and meridional wind components (m/s), with same dimensions as Z
    f0 : float
        Reference Coriolis parameter (s^-1). Default is mid-latitude value ~1e-4.

    Returns:
    --------
    rws : xarray.DataArray
        Rossby Wave Source (1/s^2), same shape as input fields.
    """

    # Earth radius [m]
    R = 6371000
    deg2rad = np.pi / 180
    
    # Get lat in radians for dx correction
    lat_rad = np.deg2rad(Z['lat'])

    # Compute geostrophic wind components
    dZ_dlat = Z.differentiate('lat') * deg2rad * R
    dZ_dlon = Z.differentiate('lon') * deg2rad * R * np.cos(lat_rad)

    ug = - (1 / f0) * dZ_dlat
    vg = (1 / f0) * dZ_dlon

    # Compute geostrophic vorticity
    dug_dlat = ug.differentiate('lat') * deg2rad * R
    dvg_dlon = vg.differentiate('lon') * deg2rad * R * np.cos(lat_rad)
    zeta_g = dvg_dlon - dug_dlat

    # Derivatives of basic flow
    dU_dlon = U.differentiate('lon') * deg2rad * R * np.cos(lat_rad)
    dV_dlat = V.differentiate('lat') * deg2rad * R
    dvg_dlat = vg.differentiate('lat') * deg2rad * R

    # Rossby wave source
    rws = - (zeta_g * dU_dlon + dV_dlat) - f0 * dvg_dlat
    rws.name = 'RWS'

    return rws