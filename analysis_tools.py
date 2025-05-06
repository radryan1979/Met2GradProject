import numpy as np
import xarray as xr
from sklearn.decomposition import PCA

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

import numpy as np
import xarray as xr

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