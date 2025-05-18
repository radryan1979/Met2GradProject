# A set of plotting functions for QGPV analysis of CMIP data
# particularly for the Southern Hemisphere
# Author: Ryan Eagan - May 2025

import xarray as xr
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from os import path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_field(field, title, time_idx, level_value, save_file=None, cmap='viridis', prj='SouthPolarStereo'):
    """Plot a horizontal map of a 3D field at given time and level.

    Parameters:
    -----------
    field : xarray.DataArray
        Wind components with dims ('time', 'plev', 'lat', 'lon')
    title : text
        The name of the field for the plot title
    time_idx : int
        Index of time step to plot, -1 if there is no time coordinate
    level_value : float
        Pressure level to plot (in Pa)
    save_file : text
        Path and file name string to save plot, if None, plot is displayed only.
    cmap : text
        Matplotlib CMAP choice
    prj : text
        Select either PlateCarree or SouthPolarStereo for plot projection
    """
    
    if time_idx < 0:
        field_sel = field.sel(plev=level_value, method="nearest")
    elif time_idx >= 0:
        # Select the data slice
        field_sel = field.sel(time=field.time[time_idx], plev=level_value, method="nearest")

    if prj == 'PlateCarree':
        proj = ccrs.PlateCarree()
    elif prj == 'SouthPolarStereo':
        proj = ccrs.SouthPolarStereo()

    # Create the plot
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=proj)
    field_plot = field_sel.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        cbar_kwargs={"label": f"{field.name}"},
    )
    
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    if time_idx < 0:
            ax.set_title(f"{title} at {int(level_value/100)} hPa")
    elif time_idx >= 0:
        ax.set_title(f"{title} at {int(level_value/100)} hPa, time: {str(field.time[time_idx].values)[:10]}")

    if save_file is None:
        plt.show()
    else:    
        plt.savefig(save_file)
    

def plot_wind_barbs(U, V, save_file=None, time_idx=0, level_value=85000, stride=5, prj='SouthPolarStereo'):
    """
    Plot wind barbs from U and V wind components at a given time and pressure level.

    Parameters:
    -----------
    U, V : xarray.DataArray
        Wind components with dims ('time', 'plev', 'lat', 'lon')
    save_file : text
        Path and file name string to save plot, if None, plot is displayed only
    time_idx : int
        Index of time step to plot
    level_value : float
        Pressure level to plot (in Pa)
    stride : int
        Skip factor for quiver density (e.g., 5 = every 5th grid point)
    prj : text
        Select either PlateCarree or SouthPolarStereo for plot projection
    """
    land = cfeature.NaturalEarthFeature(
        'physical', 'land', '110m',
        edgecolor='face',
        facecolor='#f5f5dc'  # light beige
    )

    ocean = cfeature.NaturalEarthFeature(
        'physical', 'ocean', '110m',
        edgecolor='face',
        facecolor='#a4c8ea'  # soft blue
    )
    
    if prj == 'PlateCarree':
        proj = ccrs.PlateCarree()
    elif prj == 'SouthPolarStereo':
        proj = ccrs.SouthPolarStereo()
    
    # Subset data
    u_sel = U.sel(time=U.time[time_idx], plev=level_value, method="nearest")
    v_sel = V.sel(time=V.time[time_idx], plev=level_value, method="nearest")

    # Extract lat/lon
    lats = u_sel['lat'].values
    lons = u_sel['lon'].values
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Convert to 2D for plotting
    u_plot = u_sel.values
    v_plot = v_sel.values

    # Plot
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=proj)
    
    ax.add_feature(ocean, zorder=0)
    ax.add_feature(land, zorder=1)

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_title(f"Wind Barbs at {int(level_value/100)} hPa, time: {str(U.time[time_idx].values)[:10]}")

    # Plot barbs (subset using stride to avoid overcrowding)
    ax.barbs(
        lon2d[::stride, ::stride],
        lat2d[::stride, ::stride],
        u_plot[::stride, ::stride],
        v_plot[::stride, ::stride],
        length=6,
        transform=ccrs.PlateCarree(),
    )

    if save_file is None:
        plt.show()
    else:    
        plt.savefig(save_file)

def plot_wind_streamlines(U, V, save_file=None, time_idx=0, level_value=85000, density=2, prj='SouthPolarStereo'):
    """
    Plot streamlines from U and V wind components at a specific time and pressure level.

    Parameters:
    -----------
    U, V : xarray.DataArray
        Wind components with dims ('time', 'plev', 'lat', 'lon')
    save_file : text
        Path and file name string to save plot, if None, plot is displayed only
    time_idx : int
        Index of time step to plot
    level_value : float
        Pressure level to plot (in Pa)
    density : float
        Streamline density parameter (higher = more lines)
    prj : text
        Select either PlateCarree or SouthPolarStereo for plot projection
    """
    
    land = cfeature.NaturalEarthFeature(
        'physical', 'land', '110m',
        edgecolor='face',
        facecolor='#f5f5dc'  # light beige
    )

    ocean = cfeature.NaturalEarthFeature(
        'physical', 'ocean', '110m',
        edgecolor='face',
        facecolor='#a4c8ea'  # soft blue
    )
    
    if prj == 'PlateCarree':
        proj = ccrs.PlateCarree()
    elif prj == 'SouthPolarStereo':
        proj = ccrs.SouthPolarStereo()
    
    # Select data at desired time and level
    u = U.sel(time=U.time[time_idx], plev=level_value, method="nearest")
    v = V.sel(time=V.time[time_idx], plev=level_value, method="nearest")

    # Create 2D lat/lon meshgrid
    lats = u['lat'].values
    lons = u['lon'].values
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Plot setup
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=proj)

    # Background color and features
    ax.set_extent([-180, 180, -90, -45], ccrs.PlateCarree())
    ax.add_feature(ocean, zorder=0)
    ax.add_feature(land, zorder=1)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_title(f"Wind Streamlines at {int(level_value/100)} hPa, time: {str(U.time[time_idx].values)[:10]}")

    # Streamplot requires 1D x/y grid and 2D u/v fields
    stream = ax.streamplot(
        lon2d,
        lat2d,
        u.values,
        v.values,
        transform=ccrs.PlateCarree(),
        density=density,
        color='k',  # or use speed: np.sqrt(u**2 + v**2)
        linewidth=1
    )

    plt.tight_layout()
    
    if save_file is None:
        plt.show()
    else:    
        plt.savefig(save_file)

def plot_geopotential_contours(Z, save_file=None, time_idx=0, level_value=50000, cmap='viridis', prj='SouthPolarStereo'):
    """
    Plot geopotential height contours from CMIP5 data.

    Parameters:
    -----------
    Z : xarray.DataArray
        Geopotential (m^2/s^2) with dims ('time', 'plev', 'lat', 'lon')
    save_file : text
        Path and file name string to save plot, if None, plot is displayed only
    time_idx : int
        Index of time to plot
    level_value : float
        Pressure level (in Pa)
    cmap : str
        Colormap for filled contours (optional)
    prj : text
        Select either PlateCarree or SouthPolarStereo for plot projection
    """
    
    land = cfeature.NaturalEarthFeature(
        'physical', 'land', '110m',
        edgecolor='face',
        facecolor='#f5f5dc'  # light beige
    )

    ocean = cfeature.NaturalEarthFeature(
        'physical', 'ocean', '110m',
        edgecolor='face',
        facecolor='#a4c8ea'  # soft blue
    )
    
    if prj == 'PlateCarree':
        proj = ccrs.PlateCarree()
    elif prj == 'SouthPolarStereo':
        proj = ccrs.SouthPolarStereo()
        
    # Select the desired slice
    z_plot = Z.sel(time=Z.time[time_idx], plev=level_value, method="nearest")

    # Convert from geopotential to geopotential height in meters: Z / g
    g = 9.80665
    z_height = z_plot / g

    # Get lat/lon grid
    lats = z_height['lat'].values
    lons = z_height['lon'].values
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Plot
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=proj)
    
    # Background
    ax.set_extent([-180, 180, -90, -45], ccrs.PlateCarree())
    ax.add_feature(ocean, zorder=0)
    ax.add_feature(land, zorder=1)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    # Contour levels
    levels = np.arange(np.nanmin(z_height), np.nanmax(z_height), 5)

    # # Filled contours
    # cf = ax.contourf(
    #     lon2d, lat2d, z_height,
    #     levels=levels,
    #     cmap=cmap,
    #     transform=ccrs.PlateCarree()
    # )
    
    # Line contours
    cs = ax.contour(
        lon2d, lat2d, z_height,
        levels=levels,
        colors='k',
        linewidths=0.5,
        transform=ccrs.PlateCarree()
    )
    
    # Add labels and colorbar
    ax.clabel(cs, inline=True, fontsize=8)
    #cbar = plt.colorbar(cf, orientation='vertical', pad=0.02, aspect=20, ax=ax)
    #cbar.set_label("Geopotential Height (m)")
    
    ax.set_title(f"Geopotential Height at {int(level_value/100)} hPa\nTime: {str(Z.time[time_idx].values)[:10]}")
    plt.tight_layout()
    
    if save_file is None:
        plt.show()
    else:    
        plt.savefig(save_file)

def plot_eof(eof_da, save_file=None, title=None, cmap='RdBu_r', prj='SouthPolarStereo'):
    """
    Plot streamlines from U and V wind components at a specific time and pressure level.

    Parameters:
    -----------
    eof_da : xarray.DataArray
        Xarray DataArray with the EOFS
    save_file : text
        Path and file name string to save plot, if None, plot is displayed only
    title : text
        Title for the plot.
    cmap : text
        Matplotlib CMAP
    prj : text
        Select either PlateCarree or SouthPolarStereo for plot projection
    """
    
    if prj == 'PlateCarree':
        proj = ccrs.PlateCarree()
    elif prj == 'SouthPolarStereo':
        proj = ccrs.SouthPolarStereo()
        
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.set_extent([-180, 180, -90, -45], ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    im = eof_da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        cbar_kwargs={'label': 'EOF amplitude'},
    )

    ax.set_title(title or eof_da.name)
    plt.tight_layout()
    
    if save_file is None:
        plt.show()
    else:    
        plt.savefig(save_file)

def plot_eofs_panel(eof_da, pca, fig_title, save_file=None, num_modes=4, cmap='RdBu_r', prj='SouthPolarStereo'):
    """
    Plot the first N EOFs in a panel with subplots and a shared colorbar at the bottom.

    Parameters:
    -----------
    eof_da : list<xarray.DataArray>
        List of DataArrays with an EOF mode dimension (e.g., shape = (mode, lat, lon))
    pca : list<xarray.DataArray>
        List of DataArrays with a PCA mode mode dimension (e.g., shape = (mode, lat, lon))
    fig_title : text
        Title for the figure
    save_file : text
        Path and file name string to save plot, if None, plot is displayed only
    num_modes : int
        Number of EOFs to plot (from mode 0 to num_modes-1).
    cmap : str
        Matplotlib colormap.
    prj : str
        Projection type: 'PlateCarree' or 'SouthPolarStereo'.
    """
    if prj == 'PlateCarree':
        proj = ccrs.PlateCarree()
    elif prj == 'SouthPolarStereo':
        proj = ccrs.SouthPolarStereo()

    ncols = 2
    nrows = (num_modes + 1) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5 * nrows),
                             subplot_kw={'projection': proj})

    axes = axes.flat
    vmin = min(float(eof.min().item()) for eof in eof_da)
    vmax = max(float(eof.max().item()) for eof in eof_da)
    im = None

    for i in range(num_modes):
        ax = axes[i]
        ax.set_global()
        ax.set_extent([-180, 180, -90, -45], crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)

        eof = eof_da[i]
        im = eof.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            add_colorbar=False,
            vmin=vmin, vmax=vmax
        )

        ax.set_title(f"EOF Mode {i+1} ({pca.explained_variance_ratio_[i]*100:.2f}% variance)")

    # Remove unused axes
    for j in range(num_modes, nrows * ncols):
        fig.delaxes(axes[j])

    # Shared colorbar at the bottom
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("EOF amplitude")
    
    fig.suptitle(fig_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust for colorbar space
    
    if save_file is None:
        plt.show()
    else:    
        plt.savefig(save_file)