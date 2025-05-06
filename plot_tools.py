import xarray as xr
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from os import path
import matplotlib.pyplot as plt
from pyproj import Proj, Transformer
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_field(field, title, time_idx, level_value, cmap='viridis', prj='PlateCarree'):
    """Plot a horizontal map of a 3D field at given time and level.

    Parameters:
    -----------
    field : xarray.DataArray
        Wind components with dims ('time', 'plev', 'lat', 'lon')
    title : text
        The name of the field for the plot title
    time_idx : int
        Index of time step to plot
    level_value : float
        Pressure level to plot (in Pa)
    cmap : text
        Matplotlib CMAP choice
    prj : text
        Select either PlateCarree or SouthPolarStereo for plot projection
    """
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
    ax.set_title(f"{title} at {int(level_value/100)} hPa, time: {str(field.time[time_idx].values)[:10]}")
    plt.show()

def plot_wind_barbs(U, V, time_idx=0, level_value=85000, stride=5, prj='PlateCarree'):
    """
    Plot wind barbs from U and V wind components at a given time and pressure level.

    Parameters:
    -----------
    U, V : xarray.DataArray
        Wind components with dims ('time', 'plev', 'lat', 'lon')
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

    plt.show()

def plot_wind_streamlines(U, V, time_idx=0, level_value=85000, density=2, prj='PlateCarree'):
    """
    Plot streamlines from U and V wind components at a specific time and pressure level.

    Parameters:
    -----------
    U, V : xarray.DataArray
        Wind components with dims ('time', 'plev', 'lat', 'lon')
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
    plt.show()

def plot_geopotential_contours(Z, time_idx=0, level_value=50000, cmap='viridis', prj='PlateCarree'):
    """
    Plot geopotential height contours from CMIP5 data.

    Parameters:
    -----------
    Z : xarray.DataArray
        Geopotential (m^2/s^2) with dims ('time', 'plev', 'lat', 'lon')
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
    plt.show()