"""
Kazakhstan Grid Transformation Module

This module creates NTv2 grid files for coordinate transformation
between Pulkovo 1942 (Krassovsky ellipsoid) and WGS84 coordinate systems
for the territory of Kazakhstan.

Author: Grid Processing Team
Date: October 2025
"""

import os
import numpy as np
import pandas as pd
import struct
import xarray as xr
from pyproj import Proj, CRS, Transformer
from osgeo import gdal, osr
from datetime import datetime as dt
from datetime import date
import tempfile
import pygmt

def create_netcdf_grids(df, inc):
    """
    Create NetCDF grid files using PyGMT for longitude and latitude shifts.
    
    This function processes transformation data and creates three NetCDF files:
    - Longitude shift grid
    - Latitude shift grid  
    - Zero grid for accuracy values
    
    Args:
        df (pd.DataFrame): DataFrame containing transformation data with columns:
                          'lon_src', 'lat_src', 'x_shift', 'y_shift'
        inc (float): Grid spacing in degrees
        
    Returns:
        tuple: Paths to created NetCDF files (dlon_grid, dlat_grid, zero_grid)
    """

    # Output file names for NetCDF grids
    out_dlon = 'dlon_grid.nc'  # Longitude shift grid
    out_dlat = 'dlat_grid.nc'  # Latitude shift grid  
    out_zero = 'zero_grid.nc'  # Zero grid for accuracy values
   
    # Calculate processing region from target coordinates
    region = df['lon_trg'].min(), df['lon_trg'].max(), df['lat_trg'].min(), df['lat_trg'].max()
    spacing = f'{inc}/{inc}'
    
    print(f"  DEBUG: Processing region: {region}")
    print(f"  DEBUG: Grid spacing: {spacing}")
    print(f"  DEBUG: Input data points: {len(df)}")
    
    # Prepare data for PyGMT processing
    lon = df['lon_src'].values  # Source longitude coordinates
    lat = df['lat_src'].values  # Source latitude coordinates  
    dlat = df['y_shift'].values  # Latitude shifts in arc seconds
    dlon = -df['x_shift'].values  # Inverted longitude shifts for NTv2 format
    
    print(f"  DEBUG: Longitude shift range: {dlon.min():.3f} to {dlon.max():.3f} arcsec")
    print(f"  DEBUG: Latitude shift range: {dlat.min():.3f} to {dlat.max():.3f} arcsec")
   
    print("  Processing longitude shift data...")
    # Block median filter for longitude shifts to remove outliers
    blocks_lon = pygmt.blockmedian(
        region=region,
        x=lon,
        y=lat,
        z=dlon,
        spacing=spacing,
    )
    
    print(f"  DEBUG: Longitude blockmedian output points: {len(blocks_lon)}")
    
    # Create smooth surface grid for longitude shifts       
    pygmt.surface(
        data=blocks_lon,
        region=region,
        spacing=spacing,
        outgrid=out_dlon
    )
    
    print("  Processing latitude shift data...")
    # Block median filter for latitude shifts
    blocks_lat = pygmt.blockmedian(
        region=region,
        x=lon,
        y=lat,
        z=dlat,
        spacing=spacing,
    )
    
    print(f"  DEBUG: Latitude blockmedian output points: {len(blocks_lat)}")
    
    # Create smooth surface grid for latitude shifts
    pygmt.surface(
        data=blocks_lat,
        region=region,
        spacing=spacing,
        outgrid=out_dlat
    )
    
    print("  Creating zero accuracy grid...")
    # Create zero grid for accuracy values (required by NTv2 format)
    zero_data = np.zeros_like(dlat)
    pygmt.xyz2grd(
        data=np.column_stack([lon, lat, zero_data]),
        region=region,
        spacing=spacing,
        outgrid=out_zero
    )
    
    print(f"  SUCCESS: Created NetCDF grids:")
    print(f"    - Longitude shifts: {out_dlon}")
    print(f"    - Latitude shifts: {out_dlat}")
    print(f"    - Zero accuracy: {out_zero}")

    return out_dlon, out_dlat, out_zero

def create_ntv2_binary(output_path, dlon, dlat, zero, grid_name="FAKE_GRID", cutline_path=None):
    """
    Create a binary NTv2 format file from NetCDF grids using GDAL.
    
    This function combines longitude shift, latitude shift, and accuracy grids
    into a single NTv2 binary file that can be used by coordinate transformation
    software like PROJ and pyproj.
    
    Args:
        output_path (str): Path for output NTv2 file (.gsb extension)
        dlon (str): Path to longitude shift NetCDF grid
        dlat (str): Path to latitude shift NetCDF grid  
        zero (str): Path to zero accuracy NetCDF grid
        grid_name (str): Name identifier for the grid
        cutline_path (str, optional): Path to cutline shapefile for clipping
        
    Returns:
        str: Path to the created NTv2 file
    """

    print(f"Creating NTv2 binary file: {output_path}")
    print(f"  DEBUG: Input grids:")
    print(f"    - Longitude shifts: {dlon}")
    print(f"    - Latitude shifts: {dlat}")  
    print(f"    - Zero accuracy: {zero}")
    
    # Create VRT (Virtual Dataset) combining all input grids as separate bands
    print("  Creating VRT file...")
    vrt_options = gdal.BuildVRTOptions(separate=True)
    vrt = gdal.BuildVRT('temp.vrt', [dlat, dlon, zero, zero], options=vrt_options)
    
    if vrt is None:
        raise RuntimeError("Failed to create VRT file")
    
    print("  DEBUG: VRT file created successfully")
    
    # Convert VRT to NTv2 binary format with metadata
    print("  Converting to NTv2 format...")
    creation_date = date.today().strftime("%Y%m%d")
    
    gdal.Translate(
        output_path,
        vrt,
        format='NTv2',
        creationOptions=["TILED=YES"],
        metadataOptions=[
            f'VERSION=v0.1',
            'SYSTEM_F=Pulkovo42',  # Source coordinate system
            'SYSTEM_T=GRS80',      # Target coordinate system  
            'MAJOR_F=6378245.0',   # Source ellipsoid major axis
            'MINOR_F=6356863.0',   # Source ellipsoid minor axis
            'MAJOR_T=6378137',     # Target ellipsoid major axis
            'MINOR_T=6356752.314', # Target ellipsoid minor axis
            f'SUB_NAME={grid_name}',
            f'CREATED={creation_date}',
            f'UPDATED={creation_date}'
        ]
    )

    result_filename = output_path
    
    print(f"  DEBUG: Base NTv2 file created: {output_path}")

    # Apply cutline clipping if specified
    if cutline_path:
        print(f"  Applying cutline clipping: {cutline_path}")
        result_filename = f'clipped_{os.path.basename(output_path)}'
        gdal.Warp(
            destNameOrDestDS=result_filename,
            srcDSOrSrcDSTab=output_path,
            cutlineDSName=cutline_path,
            cropToCutline=True,
            dstNodata=0,
        )
        print(f"  DEBUG: Clipped file created: {result_filename}")
    
    # Check final file size and existence
    if os.path.exists(result_filename):
        file_size = os.path.getsize(result_filename)
        print(f"  SUCCESS: NTv2 file created - {result_filename} ({file_size:,} bytes)")
    else:
        raise RuntimeError(f"Failed to create NTv2 file: {result_filename}")

    return result_filename

def write_ntv2_grid(df, inc, output_path, grid_name="FAKE_GRID"):
    """
    Main function to create NTv2 grid file from transformation data.
    
    This function orchestrates the complete process:
    1. Creates NetCDF grids using PyGMT
    2. Converts to binary NTv2 format using GDAL
    
    Args:
        df (pd.DataFrame): Transformation data with coordinate pairs and shifts
        inc (float): Grid spacing in degrees
        output_path (str): Path for output NTv2 file
        grid_name (str): Identifier name for the grid
        
    Returns:
        str: Path to the created NTv2 file
    """
    print(f"Starting NTv2 grid creation process...")
    print(f"  DEBUG: Input data shape: {df.shape}")
    print(f"  DEBUG: Grid spacing: {inc} degrees")
    print(f"  DEBUG: Output path: {output_path}")
   
    # Step 1: Create NetCDF grids using PyGMT
    print("Step 1: Creating NetCDF grids with PyGMT...")
    dlon, dlat, zero = create_netcdf_grids(df, inc)

    # Step 2: Convert NetCDF grids to binary NTv2 format
    print("Step 2: Converting to NTv2 binary format...")
    result_file = create_ntv2_binary(output_path, dlon, dlat, zero, grid_name)
    
    print(f"SUCCESS: NTv2 grid creation completed - {result_file}")

    return result_file

def validate_grid_with_pyproj(df, grid_file_path):
    """
    Validate the created NTv2 grid by comparing transformations.
    
    This function compares coordinate transformations using:
    1. Direct Helmert parameters (reference)
    2. NTv2 grid file (test)
    
    Args:
        df (pd.DataFrame): Original transformation data for comparison
        grid_file_path (str): Path to the NTv2 grid file to validate
        
    Returns:
        tuple: Mean differences in longitude and latitude (degrees)
    """
    print(f"Validating NTv2 grid: {grid_file_path}")
    print(f"  DEBUG: Validation data points: {len(df)}")
    
    # Transform using direct Helmert parameters (reference method)
    print("  Computing reference transformations using Helmert parameters...")
    lon_helm, lat_helm = Transformer.from_crs(
        CRS.from_dict({'proj': 'longlat', 'ellps': 'GRS80', 'datum': 'WGS84'}),
        CRS.from_dict({
            'proj': 'latlong',
            'ellps': 'krass',
            'towgs84': '25,-141,-78.5,0,-0.35,-0.736,0',
        }),
    ).transform(df['lon_trg'].values, df['lat_trg'].values)
    
    print(f"  DEBUG: Helmert transformation completed")
    print(f"    - Longitude range: {lon_helm.min():.6f} to {lon_helm.max():.6f}")
    print(f"    - Latitude range: {lat_helm.min():.6f} to {lat_helm.max():.6f}")
    
    # Transform using NTv2 grid file (test method)
    print("  Computing grid-based transformations...")
    try:
        lon_grid, lat_grid = Transformer.from_crs(
            CRS.from_dict({'proj': 'longlat', 'ellps': 'GRS80', 'datum': 'WGS84'}),
            CRS.from_dict({'proj': 'longlat', 'ellps': 'krass', 'nadgrids': grid_file_path}),
        ).transform(df['lon_trg'].values, df['lat_trg'].values)
        
        print(f"  DEBUG: Grid transformation completed")
        print(f"    - Longitude range: {lon_grid.min():.6f} to {lon_grid.max():.6f}")
        print(f"    - Latitude range: {lat_grid.min():.6f} to {lat_grid.max():.6f}")
        
    except Exception as e:
        print(f"  WARNING: Grid transformation failed: {e}")
        return float('inf'), float('inf')

    # Calculate differences between methods
    lon_diff = lon_helm - lon_grid
    lat_diff = lat_helm - lat_grid
    
    print(f"  DEBUG: Difference statistics:")
    print(f"    - Longitude differences: mean={lon_diff.mean():.2e}, std={lon_diff.std():.2e}")
    print(f"    - Latitude differences: mean={lat_diff.mean():.2e}, std={lat_diff.std():.2e}")
    print(f"    - Max longitude difference: {abs(lon_diff).max():.2e}")
    print(f"    - Max latitude difference: {abs(lat_diff).max():.2e}")
    
    return lon_diff.mean(), lat_diff.mean()
 

def make_fake_grid(grid_path, inc=0.05, region=(46.475, 87.325, 40.525, 55.475)):
    """
    Create a synthetic NTv2 grid for Kazakhstan coordinate transformations.
    
    This function generates a complete NTv2 grid file for transforming coordinates
    between Pulkovo 1942 (Krassovsky ellipsoid) and WGS84 systems across Kazakhstan.
    
    Args:
        grid_path (str): Output path for the NTv2 grid file (.gsb extension)
        inc (float): Grid spacing in degrees (default: 0.05° = ~5.5 km)
        region (tuple): Coverage area as (min_lon, max_lon, min_lat, max_lat)
                       Default covers Kazakhstan territory
    
    Returns:
        bool: True if grid creation and validation successful, False otherwise
    """
    print(f"=" * 80)
    print(f"CREATING KAZAKHSTAN NTv2 TRANSFORMATION GRID")
    print(f"=" * 80)
    print(f"Grid parameters:")
    print(f"  Coverage area: {region[0]:.6f}° to {region[1]:.6f}° longitude")
    print(f"                 {region[2]:.6f}° to {region[3]:.6f}° latitude")
    print(f"  Grid spacing: {inc:.6f}° (~{inc*111:.1f} km at equator)")
    print(f"  Output file: {grid_path}")
    
    # Extract grid parameters
    min_lon, max_lon, min_lat, max_lat = region
    
    # Calculate expected grid dimensions
    n_lon = int((max_lon - min_lon) / inc) + 1
    n_lat = int((max_lat - min_lat) / inc) + 1
    total_points = n_lon * n_lat
    
    print(f"  Expected dimensions: {n_lon} × {n_lat} = {total_points:,} points")
    
    # Generate coordinate grid
    print(f"Generating coordinate grid...")
    lons = []
    lats = []
    
    for lon in np.arange(min_lon, max_lon + inc, inc):
        for lat in np.arange(min_lat, max_lat + inc, inc):
            lons.append(lon)
            lats.append(lat)

    print(f"  DEBUG: Generated {len(lons):,} coordinate points")

    # Create DataFrame with target coordinates (WGS84)
    df = pd.DataFrame({
        'lon_trg': lons,  # Target longitude (WGS84)
        'lat_trg': lats,  # Target latitude (WGS84)
    })

    print(f"Computing coordinate transformations...")
    print(f"  Transform: WGS84 → Pulkovo 1942 (Krassovsky)")
    print(f"  Helmert parameters: [25, -141, -78.5, 0, -0.35, -0.736, 0]")
    
    # Transform from WGS84 to Pulkovo using Helmert parameters
    df['lon_src'], df['lat_src'] = Transformer.from_crs(
        CRS.from_dict({
            'proj': 'latlong',
            'ellps': 'GRS80',
            'datum': 'WGS84',
        }),
        CRS.from_dict({
            'proj': 'latlong',
            'ellps': 'krass',
            'towgs84': '25,-141,-78.5,0,-0.35,-0.736,0',
        }),
    ).transform(df['lon_trg'].values, df['lat_trg'].values)

    # Calculate coordinate shifts in arc seconds
    df['x_shift'] = (df['lon_trg'] - df['lon_src']) * 3600.0  # Longitude shift (arcsec)
    df['y_shift'] = (df['lat_trg'] - df['lat_src']) * 3600.0  # Latitude shift (arcsec)

    print(f"  DEBUG: Transformation statistics:")
    print(f"    - Longitude shifts: {df['x_shift'].min():.3f} to {df['x_shift'].max():.3f} arcsec")
    print(f"    - Latitude shifts: {df['y_shift'].min():.3f} to {df['y_shift'].max():.3f} arcsec")
    print(f"    - Mean longitude shift: {df['x_shift'].mean():.3f} arcsec")
    print(f"    - Mean latitude shift: {df['y_shift'].mean():.3f} arcsec")

    # Create NTv2 grid file
    print(f"Creating NTv2 grid file...")
    result = write_ntv2_grid(df, inc, grid_path, grid_name="FAKE_GRID_DIRECT")

    # Validate the created grid
    print(f"Validating created grid...")
    lon_diff, lat_diff = validate_grid_with_pyproj(df, grid_path)
    lon_diff_sec = lon_diff * 3600
    lat_diff_sec = lat_diff * 3600

    # Check validation results
    tolerance = 1e-2  # Very small tolerance for coordinate differences
    success = (abs(lon_diff_sec) < tolerance) and (abs(lat_diff_sec) < tolerance)
    
    print(f"=" * 80)
    print(f"GRID CREATION SUMMARY")
    print(f"=" * 80)
    print(f"Grid file: {result}")
    print(f"Validation differences:")
    print(f"  - Longitude: {lon_diff_sec:.3f} arcsecs")
    print(f"  - Latitude: {lat_diff_sec:.3f} arcsecs")
    print(f"Status: {'SUCCESS' if success else 'WARNING - Large differences detected'}")
    print(f"=" * 80)

    return success

if __name__ == "__main__":
    """
    Main execution block for standalone grid creation.
    
    Creates a Kazakhstan NTv2 transformation grid and validates its accuracy.
    """
    print("Kazakhstan NTv2 Grid Creator")
    print("Coordinate System Transformation: Pulkovo 1942 ↔ WGS84")
    print()
    
    # Define output file path
    grid_file_path = os.path.join(os.path.dirname(__file__), 'fake_grid.gsb')
    
    # Create the grid
    result = make_fake_grid(grid_file_path)

    # Final status
    if result:
        print("✓ Grid creation completed successfully!")
    else:
        print("⚠ Grid creation completed with warnings - check validation results")
    
    print(f"Grid file location: {grid_file_path}")