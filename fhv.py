# -*- coding: utf-8 -*-
"""
    This script includes core functions for pre-processing and vulnerability assessment.
    
    - ConvertShapeToRaster(shp_fn, rst_fn, out_fn, fieldname, out_dtype=rasterio.int32)
    - GenerateRaster(fn_out, meta, data, new_dtype=False, new_nodata=False)
    - ReprojectRaster(inpath, outpath, new_crs)
    - CropRasterShape(rst_fn, shp_fn, out_fn, all_touched=False)

    Revised at Jan-20-2020
    Donghoon Lee (dlee298@wisc.edu)
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio import transform
from rasterio.warp import calculate_default_transform, reproject, Resampling
import fiona
import gdal
import xlrd
import re


def ConvertShapeToRaster(shp_fn, rst_fn, out_fn, fieldname, out_dtype=rasterio.int32):
    """Convert shapefile to a raster with reference raster
    """
    # Open the shapefile with GeoPandas
    unit = gpd.read_file(shp_fn)
    # Open the raster file as a template for feature burning using rasterio
    rst = rasterio.open(rst_fn)
    # Copy and update the metadata frm the input raster for the output
    profile = rst.profile.copy()
    profile.update(
        dtype=out_dtype,
        compress='lzw')
    # Before burning it, we need to 
    unit = unit.assign(ID_int = unit[fieldname].values.astype(out_dtype))
    # Burn the features into the raster and write it out
    with rasterio.open(out_fn, 'w+', **profile) as out:
        out_arr = out.read(1)
        shapes = ((geom, value) for geom, value in zip(unit.geometry, unit.ID_int))
        burned = rasterio.features.rasterize(shapes=shapes, fill=0, out=out_arr, 
                                             transform=out.transform,
                                             all_touched=False)
        out.write_band(1, burned)
    print('%s is saved' % out_fn)


def ValidCellToMap(data, valid, dtype='float32', nodata=-9999):
    """Convert values of valid cells to 2d Ndarray map format.
    """

    assert valid.sum() == data.shape[0]
    tmap = np.ones(valid.shape)*nodata
    tmap[valid] = data
    return tmap.astype(dtype)


def GenerateRaster(fn_out, meta, data, new_dtype=False, new_nodata=False):

    # New Dtype
    if new_dtype is not False:
        meta.update({'dtype': new_dtype})
    # New Nodata value
    if new_nodata is not False:
        meta.update({'nodata': new_nodata})
    # Write a raster
    with rasterio.open(fn_out, 'w+', **meta) as dst:
        dst.write_band(1, data)
        print('%s is saved.' % fn_out)



def ReprojectRaster(inpath, outpath, new_crs):
    """Reproject a raster with a specific crs
    """
    dst_crs = new_crs # CRS for web meractor 

    with rasterio.open(inpath) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(outpath, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
            print("%s is saved." % outpath)

    
    
def ReprojectToReference(in_path, ref_path, out_path, out_dtype, out_nodata=None):
    """Reproject a raster to reference raster
    """
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        profile.update(
            dtype=out_dtype,
            nodata=out_nodata,
            compress='lzw')
        with rasterio.open(in_path) as src:
            with rasterio.open(out_path, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst.transform,
                        dst_crs=dst.crs,
                        resampling=Resampling.nearest)

                print("%s is saved." % out_path)
    
    

def CropRasterShape(rst_fn, shp_fn, out_fn, all_touched=False):
    """crops raster with shapefile and save as new raster (GeoTiff)


    """
    # Get feature of the polygon (supposed to be a single polygon)
    with fiona.open(shp_fn, 'r') as shapefile:
        geoms = [feature['geometry'] for feature in shapefile]
    # Crop raster including cells over the lines (all_touched)
    with rasterio.open(rst_fn) as src:
        out_image, out_transform = mask(src, geoms, 
                                        crop=True, 
                                        all_touched=all_touched)
        out_meta = src.meta.copy()
    # Update spatial transform and height & width
    out_meta.update({'driver': 'GTiff',
                     'height': out_image.shape[1],
                     'width': out_image.shape[2],
                     'transform': out_transform})
    # Write the cropped raster
    with rasterio.open(out_fn, 'w', **out_meta) as dest:
        dest.write(out_image)
        print('%s is saved.' % out_fn)


def LoadCensusINEI(fn_census, fn_label):
    '''
    Read INEI 2017 National Census data (Excel) as Pandas dataframe format.
    Spanish labels are replaced by English labels.
    '''
# =============================================================================
#     #%% INPUT
#     fn = os.path.join('census', 'P08AFILIA.xlsx')
# =============================================================================
    
    # Read variable from Excel file
    df = pd.read_excel(fn_census, 
                       skiprows = 5,
                       header=0, 
                       index_col=1,
                       skipfooter=3)
    df = df.loc[:, ~df.columns.str.contains('^unnamed', case=False)]
    cols = df.columns
    
    # Update Spanish labels to English
    dfLabel = pd.read_excel(fn_label)
    # Find all rows of variable code and value
    rowCode = np.squeeze(np.where(dfLabel['Spanish'].str.match('Nombre :') == True))
    rowLabel = np.squeeze(np.where(dfLabel['Spanish'].str.match('Value Labels') == True))
    assert len(rowCode) == len(rowLabel)
    # Find a row of the target code
    code = re.split(r"/|.xlsx", fn_census)[-2]
    cond1 = dfLabel['Spanish'].str.match('Nombre : %s' % code)
    cond2 = dfLabel['Spanish'].str.len() == (9 + len(code))
    row = np.where(cond1 & cond2)[0][0]
    # Read both Spanish and English labels
    idx = rowCode.searchsorted(row)
    df2 = dfLabel.iloc[rowLabel[idx]+1:rowCode[idx+1]]
    label_spn = df2['Spanish'].apply(lambda x: x[x.find('. ')+2:])
    label_eng = df2['English'].apply(lambda x: x[x.find('. ')+2:])
    # Check the number of columns
    nlabel = len(label_spn)
    assert nlabel == np.in1d(cols, label_spn).sum()
    # Replace Spanish labels to English
    index = [np.where(label_spn == x)[0][0] for i, x in enumerate(cols[1:])]
    df.columns = ['District'] + list(label_eng.values[index])
    df.index.name='IDDIST'
    return df

    
def CorrectDistrict(dfCensus, method):
    # District map is not consistent with 2017 Census's districts.
    # 120604 (Mazamari) and 120606 (Pangoa) of census data are merged to
    # 120699 (MAZAMARI - PANGOA) of district map
    idMerg = [120604, 120606]    
    df = dfCensus.copy()
    if method == 'sum':
        df.loc[120699] = df.loc[idMerg].sum()
    elif method == 'average':
        df.loc[120699] = df.loc[idMerg].mean()
    elif method == 'min':
        df.loc[120699] = df.loc[idMerg].min()
    elif method == 'max':
        df.loc[120699] = df.loc[idMerg].max()
    df.loc[120699].District = 'Junín, Satipo, distrito de Mazamari-Pagoa'
    return df.drop(idMerg)


def TTimeCategory(array):
    '''
    Scale travel time to 1-8
    '''
    time = array[~np.isnan(array)]
    time[time < 30] = 0
    time[(30 <= time) & (time < 60)] = 1
    time[(60 <= time) & (time < 120)] = 2
    time[(120 <= time) & (time < 180)] = 3
    time[(180 <= time) & (time < 360)] = 4
    time[(360 <= time) & (time < 720)] = 5
    time[(720 <= time) & (time < 1440)] = 6
    time[(1440 <= time) & (time < 3000)] = 7
    time[time >= 3000] = 8
    array[~np.isnan(array)] = time
    return array


def censusToRaster(out_fn, meta, idmap, data):

# =============================================================================
#     #%% Input
#     out_fn = './census/test.tif'
#     idmap = did.copy()
#     data = page5.copy()
# =============================================================================
    
    # Change metadata
    meta['dtype'] = rasterio.float32
    idmap = idmap.astype(rasterio.float32)
    meta['nodata'] = -9999
    idmap[idmap == idmap[0,0]] = -9999

    # Compare IDs between census Dataframe and idMap
    listImap = np.unique(idmap[idmap != idmap[0,0]])
    listData = data.index.values
    assert len(listImap) == len(listData)
    
    # Distributes data    
    for i in listData:
        idmap[idmap == i] = data[i]
    
    # Write a raster
    with rasterio.open(out_fn, 'w', **meta) as dest:
        dest.write(idmap[None,:,:])
        print('%s is saved.' % out_fn)
        
        



    
    
def zeroToOne(array):
    '''
    Scale data from 0 to 1
    '''
    data = array[~np.isnan(array)]
    data = (data - data.min())/(data.max()-data.min())
    array[~np.isnan(array)] = data
    
    return array



    

def affectedGdpFlood(gdp, fdep):
    '''
    Calculated total affected GDP by flood levels
    '''
    
    gdpdata = gdp.copy(); depth = fdep.copy()
    
    depth[depth <= 30] = depth[depth <= 30]/30
    depth[depth > 30] = 1
    
    gdpAfft = np.sum(depth*gdpdata)
    
    return gdpAfft
    


    
    
    
    
    
    
### Functions for Bangladesh ===================================== ###
def bbsCensus(fn):
    '''
    From BBS Census 2011 Excel data to Pandas Dataframe
    '''
    xl = pd.ExcelFile(fn)
    df = xl.parse('Output')
    code = df['Code'].values.astype(int)[:,None]
    df = df.drop(['Code', 'Upazila/Thana Name'], axis=1)
    
    return code, df


def make_raster(in_ds, fn, data, data_type, nodata=None):
    """Create a one-band GeoTiff.

    in_ds     - datasource to copy projection and geotransform from
    fn        - path to the file to create
    data      - Numpy array containing data to archive
    data_type - output data type
    nodata    - optional NoData burn_values
    """

    driver = gdal.GetDriverByName('gtiff')
    out_ds = driver.Create(
        fn, in_ds.RasterXSize, in_ds.RasterYSize, 1, data_type)
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    out_band = out_ds.GetRasterBand(1)
    if nodata is not None:
        out_band.SetNoDataValue(nodata)
    out_band.WriteArray(data)
    out_band.FlushCache()
    #out_band.ComputerStaitstics(False)
    print('"{}" is printed.'.format(fn))
    return out_ds


def upazilaToTable(df, noi, column):
    """Extracts Upazila level vertical data
    """
    import numpy as np
    
#    noi = ('Male', 'Female'); column = 'B'
    codeUp = df.A.str.extract('(\d+)')
    codeUp = codeUp[~codeUp.isna().values].values.astype(int)
    ioi = df.A.str.startswith(noi, na=False)
    count = df.loc[ioi, column].values
    count = count.reshape([int(len(count)/len(noi)),len(noi)]).astype(int)
    count = count[:-1:,:]
    table = np.concatenate((codeUp, count), 1)
    
    return table
    

def valueToMap(value, code):
    '''
    Distribute value to Yes-Code region
    '''
    
    output = np.zeros(code.shape)
    output[code] = value
    
    return output


def evaluation(name, index, code4):
    
#    index = hous
    mask = (code4 == code4[0,0])
    core = index[~mask]
    print('{} max: {:.3f}, min: {:.3f}'.format(name, core.max(), core.min()))
    
def climInterpolate(clim, code4):
    
#    clim = prec.copy()
    
    x,y = np.where((clim == clim.min()) & (code4 != code4[0,0]))
    clim[x,y] = clim[clim != clim.min()].mean()
        
    return clim




#%%
# =============================================================================
# def censusToRaster(ds, fn, imap, data):
#     '''
#     Distributes district-level data to spatial map
#     '''
#     import os
#     imap = imap.copy()
#     
#     # Distributes data    
#     for i in range(len(data)):
#         imap[imap == data[i,0]] = data[i,1]
# 
#     # Save new raster
#     if not os.path.isfile(fn):
#         out_ds = make_raster(ds, fn, imap, gdal.GDT_Float64, imap[0,0])
#         del out_ds
# 
#     return imap    
# =============================================================================
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    