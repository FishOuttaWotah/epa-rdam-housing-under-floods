from typing import Sequence
import pandas as pd
import numpy as np
import rasterio
# import rasterio.mask, rasterio.features, rasterio.plot


def read_CBS_excel(filepath:str, index_name:str, convert_to_Int64:bool=False):
    # reads excel file from Rotterdam Onderwijs010 site and processes
    dataframe = pd.read_excel(filepath, header=1,skipfooter=7,
                              index_col=0, na_values=['-'])
    dataframe.index.name = index_name
    if convert_to_Int64:
        dataframe = dataframe.astype('Int64')
    return dataframe

def get_tile_images(image, width=8, height=8):
    """
    Converts images from 2d to 3d format, with the tile format as flat 1d
    Credits to ChoF for function:
    https://stackoverflow.com/questions/48482317/slice-an-image-into-tiles-using-numpy
    :param image:
    :param width:
    :param height:
    :return:
    """
    _nrows, _ncols, depth = image.shape
    _size = image.size
    _strides = image.strides

    nrows, _m = divmod(_nrows, height)
    ncols, _n = divmod(_ncols, width)
    if _m != 0 or _n != 0:
        return None

    return np.lib.stride_tricks.as_strided(
        np.ravel(image),
        shape=(nrows, ncols, height, width, depth),
        strides=(height * _strides[0], width * _strides[1], *_strides),
        writeable=False
    )

def reconvert_np_to_rasterio_dataset(raster, transform, count, **kwargs):
    """
    Derived from solution written by Luna:
    https://gis.stackexchange.com/questions/329434/creating-an-in-memory-rasterio-dataset-from-numpy-array
    Wrapper for rasterio.mask.mask to allow for in-memory processing.

    Docs: https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html

    Args:
        raster (numpy.ndarray): raster to be masked with dim: [H, W]
        transform: affine transform object
        count: bands to record
        shapes, **kwargs: passed to rasterio.mask.mask

    Returns:
        masked: numpy.ndarray or numpy.ma.MaskedArray with dim: [H, W]
    """
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=raster.shape[0],
            width=raster.shape[1],
            count=count,
            dtype=raster.dtype,
            transform=transform,
        ) as dataset:
            dataset.write(raster, 1)
        # with memfile.open() as dataset:
        #     output, _ = rasterio.mask.mask(dataset, shapes, **kwargs)
        return memfile.open()

def clean_flood_map_with_topography(flood_map_path,
                   terrain_raster):
    # uses topography to clean the flood map


    return
