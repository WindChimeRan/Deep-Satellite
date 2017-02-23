import pandas as pd
import numpy as np
from scipy import misc
import tifffile as tiff
import pandas as pd
import numpy as np
import cv2
from shapely.wkt import loads as wkt_loads
from shapely import affinity
import matplotlib.pyplot as plt
inDir = './'
from os import listdir
imagenames_16 = listdir('./sixteen_band')
imagenames_13 = listdir('./three_band')


df = pd.read_csv('./train_wkt_v4.csv')
gs = pd.read_csv('./grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
df['ImageId'].unique()
trainImageIds = df.ImageId.unique()

def get_image_names(imageId):
    '''
    Get the names of the tiff files
    '''
    d = {'3': '{}/three_band/{}.tif'.format(inDir, imageId),
         'A': '{}/sixteen_band/{}_A.tif'.format(inDir, imageId),
         'get_image': '{}/sixteen_band/{}_M.tif'.format(inDir, imageId),
         'P': '{}/sixteen_band/{}_P.tif'.format(inDir, imageId),
         }
    return d


def stretch(bands):
    out = np.zeros_like(bands)
    for i in range(3):
        a = 0
        b = 255
        c = np.percentile(bands[:,:,i], 2)
        d = np.percentile(bands[:,:,i], 98)
        t = a + (bands[:,:,i] - c) * (b - a) / (d - c)
        t[t<a] = a
        t[t>b] = b
        out[:,:,i] =t
    return out


def get_image(image_id):
    filename = './three_band/'+image_id+'.tif'
    img = tiff.imread(filename)

    img = np.rollaxis(img, 0, 3)
    return img

def _get_image_names(base_path, imageId):
    '''
    Get the names of the tiff files
    '''
    d = {'3': path.join(base_path, 'three_band/{}.tif'.format(imageId)),  # (3, 3348, 3403)
         'A': path.join(base_path, 'sixteen_band/{}_A.tif'.format(imageId)),  # (8, 134, 137)
         'get_image': path.join(base_path, 'sixteen_band/{}_M.tif'.format(imageId)),  # (8, 837, 851)
         'P': path.join(base_path, 'sixteen_band/{}_P.tif'.format(imageId)),  # (3348, 3403)
         }
    return d


def _convert_coordinates_to_raster(coords, img_size, xymax):
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return (xmax, ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list, interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda,
                                      wkt_list_pandas):
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    return mask




for imageId in trainImageIds:
    for i in range(0, 2):
        img = get_image(imageId)
        img = stretch(img)
        set_of_mask = dict()
        mask_test = np.zeros(img.shape[0:2])
        mask = generate_mask_for_image_and_class(img.shape[0:2], imageId, i, gs, df)
        set_of_mask[i] = mask * 255 / 9 * i
        mask_test = mask_test + mask * 255 / 9 * i

    misc.imsave('../train_x/'+imageId+'.tif', img)
    misc.imsave('../train_y/'+imageId+'.tif', mask_test)


