import dask.array as da
import os
import numpy as np

def load_data(data):
    data_npy = np.load(data).astype(np.double)
    data_dask = da.from_array(data_npy,chunks=data_npy.size)
       
    return data_npy, data_dask

def to_stack(p, d):

    x = da.stack(d, axis = 0)
    da.to_npy_stack(p, x, axis = 0)

    return x

def deleteAll(r): 
    for filename in os.listdir(r) :
        os.remove(r + "/" + filename)

    return 0

def main():

    data_sky = []
    data_dirty = []
    data_psf = []

    path = os.getcwd() + '/data'
    skyPath = os.getcwd() + '/skyStack'
    dirtyPath = os.getcwd() + '/dirtyStack'
    psfPath = os.getcwd() + '/psfStack'
     
    deleteAll(skyPath)
    deleteAll(dirtyPath)
    deleteAll(psfPath)
    
    n = 25
    for i in range(n):
        sky_npy, sky = load_data(path + '/sky.npy')
        dirty_npy, dirty = load_data(path + '/dirty.npy')
        psf_npy, psf = load_data(path + '/psf.npy')
        data_sky.append(sky)
        data_dirty.append(dirty)
        data_psf.append(psf)
   
    to_stack(skyPath, data_sky)
    to_stack(dirtyPath, data_dirty)
    to_stack(psfPath, data_psf)
    
    #y = da.from_npy_stack(sky)
    
main()