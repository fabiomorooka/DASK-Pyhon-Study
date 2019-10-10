import os
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client
import time
import dask
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler, CacheProfiler, visualize

try:
    from dask.array.fft import rfftn as dask_rfftn
    from dask.array.fft import irfftn as dask_irfftn
    from pyfftw.interfaces.numpy_fft import fftn
    from pyfftw.interfaces.numpy_fft import rfftn
    from pyfftw.interfaces.numpy_fft import irfftn    
except ImportError:
    print("Installation of the dask.array and/or pyfftw package improve preformance"
           " by using fft/fftw library.")
    from dask.array.fft import rfftn as dask_rfftn
    from dask.array.fft import irfftn as dask_irfftn
    from numpy.fft import fftn as fftn
    from numpy.fft import rfftn as rfftn
    from numpy.fft import irfftn as irfftn   
 

def ir2fr(imp_resp, shape, center=None, real=True):
    """Return the frequency response from impulsionnal responses

    This function make the necessary correct zero-padding, zero
    convention, correct DFT etc. to compute the frequency response
    from impulsionnal responses (IR).

    The IR array is supposed to have the origin in the middle of the
    array.

    The Fourier transform is performed on the last `len(shape)`
    dimensions.

    Parameters
    ----------
    imp_resp : ndarray
       The impulsionnal responses.

    shape : tuple of int
       A tuple of integer corresponding to the target shape of the
       frequency responses, without hermitian property.

    center : tuple of int, optional
       The origin index of the impulsionnal response. The middle by
       default.

    real : boolean (optionnal, default True)
       If True, imp_resp is supposed real, the hermissian property is
       used with rfftn DFT and the output has `shape[-1] / 2 + 1`
       elements on the last axis.

    Returns
    -------
    y : ndarray
       The frequency responses of shape `shape` on the last
       `len(shape)` dimensions.

    Notes
    -----
    - For convolution, the result have to be used with unitary
      discrete Fourier transform for the signal (udftn or equivalent).
    - DFT are always peformed on last axis for efficiency.
    - Results is always C-contiguous.

    See Also
    --------
    udftn, uidftn, urdftn, uirdftn
    """
    if len(shape) > imp_resp.ndim:
        raise ValueError("length of shape must inferior to imp_resp.ndim")

    if not center:
        center = [int(np.floor(length / 2)) for length in imp_resp.shape]

    if len(center) != len(shape):
        raise ValueError("center and shape must have the same length")

    # Place the provided IR at the beginning of the array
    irpadded = np.zeros(shape)
    irpadded[tuple([slice(0, s) for s in imp_resp.shape])] = imp_resp

    # Roll, or circshift to place the origin at 0 index, the
    # hypothesis of the DFT
    for axe, shift in enumerate(center):
        irpadded = np.roll(irpadded, -shift,
                           imp_resp.ndim - len(shape) + axe)

    # Perform the DFT on the last axes
    if real:
        return np.ascontiguousarray(rfftn(
            irpadded, axes=list(range(imp_resp.ndim - len(shape),
                                      imp_resp.ndim))))
    else:
        return np.ascontiguousarray(fftn(
            irpadded, axes=list(range(imp_resp.ndim - len(shape),
                                      imp_resp.ndim))))

def urdftn(inarray, ndim=None, *args, **kwargs):
    """N-dim real unitary discrete Fourier transform

    This transform consider the Hermitian property of the transform on
    real input

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    ndim : int, optional
        The `ndim` last axis along wich to compute the transform. All
        axes by default.

    Returns
    -------
    outarray : array-like (the last ndim as  N / 2 + 1 lenght)
    """
    if not ndim:
        ndim = inarray.ndim

    return dask_rfftn(inarray, axes=range(-ndim, 0), *args, **kwargs) / da.sqrt(
    	da.prod(da.asarray(inarray.shape[-ndim:])))

def urdft2(inarray, *args, **kwargs):
    """2-dim real unitary discrete Fourier transform

    Compute the real discrete Fourier transform on the last 2 axes. This
    transform consider the Hermitian property of the transform from
    complex to real real input.

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    Returns
    -------
    outarray : array-like (the last dim as (N - 1) *2 lenght)

    See Also
    --------
    udft2, udftn, urdftn
    """
    return urdftn(inarray, 2, *args, **kwargs)

def uirdftn(inarray, ndim=None, *args, **kwargs):
    """N-dim real unitary discrete Fourier transform

    This transform consider the Hermitian property of the transform
    from complex to real real input.

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    ndim : int, optional
        The `ndim` last axis along wich to compute the transform. All
        axes by default.

    Returns
    -------
    outarray : array-like (the last ndim as (N - 1) * 2 lenght)
    """
    if not ndim:
        ndim = inarray.ndim

    return dask_irfftn(inarray, axes=range(-ndim, 0), *args, **kwargs) * da.sqrt(
       da.prod(da.asarray(inarray.shape[-ndim:-1])) * (inarray.shape[-1] - 1) * 2)

def uirdft2(inarray, *args, **kwargs):
    """2-dim real unitary discrete Fourier transform

    Compute the real inverse discrete Fourier transform on the last 2 axes.
    This transform consider the Hermitian property of the transform
    from complex to real real input.

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    Returns
    -------
    outarray : array-like (the last ndim as (N - 1) *2 lenght)

    See Also
    --------
    urdft2, uidftn, uirdftn
    """
    return uirdftn(inarray, 2, *args, **kwargs)

def wiener(data, aux, fr, fr_npy, L):
    return uirdft2((da.conj(fr) * urdft2(data) + L * urdft2(aux)) / (da.absolute(fr_npy)**2 + L))

def load_data(data, n):
    data_npy = np.load(data).astype(np.double)
    data_dask = da.from_array(data_npy,chunks=data_npy.size)
    data_ndim = []
    for i in range(n):
        data_ndim.append(data_dask)
    data = da.stack(data_ndim, axis = 0)
    
    return data_npy, data

def scheduling():

    global quad
    
    #Parametres du code
    lamb = 0.0005
    quad_freq = []
    
    #Boucle pour faire le calcul de toutes les frequences du cube.
    for i in range(dirty.shape[0]):
        fr_npy = ir2fr(psf[i], sky[i].shape)
        fr = da.from_array(fr_npy, chunks = psf[i].shape)
        quad_freq.append(wiener(dirty[i], da.zeros_like(sky[i]), fr, fr_npy, lamb))

    quad = da.stack(quad_freq, axis=0)
    #quad.visualize()    
    
def show_courbes():
    
    for i in range(len(dirty)):
        plt.figure(i + 1)
        plt.clf()
        plt.plot((dirty[i])[800], label='dirty')
        plt.plot((quad[i])[800], label='quad')
        plt.plot((sky[i])[800], label='true')
        plt.legend()
    plt.show(block=False)
    while(plt.fignum_exists(1)):
        try:
            plt.pause(100000)
            plt.close("all")
        except:
            break

def show_images():

    for i in range(len(dirty)):
        plt.figure(i+1)
        plt.clf()
        plt.imshow(quad[i], vmin = da.min(quad[i]), vmax = da.max(quad[i]))
        plt.title('quad' + str(i))
    plt.show(block=False)
    while(plt.fignum_exists(1)):
        try:
            plt.pause(100000)
            plt.close("all")
        except:
            break

def measuring():

    global sky
    global dirty
    global psf
     
    list_schedule = []
    list_compute = []
    list_total = []
    list_load = []
    n = 5
    
    for i in range (1):
        start_time1 = time.time()
        #sky_npy, sky = load_data(os.path.split(os.path.split(os.getcwd())[0])[0] + '/sky.npy', n)
        #dirty_npy, dirty = load_data(os.path.split(os.path.split(os.getcwd())[0])[0] + '/dirty.npy', n)
        #psf_npy, psf = load_data(os.path.split(os.path.split(os.getcwd())[0])[0] + '/psf.npy', n)
        sky = da.from_npy_stack(os.getcwd() + '/skyStack')
        dirty = da.from_npy_stack(os.getcwd() + '/dirtyStack')
        psf = da.from_npy_stack(os.getcwd() + '/psfStack')
        end_time1 = time.time()
        
        start_time2 = time.time()
        scheduling()
        end_time2 = time.time()
	
        #pbar = ProgressBar()
        #with pbar, Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler() as cprof:	
        start_time3 = time.time()        
        quad.compute()
        end_time3 = time.time()
        
        list_load.append(end_time1 - start_time1)
        list_schedule.append(end_time2 - start_time2)
        list_compute.append(end_time3 - start_time3)
        list_total.append(end_time3 - start_time1)
	#visualize([prof, rprof, cprof])
    
    print('num de dimension: {}'.format(n))
    print('load time: {}'.format(round(sum(list_load)/len(list_load), 4)))
    print('scheduling time: {}'.format(round(sum(list_schedule)/len(list_schedule), 4)))
    print('compute time: {}'.format(round(sum(list_compute)/len(list_compute), 4)))
    print('total time: {}'.format(round(sum(list_total)/len(list_total), 4)))
	
measuring()