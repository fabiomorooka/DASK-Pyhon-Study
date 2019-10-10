import os
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client
import time
import cProfile
import timeit
from dask import delayed
import dask

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

@delayed
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

@delayed
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

@delayed
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

@delayed
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

@delayed
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

@delayed
def wiener(data, aux, fr, L):
    l_del = delayed(L)
    return uirdft2((da.conj(delayed(fr)) * urdft2(data) + l_del * urdft2(aux)) * ((da.absolute(fr_npy)**2 + l_del)**-1))
    #Operateur '/' ne support division de deux delayed objects

@delayed(nout=2)
def load_data(data):
    data_npy = np.load(data).astype(np.double)
    data_dask = da.from_array(data_npy,chunks=data_npy.size)
    return data_npy, data_dask

def main():

    global fr_npy

    client = Client('tcp://192.168.43.199:8786')
    lamb = delayed(0.0005)

    sky_data = os.getcwd() + '/data/sky.npy'
    dirty_data = os.getcwd() + '/data/dirty.npy'
    psf_data = os.getcwd() + '/data/psf.npy'

    sky_npy, sky = load_data(sky_data)
    dirty_npy, dirty = load_data(dirty_data)
    psf_npy, psf = load_data(psf_data)
    
    fr_npy = ir2fr(psf_npy, sky.shape)
    fr = delayed(da.from_array)(fr_npy, psf.shape)
    #b = delayed(uirdft2)(fr.conj() * delayed(urdft2)(dirty))

    quad = wiener(dirty, delayed(da.zeros_like)(sky), fr, lamb)
    
    #quad.visualize()

    '''
    plt.figure(1)
    plt.clf()
    plt.plot(dirty.compute()[800], label='dirty')
    plt.plot(sky.compute()[800], label='true')
    #plt.plot(quad.compute()[800], label='quad')
    #plt.plot(hub[800], label='hub')
    plt.legend()
    
    plt.figure(2)
    plt.clf()
    plt.imshow(quad.compute(), vmin = da.min(quad.compute()), vmax = da.max(quad.compute()))
    plt.title('quad')
    '''
    plt.show()
    
for i in range(1):
    start_time = time.time()
    main()
    end_time = time.time()
    print(end_time - start_time)