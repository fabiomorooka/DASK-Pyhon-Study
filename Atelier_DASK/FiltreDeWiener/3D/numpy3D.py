import os
import time
import matplotlib.pyplot as plt
import numpy as np

try:
    from pyfftw.interfaces.numpy_fft import fftn
    from pyfftw.interfaces.numpy_fft import rfftn
    from pyfftw.interfaces.numpy_fft import irfftn
except ImportError:
    print("Installation of the pyfftw package improve preformance"
                 " by using fftw library.")
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
        raise ValueError("length of shape must be inferior to imp_resp.ndim")

    if not center:
        center = [int(np.floor(length / 2))
                  for length in imp_resp.shape[-len(shape):]]

    if len(center) != len(shape):
        raise ValueError("center and shape must have the same length")

    # Place the provided IR at the beginning of the array
    irpadded = np.zeros(imp_resp.shape[:len(shape) - 1] + shape)
    irpadded[tuple([slice(0, s) for s in imp_resp.shape])] = imp_resp

    # Roll, or circshift to place the origin at index 0, the
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

    return rfftn(inarray, axes=range(-ndim, 0), *args, **kwargs) / np.sqrt(
        np.prod(inarray.shape[-ndim:]))

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


    return irfftn(inarray, axes=range(-ndim, 0), *args, **kwargs) * np.sqrt(
        np.prod(inarray.shape[-ndim:-1]) * (inarray.shape[-1] - 1) * 2)

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

def wiener(data, aux, fr, L):
    return uirdft2((np.conj(fr) * urdft2(data) + L * urdft2(aux)) / (np.abs(fr)**2 + L))

def load_data(data, dim):
    return np.tile(np.load(data).astype(np.double)[np.newaxis], (dim, 1, 1))


def scheduling():

    global quad 

    #Parametres du code
    lamb = 0.0005 

    #Boucle pour faire le calcul de toutes les frequences du cube.
    fr = ir2fr(psf, shape=sky.shape[1:])
    quad = wiener(dirty, np.zeros_like(sky), fr, lamb)
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
        plt.imshow(quad[i], vmin = np.min(quad[i]), vmax = np.max(quad[i]))
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
    global n

    list_total = []
    list_load = []
    list_wiener = []
    n = 5

    for i in range (1):
        start_time1 = time.time()
        sky = load_data(os.path.split(os.path.split(os.getcwd())[0])[0] + '/sky.npy', n)
        dirty = load_data(os.path.split(os.path.split(os.getcwd())[0])[0] + '/dirty.npy', n)
        psf = load_data(os.path.split(os.path.split(os.getcwd())[0])[0] + '/psf.npy', n)
        end_time1 = time.time()
    
        start_time2 = time.time()
        scheduling()
        end_time2 = time.time()

        list_load.append(end_time1 - start_time1)
        list_wiener.append(end_time2 - start_time2)
        list_total.append(end_time2 - start_time1)	

    print('num de dimension: {}'.format(n))
    print('load time: {}'.format(round(sum(list_load)/len(list_load), 4)))
    print('compute time: {}'.format(round(sum(list_wiener)/len(list_wiener), 4)))
    print('total time: {}'.format(round(sum(list_total)/len(list_total), 4)))
	
measuring()