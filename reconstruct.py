import numpy as np
import numpy.ma as ma
from pyqstem.wave import Wave
import collections

def spatial_frequencies(shape, sampling):
    """Return spatial frequency at each pixel of image
    
    Args:
        shape (2 int): image shape
        sampling (2 float): image sampling
    """
    
    dkx=1/(shape[0]*sampling[0])
    dky=1/(shape[1]*sampling[1])

    if shape[0]%2==0:
        kx = np.fft.fftshift(dkx*np.arange(-shape[0]/2,shape[0]/2,1))
    else:
        kx = np.fft.fftshift(dkx*np.arange(-shape[0]/2-.5,shape[0]/2-.5,1))

    if shape[1]%2==0:
        ky = np.fft.fftshift(dky*np.arange(-shape[1]/2,shape[1]/2,1))
    else:
        ky = np.fft.fftshift(dky*np.arange(-shape[1]/2-.5,shape[1]/2-.5,1))

    ky,kx = np.meshgrid(ky,kx)

    k2 = kx**2+ky**2
    
    return kx, ky, k2

def reconstruct(images,ctfs,energy,sampling,tolerance=1e-6,cutoff=2,maxiter=50,epsilon=1e-12):
    
    """Reconstruct the exit-plane wavefunction from a series of 
    transmission eletronc microscopy (TEM) images and the assumed 
    corresponding contrast transfer functions(CTFs).
    
    Args:
        images (list of 2d arrays): series of TEM images
        ctfs (list of CTF objects): CTF's corresponding the images
        energy (float): electron energy in keV
        sampling (float): image sampling in Angstrom/pixel
        tolerance (float): mean squared error tolerance
        maxiter (int): maximum number of iterations
        cutoff (float): maximum spatial frequency reconstructed
        epsilon (float): additive constant for avoiding division by zero

    Returns:
        Wave object: Reconstructed exit wave

    """
    
    images=np.array(images)
    shape=images.shape
    
    if images.shape[0] != len(ctfs):
        raise RuntimeError('Number of images not matching number of CTFs')
    
    if not isinstance(sampling, collections.Iterable):
        sampling = (sampling,)*2
    
    ctf_arr=np.zeros(shape,dtype=np.complex128)
    inv_ctf_arr=np.zeros(shape,dtype=np.complex128)
    
    for i,ctf in enumerate(ctfs):
        ctf_arr[i] = ctf.copy().as_array(shape=shape[1:],sampling=sampling,energy=energy)
        inv_ctf = ctf_arr[i].copy()
        inv_ctf_arr[i] = 1/(inv_ctf+epsilon)
    
    if cutoff is not None:
        _,_,k2=spatial_frequencies(shape[1:], sampling)
        mask=k2>cutoff**2
        inv_ctf_arr = ma.masked_array(inv_ctf_arr,mask=np.tile(mask,(shape[0],)+(1,1)))
    
    A = np.sqrt(images)
    image_waves = A.copy().astype(np.complex128)
    
    MSEold=np.inf
    for i in range(maxiter):
        exit_waves = np.fft.fft2(image_waves) * inv_ctf_arr # propagate to exit plane
        mean_exit_wave = np.mean(exit_waves, axis=0)

        image_waves = mean_exit_wave * ctf_arr # propagate to image planes
        image_waves = np.fft.ifft2(image_waves)

        MSE = np.mean((A - np.abs(image_waves))**2)

        image_waves = A * np.exp(1.j * np.angle(image_waves))

        print("{0: < 3} : MSE = {1:.6f} [{2:.6f}] ".format(i,MSE,MSEold-MSE))

        if i > 0:
            if MSEold-MSE < tolerance:
                break
        MSEold=MSE

        if i >= maxiter:
            break

    mean_exit_wave=np.fft.ifft2(mean_exit_wave)

    return Wave(mean_exit_wave,energy,sampling)