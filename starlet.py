import numpy as np
from scipy.signal import convolve2d

def b3spline_fast(step_hole):
    """This function returns 2D B3-spline kernel for the 'a trou' algorithm.
    INPUT:  step_hole, non-negative integer(number of holes)
    OUTPUT: 2D numpy array """
    step_hole = int(step_hole)
    c1 = 1./16
    c2 = 1./4
    c3 = 3./8
    length = 4*step_hole+1
    kernel1d = np.zeros((1,length))
    kernel1d[0,0] = c1
    kernel1d[0,-1] = c1
    kernel1d[0,step_hole] = c2
    kernel1d[0,-1-step_hole] = c2
    kernel1d[0,2*step_hole] = c3
    kernel2d = np.dot(kernel1d.T,kernel1d)
    return kernel2d

def soft_thresh(signal, threshold):
    """This function returns the result of a soft thresholding operation.
    INPUT: signal, Numpy Array
           threshold, Numpy Array
    OUTPUT: res, Numpy Array"""
    res = np.sign(signal) * (np.abs(signal) - threshold) * (np.abs(signal) >= threshold)
    return res

def MS_soft_thresh(wave_coef, n_sigma):
    """This function returns the result of a multi-scale soft thresholding
    operation perfromed on wave_coef and using the coefficients of n_sigma as
    thresholds.
    INPUT: wave_coef, Numpy Array
           n_sigma, Numpy Array
    OUTPUT: wave_coef_rec_MS, Numpy Array"""
    wave_coef_rec_MS = np.zeros(wave_coef.shape)
    for i,wave in enumerate(wave_coef):
        # Denoise image
        wave_coef_rec_MS[i,:,:] = soft_thresh(wave, n_sigma[i])
    return wave_coef_rec_MS

def star2d(im,scale,gen2=True):
    """This function returns the starlet transform of an image.
    INPUT:  im, 2D numpy array
            scale, positive integer (number of scales)
            gen2, boolean (to select the starlets generation)
    OUTPUT: 3D numpy array """
    (nx,ny) = np.shape(im)
    nz = scale
    wt = np.zeros((nz,nx,ny))
    step_hole = 1
    im_in = np.copy(im)

    for i in np.arange(nz-1):
        kernel2d = b3spline_fast(step_hole)
        im_out = convolve2d(im_in, kernel2d, boundary='symm',mode='same')

        if gen2:
            im_aux = convolve2d(im_out, kernel2d, boundary='symm',mode='same')
            wt[i,:,:] = im_in - im_aux
        else:        
            wt[i,:,:] = im_in - im_out

        im_in = np.copy(im_out)
        step_hole *= 2

    wt[nz-1,:,:] = np.copy(im_out)

    return wt

def istar2d(wtOri,gen2=True):
    """This function reconstructs the image from its starlet transformation.
    INPUT:  wtOri, 3D numpy array
            gen2, boolean (to precise the starlets generation)
    OUTPUT: 3D numpy array """
    (nz,nx,ny) = np.shape(wtOri)
    wt = np.copy(wtOri)
    if gen2:
        '''
        h' = h, g' = Dirac
        '''
        step_hole = pow(2,nz-2)
        imRec = np.copy(wt[nz-1,:,:])
        for k in np.arange(nz-2,-1,-1):
            kernel2d = b3spline_fast(step_hole)
            im_out = convolve2d(imRec, kernel2d, boundary='symm',mode='same')
            imRec = im_out + wt[k,:,:]
            step_hole /= 2            
    else:
        '''
        h' = h, g' = Dirac + h
        '''
        imRec = np.copy(wt[nz-1,:,:])
        step_hole = pow(2,nz-2)
        for k in np.arange(nz-2,-1,-1):
            kernel2d = b3spline_fast(step_hole)
            imRec = convolve2d(imRec, kernel2d, boundary='symm',mode='same')
            im_out = convolve2d(wt[k,:,:], kernel2d, boundary='symm',mode='same')
            imRec += wt[k,:,:]+im_out
            step_hole /= 2
            
    return imRec

def conv(image, kernel):
    
    return convolve2d(image, kernel, mode='same')

def ST(image, beta, sigma, psf):
    
    scl = int(np.ceil(np.log2(np.sqrt(image.size))))
    shape = np.array(image.shape)
    dirac = np.zeros(shape)
    dirac[shape[0]//2, shape[1]//2] = 1.
    scales = star2d(dirac, scale=scl)
    std_scales = np.array([np.linalg.norm(conv(scale, psf),'fro') for scale in scales])
  
    sigma_scales = sigma * std_scales

    # Set k for k-sigma thresholding
    k = 3

    # (k+plus) for the finest scale
    plus = 0

    # build thresholds
    thresholds = np.array([(k+plus)*sigma_scales[:1], k*sigma_scales[1:]])

    # Starlet transform of image
    alpha = star2d(image, scale=scl)

    # Multiscale threshold except coarse scale
    alpha[:-1] = MS_soft_thresh(alpha[:-1], beta*thresholds[-1])

    # Apply the adjoint of the starlets on alpha
    return istar2d(alpha), thresholds, scales

def ST_radio(image, beta, sigma, psf):
    
    scl = int(np.ceil(np.log2(np.sqrt(image.size))))
    shape = np.array(image.shape)
    dirac = np.zeros(shape)
    dirac[shape[0]//2, shape[1]//2] = 1.
    scales = star2d(dirac, scale=scl)
    std_scales = np.array([np.linalg.norm(conv(conv(psf, np.rot90(psf, 2)), scale),'fro') for scale in scales])
  
    sigma_scales = sigma * std_scales

    # Set k for k-sigma thresholding
    k = 3

    # (k+plus) for the finest scale
    plus = 0

    # build thresholds
    thresholds = np.array([(k+plus)*sigma_scales[:1], k*sigma_scales[1:]])

    # Starlet transform of image
    alpha = star2d(image, scale=scl)

    # Multiscale threshold except coarse scale
    alpha[:-1] = MS_soft_thresh(alpha[:-1], beta*thresholds[-1])

    # Apply the adjoint of the starlets on alpha
    return istar2d(alpha), thresholds, scales