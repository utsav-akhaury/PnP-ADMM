import numpy as np
from skimage.measure import label
import tensorflow as tf

def fftconvolve(image, kernel):

    image = tf.expand_dims(tf.expand_dims(image, axis=0), axis=-1)
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)
    result = tf.cast(tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME'), tf.float32)
    return tf.squeeze(result)

def fft(data):

    return tf.math.real( tf.convert_to_tensor(1.0/data.get_shape().as_list()[0], tf.complex64) * 
                         tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(tf.cast(data, tf.complex64)))) )

def ifft(data):

    return tf.math.real( tf.convert_to_tensor(data.get_shape().as_list()[0], tf.complex64) *
                         tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(tf.cast(data, tf.complex64)))) )

def nmse(signal_1, signal_2):

    return tf.keras.backend.get_value(tf.norm(signal_2 - signal_1)**2 / tf.norm(signal_1)**2)

def soft_thresh(data, threshold):
    
    return tf.cast(tf.math.sign(data) * (data - threshold) * tf.cast(data >= threshold, tf.float32), tf.float32)

def sigma_mad(signal):
    
    """This function returns the estimate of the standard deviation of White
    Additive Gaussian Noise using the Mean Absolute Deviation method (MAD).
    INPUT: signal, Numpy Array
    OUTPUT: sigma, scalar"""
    
    sigma = 1.4826 * np.median(np.abs(signal - np.median(signal)))
    return sigma

def blob_mask(img,background=0,connectivity=2):
    
    """This function keeps the biggest blob in the image
    
    INPUT: img, Numpy Array
           background, integer
           connectivity, integer
    OUTPUT: mask, boolean Numpy Array"""
    
    labels = label(img,background=background,connectivity=connectivity)
    
    # find the biggest blob
    indices = np.unique(labels)
    sizes = np.zeros(indices.shape)
    for i in indices[1:]:
        sizes[i] = (labels==i).sum()
    main_blob_label = np.argmax(sizes)
    main_blob_estimate = (labels==main_blob_label)*main_blob_label
    
    # extract mask
    mask = (labels-main_blob_estimate)==0
    return mask


# Projection - Enforce non-negative values
def proj(xi):
    
    return tf.cast(tf.math.maximum(xi, 0.0), tf.float32)
 
# H operator
def H(data, psf):
    
    return fftconvolve(data, psf)

# H transpose operator
def Ht(data, psf):
    
    return fftconvolve(data, tf.reverse(tf.reverse(psf, axis=[0]), axis=[1]))      # rotate by 180

# The gradient
def grad(y, x_rec, psf):

    return Ht(H(x_rec, psf) - y, psf)

# The cost function
def cost_func(y, x_rec, psf, var):

    L_d = (tf.cast(0.5, tf.float32) / var) * tf.norm(y - H(x_rec, psf))**2      
    return tf.keras.backend.eval(L_d)

# Spectral value
def max_sv(psf):
    
    H = tf.signal.fft2d(tf.cast(psf, tf.complex64))
    normH = tf.math.abs(tf.reverse(tf.reverse(H, axis=[0]), axis=[1]) * H)
    return tf.cast(tf.math.reduce_max(normH), tf.float32)    
    
# Compute gradient step size   
def get_alpha(sv):

    return (tf.cast(1.0, tf.float32) / (sv * tf.cast(1.0 + 1.0e-5, tf.float32)))

def runFBS(y, x_0, psf, grad, sigma, n_iter, model, gal_target):    

    # Convert arrays to tensors 
    x_0 = tf.cast(x_0, tf.float32)
    x_k = x_0     
    sigma = tf.cast(sigma, tf.float32)

    # declare variables    
    cost = np.full(n_iter, np.inf)
    nmse_arr = np.full(n_iter, np.inf)
    
    # FISTA parameters  
    x_k = x_0
    t_k = tf.cast(1.0, tf.float32)
      
    # square of spectral radius of convolution matrix
    sv = max_sv(psf)           

    # The gradient descent step
    alpha = get_alpha(sv) 

    for k in range(n_iter):
        
        ## FISTA update
        x_k1 = x_k - alpha * grad(y, x_k, psf) 
#         x_k1 = proj(x_k1)      
                                            
        t_k1 = ( (tf.cast(1.0, tf.float32) + tf.math.sqrt(tf.cast(4.0, tf.float32)*t_k**2 + tf.cast(1.0, tf.float32))) 
                / tf.cast(2.0, tf.float32) )
        
        lambda_fista = tf.cast(1.0, tf.float32) + (t_k -  tf.cast(1.0, tf.float32)) / t_k1
        x_k1 = x_k + lambda_fista * (x_k1 - x_k)  
        
        # U-Net Denoising
        x_k1 = tf.expand_dims(tf.expand_dims(x_k1, axis=0), axis=-1)
        x_k1 = tf.cast(tf.squeeze(model(x_k1)), tf.float32)
        
        ## Cost
        cost[k] = cost_func(y, x_k1, psf, var=sigma**2)  
    
        # Update variables
        x_k = x_k1
                
        # Compute NMSE
        nmse_arr[k] = nmse(gal_target, x_k)  
        
        # Stopping Criteria
        if (np.abs(nmse_arr[k]-nmse_arr[k-1]) < 1e-8) or (np.abs(cost[k]-cost[k-1]) < 1e-8):
            return x_k
                 
    return x_k

def main_FBS(batch, model):
    
    n_iter = 150
    x_0 = np.zeros(batch['inputs'][0].shape)
    x_opt = np.zeros(batch['inputs'].shape)                               
    sigma = 23.59 / 4000.0  
    
    # Deconvolve given images
    for gal_ind in range(batch['inputs'].shape[0]):            

        gal_input = tf.cast(np.squeeze(batch['inputs'][gal_ind]), tf.float32)
        gal_target = np.squeeze(batch['targets'][gal_ind])
        psf = tf.cast(np.squeeze(batch['psf_cfht'][gal_ind]) , tf.float32)    

        # Deconvolve the image
        x_opt[gal_ind] = runFBS(gal_input, x_0, psf, grad, sigma, n_iter, model, gal_target)
        
    return x_opt