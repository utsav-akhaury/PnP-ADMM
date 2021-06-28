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