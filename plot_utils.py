import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


def comparison(x_opt, gal_target_tf, gal_input_tf, psf_tf, fftconvolve):

    temp = tf.concat([x_opt, gal_target_tf], 1)

    fig = plt.figure(figsize=(20,20))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2]) 

    ax1 = plt.subplot(gs[0])
    plt.title('Observation')
    ax1 = plt.gca()
    im1 = ax1.imshow(gal_input_tf)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)

    ax2 = plt.subplot(gs[1])
    plt.title('Optimal Reconstruction vs. Ground Truth')
    ax2 = plt.gca()
    im2 = ax2.imshow(tf.keras.backend.get_value(temp))
    ax2.axis('off')
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="2.5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)

    plt.tight_layout()


    fig2 = plt.subplots(1,3, figsize=(24,24))
    plt.subplot(131)
    plt.title('Residual (Y - H*X_est)')
    ax1 = plt.gca()
    residual = gal_input_tf - fftconvolve(psf_tf, tf.cast(x_opt, tf.float32))
    im1 = ax1.imshow(tf.keras.backend.get_value(residual))
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)

    plt.subplot(132)
    plt.title('Log [FFT (Residual (Y - H*X_est))]')
    ax2 = plt.gca()
    im2 = ax2.imshow(tf.keras.backend.get_value(tf.math.log(residual)))
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)

    plt.subplot(133)
    plt.title('Error (X_true - X_est)')
    ax3 = plt.gca()
    im3 = ax3.imshow(tf.keras.backend.get_value(gal_target_tf - x_opt))
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax3)

    plt.show(fig2)
    
    
    
def nmse(nmse_arr, n_iter):
    min_nmse = np.min(nmse_arr)
    min_iter_nmse = np.where(nmse_arr == min_nmse)[0] + 1

    plt.figure(figsize=(16,8))
    plt.plot(np.arange(1,n_iter+1), nmse_arr)
    plt.title('NMSE vs. # Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('NMSE')
    plt.show()
    print('\nMinimum NMSE = {} (at {} iterations)'.format(min_nmse, min_iter_nmse))
    
    
def cost(cost, n_iter):
    min_cost = np.min(cost)
    min_iter_cost = np.where(cost == min_cost)[0] + 1

    plt.figure(figsize=(16,8))
    plt.plot(np.arange(1,n_iter+1), cost)
    plt.title('Cost vs. # Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
    print('\nMinimum Cost = {} (at {} iterations)'.format(min_cost, min_iter_cost))