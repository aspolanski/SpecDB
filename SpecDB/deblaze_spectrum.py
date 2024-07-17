#!/usr/bin/env python3

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
import smplotlib
from scipy.interpolate import interp1d, interp2d
import glob,sys,os
from scipy.ndimage import median_filter



def logger(log_string=None,log_path=None):

    #check for log and create new one

    if not os.path.isfile("./deblaze.log"):

        if not log_path:
            with open("./deblaze.log", "a") as log_file:
                log_file.write("BEGIN DEBLAZE LOG FILE\n")
        else:
            with open(f"{log_path}deblaze.log", "a") as log_file:
                log_file.write("BEGIN DEBLAZE LOG FILE\n")

    elif not log_string:

        pass

    else:

        with open("./deblaze.log", "a") as log_file:
            log_file.write(log_string)


def special_median_smooth(spec):

    #niche function for HIRES. Performs a median smooth with a 3x3 window but
    #replaces orders 9 & 10 with a median smooth of 2x2
    #since a 3x3 artificially lowers continuum down for these orders

    smoothed = median_filter(spec,[3,3])

    smoothed[9:11,:] = median_filter(spec,[2,2])[9:11,:]

    return smoothed

def make_new_fits(hdu,spec,errspec,w_soln,write=False,write_path=None):

    # helpler function to create new HIRES fits file

    hdu[0] = fits.PrimaryHDU(spec,header=hdu[0].header)
    err_hdu = fits.ImageHDU(errspec)
    w_hdu = fits.ImageHDU(w_soln)
    hdu.append(err_hdu)
    hdu.append(w_hdu)

    if not write:
        return hdu
    else:
        hdu.writeto(f"{write_path}",overwrite=True)



def unpack_fits(in_file):

    # helper function to read in HIRES data

    hdu = fits.open(in_file)
    star_name = hdu[0].header['TARGNAME']
    file_name = in_file.split('/')[-1].replace('.','_')[:-5]
    rstardum = hdu[0].data * 2.09

    return hdu, rstardum, star_name, file_name

def fix_artifacts(in_spec):

    #function to fix weird pixels at both ends of the spectra

    #just doing a blanket fix, its only 32 pixels out of 4021*16.

    #input is the full 16 orders

    spec= np.copy(in_spec)

    for i in range(spec.shape[0]):

        spec[i,:] = interp1d(np.arange(0,len(spec[i,1:-1])),
                       spec[i,1:-1],
                       fill_value='extrapolate')(np.arange(0,len(spec[i,:])))

    return spec

def interpolate_hot_pixel(spec,log=False):

    hot_pixel_mask = spec > (np.nanmedian(spec) + 4*np.nanstd(spec))
    hot_pixel_mask = hot_pixel_mask + np.roll(hot_pixel_mask,shift=1) + np.roll(hot_pixel_mask,shift=-1)

    if sum(hot_pixel_mask) == 0:

        return(spec)

    else:
        if not log:
            print(f'{sum(hot_pixel_mask)} outliers found. Interpolating...')
        else:
            logger(f'{sum(hot_pixel_mask)} outliers found. Interpolating.\n')

        for i in range(3):


            grid = np.arange(0,len(spec))[~hot_pixel_mask]

            new = interp1d(grid,
                           spec[~hot_pixel_mask],
                           fill_value='extrapolate')(np.arange(0,len(spec)))

            hot_pixel_mask = new > (np.nanmean(new) + 4*np.nanstd(new))
            #print(np.where(hot_pixel_mask==True))
            if sum(hot_pixel_mask) == 0:
                return new
                break

            i += 1


        return new


def remove_outliers(in_spec,sigma,warning_threshhold=10,log=False):

    spec= np.copy(in_spec)

    #get median and std of continuum normalized spectra

    med = np.nanmedian(spec,axis=0)
    std = np.nanstd(spec,axis=0)

    running_count = 0

    for i in range(spec.shape[0]):

        outlier_mask = spec[i,:] >= (med+sigma*std)

        outlier_idx = np.where(outlier_mask==True)[0]

        running_count+=sum(outlier_mask)

        if sum(outlier_mask) == 0:
            pass
        else:

            spec[i,:][outlier_idx] = interp1d(np.arange(0,len(spec[i,:]))[~outlier_mask],
                       spec[i,:][~outlier_mask],
                       fill_value='extrapolate')(np.arange(0,len(spec[i,:])))[outlier_idx]

    if running_count >= warning_threshhold:
        if not log:
            print(f'Order: {i}: Warning! More than 10 pixels have been flagged as outliers, does {running_count} seem like too many?')
        else:
            logger(f'Order: {i}: Warning! More than 10 pixels have been flagged as outliers, does {running_count} seem like too many?\n')

    return spec


def find_continuum(spec,npl,keep_fraction=0.95):

    # function to find the "continuum" pixels of a spectrum. 
    
    # starts with an initial polynomial fit, rejects 10% of the lowest flux value
    # pixels then fits with another polynomial. This is repeated for 5 iterations,
    # rejecting a higher fraction of the lowest flux pixels, essentially masking
    # out deep spectral lines.
    # The final polynomial is divided out to obtain a normalized spectrum. Pixels that
    # are 3 standard deviations below the median are masked then the highest "keep_fraction"
    # of pixels are taken as the continuum points in the spectrum

    # This function is optimized to run on small bins of a total spectrum.

    x = np.arange(len(spec))

    fractions = [0.1,0.2,0.3,0.4,0.5]
    intermediate_mask = np.ones(spec.shape,dtype=bool)
    for i in range(len(fractions)):

        c = np.poly1d(np.polyfit(x[intermediate_mask],spec[intermediate_mask],npl))(x)

        sfrac = np.quantile(np.sort(spec/c),fractions[i])

        intermediate_mask = intermediate_mask *(spec/c>=sfrac)

    spec=spec/c
    intermediate_mask = spec > np.nanmedian(spec) - 3.0*np.nanstd(spec)

    #c = np.poly1d(np.polyfit(x[intermediate_mask],spec[intermediate_mask],npl))(x)

    sfrac = np.quantile(np.sort(spec),keep_fraction)

    contiuum_points = intermediate_mask *(spec>=sfrac)

    return contiuum_points

def plot_blazes(conts,ax):

    #plot the blaze functions for each order and the median
    #input is a list of continuum functions

    cs = np.vstack(conts)

    cs = cs/np.median(cs,axis=1)[:,np.newaxis]

    master = np.median(cs,axis=0)

    colors = cm.RdPu_r(np.linspace(0,0.6,len(conts)))

    #fig, ax = plt.subplots(figsize=(5,6))


    for i in range(len(conts)):
        ax.plot(np.arange(0,len(cs[0,:])),cs[i,:],color=colors[i])

    ax.plot(np.arange(0,len(cs[0,:])),master,linewidth=4,color='k',label='Median')
    ax.legend(loc='upper left',frameon=True)
    ax.ticklabel_format(style='plain')
    plt.tight_layout()
    #return fig

def plot_spectrum(spec,smoothed_spec,conts,masks,ax,plot_mask=False,star_name=None,file_name=None):

    #fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(16,11),sharex=True)
    #fig.subplots_adjust(hspace=0)
    x = 0
    l = len(spec[0,:])

    ax[0].grid(visible=True,color='grey')
    ax[1].grid(visible=True,color='grey')


    for i in range(spec.shape[0]):

        ax[0].plot(np.arange(x,x+l),smoothed_spec[i,:],linewidth=0.5,color='k')
        ax[0].plot(np.arange(x,x+l),conts[i],linewidth=3,zorder=2000,color='purple')

        if plot_mask:
            ax[0].scatter(np.arange(x,x+l)[masks[i]],
                          smoothed_spec[i,:][masks[i]],
                          marker='o',edgecolor='b',color='none',s=30,zorder=100)


        ax[1].plot(np.arange(x,x+l),spec[i,:],linewidth=0.5,color='k')

        x+=len(rstardum[i,:])+1000

    title_str = f'Star: {star_name.strip()}  File: {file_name}'
    ax[0].set_title(title_str,loc='left')

    ax[0].ticklabel_format(style='plain')
    ax[1].set_xlim(ax[0].get_xlim())
    xs = ax[0].get_xlim()
    ax[1].fill_between(xs,y1=0.99,y2=1.01,color='g',alpha=0.4)
    ax[1].axhline(y=1.0)



    #return fig

def plot_single_order(spec,smoothed_spec,order,conts,masks,ax):

    #fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(13,11),sharex=True)
    #fig.subplots_adjust(hspace=0)
    x = 0
    l = len(spec[order,:])

    ax[0].grid(visible=True,color='grey')
    ax[1].grid(visible=True,color='grey')

    ax[0].plot(np.arange(x,x+l),smoothed_spec[order,:],linewidth=0.5,color='k')
    ax[0].plot(np.arange(x,x+l),conts[order],linewidth=3,zorder=2000,color='purple',label='Fitted Continuum')

    ax[0].scatter(np.arange(x,x+l)[masks[order]],
                          smoothed_spec[order,:][masks[order]],
                          marker='o',edgecolor='b',color='none',s=50,linewidths=2,zorder=100,label='Continuum Points')

    ax[1].plot(np.arange(x,x+l),spec[order,:],linewidth=0.5,color='k')
    ax[1].set_xlim(ax[0].get_xlim())
    xs = ax[0].get_xlim()
    ax[1].fill_between(xs,y1=0.99,y2=1.01,color='g',alpha=0.4)
    ax[1].axhline(y=1.0)

    ax[0].text(x=50,y=(np.max(smoothed[order,:])*0.95),s=f'Order: {1+order}',fontsize=20)
    ax[0].legend(loc='upper right',frameon=True,reverse=True)
    ax[0].ticklabel_format(style='plain')
    #ax[0].set_xlim(3000,4000)
    plt.tight_layout()
    #return fig



def deblaze(spec,nbins,npl):

    sub_spec = np.array_split(spec,nbins) #split the orders into bins

    cont = []
    for b in range(len(sub_spec)):
        #find the continuum in each order bin
        cont.append(find_continuum(sub_spec[b],npl))

    cont = np.concatenate(cont, axis=0 )

    x = np.arange(0,len(spec))
    #fit a high order polynomial through the apparent conituum
    poly = np.polyfit(x[cont],spec[cont],9)
    c1 = np.poly1d(poly)(x)

    #mask out any points below 0.9 relative flux that were marked at continuua
    new_cont = cont * (spec/c1>=0.95 )

    #refit
    c2 = np.poly1d(np.polyfit(x[new_cont],spec[new_cont],9))(x)

    return(spec, new_cont, c2)

def scale_spectra(cont_spec):

    scale_factor = np.quantile(np.sort(cont_spec[~np.isnan(cont_spec)]),0.95) #determine scale factor without nans

    return cont_spec/scale_factor


if __name__ == "__main__":

    if '-file' in sys.argv:
        p = sys.argv.index('-file')
        file = str(sys.argv[p+1])
    else:
        sys.exit("No spectrum provided!")

    if '-outdir' in sys.argv:
        p = sys.argv.index('-outdir')
        output_directory = sys.argv[p+1]
    else:
        output_directory = './deblazed/'

    if '-log' in sys.argv:
        if_log = True
    else:
        if_log = False

    if '-plot' in sys.argv:
        if_plot = True
    else:
        if_plot = False


hdu,rstardum,star_name,file_name = unpack_fits(file)
os.mkdir(f'{output_directory}{file_name}/')

if if_log:
    logger(log_path=f'{output_directory}{file_name}/')
    logger(f'\n########## {file_name} ##########\n')
else:
    pass


errspec = np.sqrt(1./rstardum + 0.005**2)
w_soln = fits.open("./keck_rwav.fits")[0].data
rstardum = fix_artifacts(rstardum)

for i in range(rstardum.shape[0]):
    if if_log:
        rstardum[i,:] = interpolate_hot_pixel(rstardum[i,:],log=True)
    else:
        rstardum[i,:] = interpolate_hot_pixel(rstardum[i,:],log=False)
smoothed = special_median_smooth(rstardum)

spec = rstardum[0,:]
nbins=[20,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
db_spec = np.zeros(rstardum.shape)
#for i in range(rstardum.shape[0]):
ms = []
cs = []
for i in range(rstardum.shape[0]):
        
    db, marray,c = deblaze(smoothed[i,:],nbins[i],3)
    db_spec[i,:] = rstardum[i,:]/c #you're doing this because you smooth the spectrum first
    negs = np.where(db_spec[i,:] <= 0.0)[0] #check for negative values and set to NaN
    db_spec[i,:][negs] = np.nan
    ms.append(marray)
    cs.append(c)

if if_log:    
    db_spec = remove_outliers(db_spec,3,log=True)
else:
    db_spec = remove_outliers(db_spec,3,log=False)


if if_plot:
    fig = plt.figure(figsize=(15,15),layout="tight")
    gs = gridspec.GridSpec(6,6, figure=fig)
    gs_full = gridspec.GridSpecFromSubplotSpec(2, 2,
                    subplot_spec=gs[0:3,0:6], wspace=0.1, hspace=0.0)
    sub_ax1 = fig.add_subplot(gs_full[0,:])
    sub_ax2 = fig.add_subplot(gs_full[1,:],sharex=sub_ax1)
    sub_ax1.tick_params(labelbottom=False)
    plot_spectrum(db_spec,smoothed,cs,ms,[sub_ax1,sub_ax2],star_name=star_name,file_name=file_name)
    gs_order = gridspec.GridSpecFromSubplotSpec(2, 2,
                        subplot_spec=gs[3:6,0:4], wspace=0.1, hspace=0.0)
    sub_ax3 = fig.add_subplot(gs_order[0,:])
    sub_ax4 = fig.add_subplot(gs_order[1,:],sharex=sub_ax3)
    sub_ax3.tick_params(labelbottom=False)
    plot_single_order(db_spec,smoothed,2,cs,ms,[sub_ax3,sub_ax4])
    ax3 = fig.add_subplot(gs[3:5, 4:])
    plot_blazes(cs,ax3)
    fig.savefig(f"./deblaze_plots/{file_name}.jpg",dpi=150)
else:
    pass


write_name = '/'  +file_name + '_deblaze' + ".fits"
make_new_fits(hdu,db_spec,errspec,w_soln,write=True,write_path=f'{output_directory+file_name}'+write_name)
    

