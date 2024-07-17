# SpecDB - Continuum Normalized Spectra for Keck/HIRES

This repo hosts some functions that were used to deblaze spectra from the middle 'r' chip of the High Resolution Echelle Spectrograph from Keck. For more information about HIRES data products please refer to [the California Kepler Survey website](https://california-planet-search.github.io/cks-website/) and [Petigura et al. 2017](https://ui.adsabs.harvard.edu/abs/2017AJ....154..107P/abstract). These functions were used in Polanski et al. 202X.

The steps in obtaining continuum normalized spectra are partially taken from [Valenti and Fischer 2005](https://iopscience.iop.org/article/10.1086/430500):

1. 3 iterations of 4$\sigma$ rejection, with outliers being interpolated. (per order)
2. A median filter to the entire 2D spectrum, with a footprint of 3x3 pixels, to bring the deepest spectral lines closer to the continuum in adjecent orders. 
3. Each smoothed order is then split into 10-20 bins and begin an iterative process to identify continuum pixels in each bin:
  * A 3rd order polynomial is fit the flux. The lowest 10% of flux values are masked and the polynomial is refit. This is repeated 5 times, ending with a rejection of 50% of lowest flux values. This attempts to progressively mask out deep spectral lines.
  * The final polynomial is divided out of the spectrum, and we reject any pixels that are 3$\sigma$ below the median of the spectrum.
  * The top 5-10% of flux values are then taken as the continuum points in that bin.
4. The continuum pixels in each order then fit with a high-order polynomial (N=9) which is then divided from the original, unsmoothed spectrum.
5. Finally, the deblazed spectrum is scaled such that 5% of the flux values exceed unity.


<img src="https://github.com/aspolanski/SpecDB/blob/main/continuum_finding.gif" width="600" height="600" />

An animation showing the iterative process descriped in Step 3.

This algorithm was optimized for stars ranging from F type to early K. It may also be easily adaptable to other echelle spectrographs (to-do?)

