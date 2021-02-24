from .observable import Observable
from .galaxy_shear import Shear
from astropy.io import fits
import healpy as hp

class PSF(Shear):
    def __init__(self, config, nside, mask_mode, psf_filepath, include_obs=False, include_w=False, *args, **kwargs):
        # nzbins = 3 for obs, piff, residual
        zbins = ['obs', 'p', 'q', 'w']
        self.include_obs = include_obs
        self.include_w = include_w
        if not include_obs:
            zbins.remove('obs')
        if not include_w:
            zbins.remove('w')

        super(PSF, self).__init__(config, nside, 'psf', zbins, mask_mode, data_dir=None, *args, **kwargs)

        self.obs_name = self.obs_name.replace('galaxy_shear', 'PSF')

        self.psf_filepath = psf_filepath

        self._make_cats()

    def _make_cats(self, save=True):
        data = fits.open(self.psf_filepath)[1].data

        for ibin in self.zbins:
            self.cats[ibin] = {}
            self.cats[ibin]['ra'] = data['ra']
            self.cats[ibin]['dec'] = data['dec']

        if self.include_obs:
            # PSF rescaled by size (w)
            self.cats['obs']['e1'] = data['obs_e1']
            self.cats['obs']['e2'] = data['obs_e2']
        # PSF model (p)
        self.cats['p']['e1'] = data['piff_e1']
        self.cats['p']['e2'] = data['piff_e2']
        # PSF residuals (q)
        self.cats['q']['e1'] = data['obs_e1'] - data['piff_e1']
        self.cats['q']['e2'] = data['obs_e2'] - data['piff_e2']
        if self.include_w:
            # PSF rescaled by size (w)
            self.cats['w']['e1'] = data['obs_e1'] * (1. - data['obs_T']/data['piff_T'])
            self.cats['w']['e2'] = data['obs_e2'] * (1. - data['obs_T']/data['piff_T'])
        for k in self.cats.keys():
            self.cats[k]['weight'] = np.ones_like(self.cats[k]['e1'])

