from .observable import *
from .galaxy_shear import Shear

class PSF(Shear):
    def __init__(self, config, nside, mask_mode, psf_filepath, *args, **kwargs):
        # nzbins = 3 for obs, piff, residual

        super(PSF, self).__init__(config, nside, 'psf', ['p', 'q', 'w'], mask_mode, data_dir=None, *args, **kwargs)

        # def __init__(self, config, nside, mode, nzbins, mask_mode, data_dir='../data', *args, **kwargs):

        # self.spin = 2

        # self.mask_mode = mask_mode

        self.obs_name = 'PSF'
        # self.map_names = ['count', 'e1', 'e2']

        # self.obs_name = 'PSF'
        # self.map_names = ['e1', 'e2']

        self.psf_filepath = psf_filepath

        self._make_cats()

    def _make_cats(self, save=True):
        data = fits.open(self.psf_filepath)[1].data

        for ibin in self.zbins:
            self.cats[ibin] = {}
            self.cats[ibin]['ra'] = data['ra']
            self.cats[ibin]['dec'] = data['dec']

        # PSF model (p)
        self.cats['p']['e1'] = data['piff_e1']
        self.cats['p']['e2'] = data['piff_e2']
        # PSF residuals (q)
        self.cats['q']['e1'] = data['obs_e1'] - data['piff_e1']
        self.cats['q']['e2'] = data['obs_e2'] - data['piff_e2']
        # PSF rescaled by size (w)
        self.cats['w']['e1'] = data['obs_e1'] * (1. - data['obs_T']/data['piff_T'])
        self.cats['w']['e2'] = data['obs_e2'] * (1. - data['obs_T']/data['piff_T'])

    # def make_maps(self, save=True):
    #     data = fits.open(self.psf_filepath)[1].data

    #     self.cats = {}
    #     for ibin in self.zbins:

    #     cat['ra'] = data['ra']
    #     cat['dec'] = data['dec']

    #     # PSF model (p)
    #     cat['p_e1'] = data['piff_e1']
    #     cat['p_e2'] = data['piff_e2']
    #     # PSF residuals (q)
    #     cat['q_e1'] = data['obs_e1'] - data['piff_e1']
    #     cat['q_e2'] = data['obs_e2'] - data['piff_e2']
    #     # PSF rescaled by size (w)
    #     cat['w_e1'] = data['obs_e1'] * (1. - data['obs_T']/data['piff_T'])
    #     cat['w_e2'] = data['obs_e2'] * (1. - data['obs_T']/data['piff_T'])

    #     keys = ['p_e1', 'p_e2', 'q_e1', 'q_e2', 'w_e1', 'w_e2']
    #     quantities, count, mask = ca.cosmo.make_healpix_map(cat['ra'], cat['dec'],
    #                                                 quantity=[cat[_x] for _x in keys],
    #                                                 nside=self.nside, fill_UNSEEN=True, mask=None, weight=None)
        
    #     self.maps['p']['e1']  = quantities[0]
    #     self.maps['p']['e2']  = quantities[1]
    #     self.maps['q']['e1']  = quantities[2]
    #     self.maps['q']['e2']  = quantities[3]
    #     self.maps['w']['e1']  = quantities[4]
    #     self.maps['w']['e2']  = quantities[5]

    #     self.cats = {}
    #     for ibin in self.zbins:
    #         self.maps[ibin]['count'] = count
    #         self.cats[ibin]['ra'] = data['ra']
    #         self.cats[ibin]['dec'] = data['dec']
    #         self.cats[ibin]['e1'] = 
    #         if self.mask_mode == 'binary':
    #             self.masks[ibin] = mask.astype(float)
    #         elif self.mask_mode == 'count':
    #             self.masks[ibin] = count.astype(float)

    #     if save:
    #         self.save_maps()            

    # def make_fields(self, hm, include_templates=True):
    #     for ibin in tqdm(self.zbins, desc='{}.make_fields'.format(self.obs_name)):
    #         self.fields[ibin] = nmt.NmtField(self.masks_apo[ibin],
    #                                     [self.maps[ibin]['e1'], self.maps[ibin]['e2']],
    #                                     templates=None,
    #                                     **hm.fields_kw)

    # def compute_ipix(self): 
    #     if not hasattr(self, 'ipix'):
    #         self.ipix = {}
    #         for ibin in tqdm(self.zbins, desc='{}.compute_ipix'.format(self.obs_name)):
    #             cat = self.cats[ibin]
    #             self.ipix[ibin] = hp.ang2pix(self.nside, (90-cat['dec'])*np.pi/180.0, cat['ra']*np.pi/180.0)

    # def _compute_random_auto_cls(self, hm, ibin, nrandom, save_wsp=True):
    #     npix = self.npix
    #     cat = self.cats[ibin]
    #     mask_apo = self.masks_apo[ibin]

    #     field_0 = self.get_field(hm, ibin)

    #     # wsp = nmt.NmtWorkspace()
    #     # suffix = '{}_{}_{}_{}'.format(self.obs_name, self.obs_name, ibin, ibin)
    #     # wsp_filename = hm.load_workspace_if_exists(wsp, suffix, return_filename=True)
    #     # if not wsp_filename: #load_workspace_if_exists
    #     #     wsp.compute_coupling_matrix(field_0, field_0, hm.b)
    #     #     if save_workspace:
    #     #         wsp_filename = hm.save_workspace(wsp, suffix, return_filename=True)
    #     wsp = hm.get_workspace(self, self, ibin, ibin, save_wsp=save_wsp)
    #     wsp_filename = hm.get_workspace_filename(self, self, ibin, ibin)

    #     # cls = {}
    #     # cls['true'] = compute_master(field_0, field_0, wsp)

    #     # if nrandom > 0:
    #     Nobj = len(cat)
    #     self.compute_ipix()

    #     # count = np.zeros(npix, dtype=float)
    #     ipix = self.ipix[ibin] #hp.ang2pix(self.nside, (90-cat['dec'])*np.pi/180.0, cat['ra']*np.pi/180.0)
    #     # np.add.at(count, ipix, 1.)
    #     count = self.maps[ibin]['count']
    #     bool_mask = (count > 0.)

    #     _cls = []

    #     if hm.nproc==0:
    #         for i in trange(nrandom, desc='{}._compute_random_auto_cls [bin {}]'.format(self.obs_name, ibin)):
    #             _cls.append(_randrot_cls(cat['e1'], cat['e2'], ipix, npix, bool_mask, mask_apo, count, hm.purify_e, hm.purify_b, wsp))

    #     else:
    #         args = (cat['e1'], cat['e2'], ipix, npix, bool_mask, mask_apo, count, hm.purify_e, hm.purify_b, wsp_filename) # self.nside, hm.lmax, hm.nlb)
    #         _multiple_results = [hm.pool.apply_async(_multiproc_randrot_cls, (len(_x), args, pos+1)) for pos, _x in enumerate(np.array_split(range(nrandom), hm.nproc)) if len(_x)>0]
    #         for res in tqdm(_multiple_results, desc='{}._compute_random_auto_cls [bin {}]<{}>'.format(self.obs_name, ibin, os.getpid()), position=0):
    #             _cls += res.get()
    #         print("\n")

    #     return np.array(_cls)