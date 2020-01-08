from .observable import *
from .utils import *
# import ray
# ray.init()

class Shear(Observable):
    def __init__(self, config, nside, mode, nzbins, mask_mode, data_dir='../data', dry_run=False, *args, **kwargs):
        self.obs_name = 'galaxy_shear'
        self.map_names = ['count', 'e1', 'e2']

        super(Shear, self).__init__(config, nside, mode, nzbins, self.obs_name, self.map_names, *args, **kwargs)

        self.spin = 2
        self.kernel = 'nz_source'
        self.type = twopoint.Types.galaxy_shear_emode_fourier

        self.data_dir = data_dir

        self.cats = {}

        mask_mode_split = mask_mode.split('>')
        assert mask_mode_split[0] in ['binary', 'count']
        self.mask_mode = mask_mode_split[0]
        if len(mask_mode_split) == 2:
            self.count_cut = int(mask_mode_split[1])
        else:
            self.count_cut = 0

        if not dry_run:
            if mode.startswith('buzzard'):
                self._init_buzzard()
            elif mode=='data_sub' or mode=='mastercat':
                self._init_data()
            elif mode=='mock':
                self._init_mock()
            elif mode=='full':
                self._init_full(kwargs['filename_template'], kwargs['dict'], kwargs['flip_e2'], kwargs.get('single_file', False), kwargs.get('ext_template', 'zbin_{}'), kwargs.get('ipix_instead_of_radec', False))
            elif mode=='tables':
                self._init_tables(kwargs['tables'], kwargs['dict'], kwargs['flip_e2'], kwargs.get('ext_template', 'zbin_{}', kwargs.get('ipix_instead_of_radec', False)))
            elif mode=='flask':
                self._init_flask(kwargs['isim'], kwargs['cookie'])
            elif mode=='psf' or 'dry':
                pass
            else:
                raise ValueError("Given `mode` argument ({}) is not correct.".format(mode))

    def _init_buzzard(self):
        # self.zlims = [(.2, .43), (.43,.63), (.64,.9), (.9, 1.3), (.2,1.3)][:self.nzbins]
        for ibin in self.prog(self.zbins):
            filename = os.path.join(self.data_dir, "Niall_WL_y3_bin_{}.fits".format(ibin+1))
            _cat = fits.open(filename)[1].data

            cat = {}
            cat['ra'] = _cat['RA']
            cat['dec'] = _cat['DEC']

            if self.mode == 'buzzard':
                cat['e1'] = _cat['E1']
                cat['e2'] = _cat['E2']
            if self.mode == 'buzzard_truth':
                cat['e1'] = _cat['G1']
                cat['e2'] = _cat['G2']

            self.cats[ibin] = cat

    def _init_data(self):
        for ibin in self.prog(self.zbins):
            filename = os.path.join(self.data_dir, "source_s{}.fits".format(ibin+1))
            _cat = fits.open(filename)[1].data

            cat = {}
            cat['ra'] = _cat['RA']
            cat['dec'] = _cat['DEC']

            cat['e1'] = _cat['g1']
            cat['e2'] = -1.0 * _cat['g2']

            self.cats[ibin] = cat

    def _init_mock(self):
        for ibin in self.prog(self.zbins):
            filename = os.path.join(self.data_dir, "source_s{}.fits".format(ibin+1))
            _cat = fits.open(filename)[1].data

            cat = {}
            cat['ra'] = _cat['RA']
            cat['dec'] = _cat['DEC']

            cat['e1'] = _cat['e1']
            cat['e2'] = _cat['e2']

            self.cats[ibin] = cat

    def _init_full(self, filename_template, dict, flip_e2, single_file=False, ext_template='zbin_{}', ipix_instead_of_radec=False):
        if single_file:
            full_cat = fits.open(os.path.join(self.data_dir, filename_template))
        if ipix_instead_of_radec:
            self.ipix = {}
        for ibin in self.prog(self.zbins):
            if single_file:
                _cat = full_cat[ext_template.format(ibin)].data
            else:
                filename = os.path.join(self.data_dir, filename_template.format(ibin+1))
                _cat = fits.open(filename)[1].data

            cat = {}
            if ipix_instead_of_radec:
                self.ipix[ibin] = _cat[dict['ipix']]
            else:
                cat['ra'] = _cat[dict['ra']]
                cat['dec'] = _cat[dict['dec']]

            cat['e1'] = _cat[dict['e1']]
            cat['e2'] = _cat[dict['e2']]

            if flip_e2:
                cat['e2'] *= -1.0

            self.cats[ibin] = cat    

    def _init_tables(self, tables, dict, flip_e2=False, ext_template='zbin_{}', ipix_instead_of_radec=False):
        is_list = isinstance(tables, list)
        if ipix_instead_of_radec:
            self.ipix = {}
        for i, ibin in self.prog(enumerate(self.zbins)):
            if is_list:
                _cat = tables[i]
            else:
                _cat = tables[ext_template.format(ibin)]

            cat = {}
            if ipix_instead_of_radec:
                self.ipix[ibin] = _cat[dict['ipix']]
            else:
                cat['ra'] = _cat[dict['ra']]
                cat['dec'] = _cat[dict['dec']]

            cat['e1'] = _cat[dict['e1']]
            cat['e2'] = _cat[dict['e2']]

            if flip_e2:
                cat['e2'] *= -1.0

            self.cats[ibin] = cat

    def _init_flask(self, isim, cookie):
        for ibin in self.prog(self.zbins):
            filename = os.path.join(self.data_dir, 'src-cat_s{}_z{}_ck{}.fits'.format(isim, ibin+1, cookie))
            _cat = fits.open(filename)[1].data

            cat = {}
            cat['ra'] = _cat['RA']
            cat['dec'] = _cat['DEC']

            cat['e1'] = _cat['GAMMA1']
            cat['e2'] = -1.0 * _cat['GAMMA2']

            self.cats[ibin] = cat   

    def make_maps(self, save=True):
        keys = ['e1', 'e2']
        self.get_ipix()
        for ibin in self.prog(self.zbins, desc='{}.make_maps'.format(self.obs_name)):
            cat = self.cats[ibin]
            quantities, count, mask = ca.cosmo.make_healpix_map(None, None,
                                                    quantity=[cat[_x] for _x in keys],
                                                    nside=self.nside, fill_UNSEEN=True,
                                                    mask=None, weight=None,
                                                    ipix=self.ipix[ibin])
            for j, key in enumerate(keys):
                self.maps[ibin][key] = quantities[j]

            # Count cut
            count_cut_mask = (count>self.count_cut)

            self.maps[ibin]['count'] = count * count_cut_mask.astype(int)
            self.masks[ibin] = mask.astype(float) * count_cut_mask.astype(float)

        if save:
            self.save_maps()        

    def make_masks_apo(self):
        super(Shear, self).make_masks_apo()

        # Apply inverse-variance weighting after apodization
        if self.mask_mode =='count':
            for ibin in self.zbins:
                self.masks_apo[ibin] *= self.maps[ibin]['count']

    def make_fields(self, hm, include_templates=True):
        if hm.purify_b or hm.purify_e:
            print("WARNING: E/B-mode purification requires unseen pixels to be set to zero. Replacing...")
            for ibin in self.zbins:
                self.maps[ibin]['e1'] = hpunseen2zero(self.maps[ibin]['e1'])
                self.maps[ibin]['e2'] = hpunseen2zero(self.maps[ibin]['e2'])

        for ibin in self.prog(self.zbins, desc='{}.make_fields'.format(self.obs_name)):
            self.fields[ibin] = nmt.NmtField(self.masks_apo[ibin],
                                        [self.maps[ibin]['e1'], self.maps[ibin]['e2']],
                                        templates=self._get_templates_array() if include_templates else None,
                                        **hm.fields_kw)

    def make_randomized_fields(self, hm, ibin, nrandom=1):
        bool_mask = (self.maps[ibin]['count'] > 0.)
        self.get_ipix()
        fields = []
        for i in self.prog(nrandom):
            e1_map, e2_map = _randrot_maps(self.cats[ibin]['e1'], self.cats[ibin]['e2'], self.ipix[ibin], self.npix, bool_mask, self.maps[ibin]['count'])
            field =  nmt.NmtField(self.masks_apo[ibin], [e1_map, e2_map],
                                    templates=None,
                                    **hm.fields_kw)
            fields.append(field)

        return fields

    def _get_info(self):
        import pandas as pd

        info = {}
        info['e1std'] = []
        info['e2std'] = []
        info['e12std'] = []
        info['fsky'] = []
        info['Ngal'] = []
        info['area (std)'] = []
        info['area (sq. deg.)'] = []
        info['nbar (gal/std)'] = []
        info['nbar (gal/arcmin)'] = []

        for i in self.zbins:
            e1std = np.std(self.cats[i]['e1'])
            e2std = np.std(self.cats[i]['e2'])
            e12std = np.sqrt(e1std**2+e2std**2)
            fsky = np.sum(self.maps[i]['count']>0.) * 1./ len(self.masks[i])
            Ngal = int(np.sum(self.maps[i]['count'][self.masks[i].astype(bool)]))
            area = 4.*np.pi * fsky
            area_sqdeg = area * (180./np.pi)**2
            nbar_std = Ngal / area
            nbar_arcmin = nbar_std / (180*60/np.pi)**2
            
            info['e1std'].append(e1std)
            info['e2std'].append(e2std)
            info['e12std'].append(e12std)
            info['fsky'].append(fsky)
            info['Ngal'].append(Ngal)
            info['area (std)'].append(area)
            info['area (sq. deg.)'].append(area_sqdeg)
            info['nbar (gal/std)'].append(nbar_std)
            info['nbar (gal/arcmin)'].append(nbar_arcmin)

        df = pd.DataFrame(index=self.zbins, data=info)
        return df

    def compute_ipix(self): 
        self.ipix = {}
        for ibin in self.prog(self.zbins, desc='{}.compute_ipix'.format(self.obs_name)):
            cat = self.cats[ibin]
            self.ipix[ibin] = hp.ang2pix(self.nside, (90-cat['dec'])*np.pi/180.0, cat['ra']*np.pi/180.0)
    
    def set_ipix(self, ipixs):
        self.ipix = {}
        for ibin, ipix in zip(self.zbins, ipixs):
            self.ipix[ibin] = ipix
    
    def get_ipix(self):
        if not hasattr(self, 'ipix'):
            self.compute_ipix()
        return self.ipix

    # @profile
    def _compute_random_auto_cls(self, hm, ibin, nrandom):
        npix = self.npix
        cat = self.cats[ibin]
        mask_apo = self.masks_apo[ibin]

        wsp = hm.get_workspace(self, self, ibin, ibin)#, save_wsp=save_wsp)
        wsp_filename = hm.get_workspace_filename(self, self, ibin, ibin)

        Nobj = len(cat)
        self.get_ipix()

        ipix = self.ipix[ibin] #hp.ang2pix(self.nside, (90-cat['dec'])*np.pi/180.0, cat['ra']*np.pi/180.0)
        count = self.maps[ibin]['count']
        bool_mask = (count > 0.)

        _cls = []

        if hm.nproc==0:
            for i in self.prog(nrandom, desc='{}._compute_random_auto_cls [bin {}]'.format(self.obs_name, ibin)):
                _cls.append(_randrot_cls(cat['e1'], cat['e2'], ipix, npix, bool_mask, mask_apo, count, hm.purify_e, hm.purify_b, wsp))

        else:
            args = (cat['e1'], cat['e2'], ipix, npix, bool_mask, mask_apo, count, hm.purify_e, hm.purify_b, wsp_filename) # self.nside, hm.lmax, hm.nlb)
            _multiple_results = [hm.pool.apply_async(_multiproc_randrot_cls, (len(_x), args, pos+1)) for pos, _x in enumerate(np.array_split(range(nrandom), hm.nproc)) if len(_x)>0]
            for res in self.prog(_multiple_results, desc='{}._compute_random_auto_cls [bin {}]<{}>'.format(self.obs_name, ibin, os.getpid()), position=0):
                _cls += res.get()
            print("\n")

        return np.array(_cls)

    def plot_auto_cls(self, hm, remove_Nl=False, **kwargs):
        cls = {}
        titles = {}
        ylabels = {}

        _titles = ['EE', 'EB', 'BB']

        idx_EB = [0, 1, 3]

        for i, zbin in enumerate(self.zbins):
            for j in range(3):
                k = (i,j)
                titles[k] = _titles[j] + ' [bin %i]'%(i+1)
                ylabels[k] = '$C_\\ell$'
                ylabels[k] = '$C_\\ell$'
                if 'factor_ell' in kwargs.keys():
                    if kwargs['factor_ell'] == 1:
                        ylabels[k] = '$\\ell C_\\ell$'
                    if kwargs['factor_ell'] == 2:
                        ylabels[k] = '$\\ell (\\ell+1) C_\\ell$'
                else:
                    ylabels[k] = '$C_\\ell$'
                cls[k] = {}
                cls[k]['data'] = np.copy(hm.cls[(self.obs_name, self.obs_name)][(zbin, zbin)]['data'][idx_EB[j]])
                cls[k]['random'] = np.copy(hm.cls[(self.obs_name, self.obs_name)][(zbin, zbin)]['random'][:,idx_EB[j],:])
                if remove_Nl:
                    clr_r_m = np.mean(cls[k]['random'], axis=0)
                    cls[k]['data'] -= clr_r_m
                    cls[k]['random'] -= clr_r_m

        return self.plot_cls(hm, cls, self.nzbins, 3, figname='auto', titles=titles, ylabels=ylabels, **kwargs)

    def plot_cls_BB_only(self, hm, remove_Nl=False, **kwargs):
        cls = {}
        titles = {}
        ylabels = {}

        for i in range(1):
            for j, zbin in enumerate(self.zbins):
                k = (i,j)
                titles[k] = 'BB spectrum [bin %i]'%(zbin+1)
                if 'factor_ell' in kwargs.keys():
                    if kwargs['factor_ell'] == 1:
                        ylabels[k] = '$\\ell C_\\ell ^{\\rm BB}$'
                    if kwargs['factor_ell'] == 2:
                        ylabels[k] = '$\\ell (\\ell+1) C_\\ell ^{\\rm BB}$'
                else:
                    ylabels[k] = '$C_\\ell ^{\\rm BB}$'
                cls[k] = {}
                cls[k]['data'] = np.copy(hm.cls[(self.obs_name, self.obs_name)][(zbin, zbin)]['data'][3])
                cls[k]['random'] = np.copy(hm.cls[(self.obs_name, self.obs_name)][(zbin, zbin)]['random'][:,3,:])
                if remove_Nl:
                    clr_r_m = np.mean(cls[k]['random'], axis=0)
                    cls[k]['data'] -= clr_r_m
                    cls[k]['random'] -= clr_r_m

        return self.plot_cls(hm, cls, 1, self.nzbins, figname='BB', titles=titles, ylabels=ylabels, **kwargs)


    # def _compute_cross_template_cls(self, hm, ibin, nrandom=0, save=True):
    #     npix = self.npix
    #     cat = self.cats[ibin]
    #     mask_apo = self.masks_apo[ibin]

    #     logging.info("_compute_cross_template_cls: Making field_0")
    #     field_0 = self.get_field(hm, ibin, include_templates=False)

    #     logging.info("_compute_cross_template_cls: Making template_fields and wsp_dir")
    #     template_fields = {}
    #     wsp_dir  = {}
    #     for tempname, temp in self.prog(self.templates_dir.items(), desc='{}.compute_cross_template_cls [bin {}]'.format(self.obs_name, ibin)):
    #         mask = np.logical_not((temp == hp.UNSEEN) | (temp == 0.0)) # kinda dangerous...
    #         template_fields[tempname] = nmt.NmtField(mask, [temp])
    #         wsp_dir[tempname] = nmt.NmtWorkspace()
    #         wsp_dir[tempname].compute_coupling_matrix(field_0, template_fields[tempname], hm.b)
    #         hm.cls[(self.obs_name, tempname)][ibin] = {}
    #         hm.cls[(self.obs_name, tempname)][ibin]['data'] = compute_master(field_0, template_fields[tempname], wsp_dir[tempname])
    #         if nrandom > 0:
    #             hm.cls[(self.obs_name, tempname)][ibin]['random'] = []

    #     if nrandom > 0:
    #         Nobj = len(cat)
    #         self.get_ipix()

    #         count = np.zeros(npix, dtype=float)
    #         ipix = self.ipix[ibin] #hp.ang2pix(self.nside, (90-cat['dec'])*np.pi/180.0, cat['ra']*np.pi/180.0)
    #         np.add.at(count, ipix, 1.)
    #         bool_mask = (count > 0.)

    #         if hm.nproc==0:
    #             cls_r = []
    #             for i in self.prog(nrandom, desc='{}.compute_cross_template_cls [bin {}]'.format(self.obs_name, ibin)):
    #                 cls_r.append(_randrot_cross_cls(cat['e1'], cat['e2'], ipix, npix, bool_mask, mask_apo, count, hm.purify_e, hm.purify_b, template_fields, wsp_dir))
    #             for tempname in self.templates_dir.keys():
    #                 hm.cls[(self.obs_name, tempname)][ibin]['random'] = np.array([_x[tempname] for _x in cls_r])

    #         else:
    #             args = (cat['e1'], cat['e2'], ipix, npix, bool_mask, mask_apo, count, hm.purify_e, hm.purify_b, self.nside, hm.lmax, hm.nlb)
    #             _multiple_results = [hm.pool.apply_async(_multiproc_randrot_cross_cls, (len(_x), args, self.templates_dir, pos+1)) for pos, _x in enumerate(np.array_split(range(nrandom), hm.nproc)) if len(_x)>0]
    #             cls_r = []
    #             for res in self.prog(_multiple_results, desc='{}.compute_cross_template_cls [bin {}]<{}>'.format(self.obs_name, ibin, os.getpid()), position=0):
    #                 cls_r += res.get()

    #             for tempname in self.templates_dir.keys():
    #                 hm.cls[(self.obs_name, tempname)][ibin]['random'] = np.array([_x[tempname] for _x in cls_r])

    #             print("\n")

    # def plot_cross_template_cls(self, hm, showchi2=False, EB=0):
    #     ntemp = len(list(self.templates_dir.keys()))

    #     fig, axes = plt.subplots(ntemp, self.nzbins, figsize=(4*self.nzbins, 3*ntemp))
    #     ell = hm.b.get_effective_ells()

    #     # EB = 0 #0 for E, 1 for B
    #     ylabels = ['$\\ell (\\ell+1) C_\\ell^{{\\rm syst} \\times \\gamma_{\\rm %s}}$'%s for s in ['E', 'B']]
    #     gamma_label = ['$\\gamma_{\\rm %s}$'%s for s in ['E', 'B']]

    #     factor = ell*(ell+1)

    #     import scipy

    #     chi2 = {}
    #     for i, ibin in enumerate(self.zbins):
    #         for ik, key in enumerate(self.templates_dir.keys()):
    #             ax = axes[ik, i]
    #             ax.axhline(y=0, c='0.8', lw=1)
    #             nrandom = hm.cls[(self.obs_name, key)][ibin]['random'].shape[0]
    #             for j in range(nrandom):
    #                 ax.plot(ell, factor*hm.cls[(self.obs_name, key)][ibin]['random'][j,EB,:], c='r', alpha=max(0.01, 1./nrandom))
    #             if showchi2:
    #                 _chi2 = get_chi2_smoothcov(hm.cls[(self.obs_name, key)][ibin]['data'][EB,:], hm.cls[(self.obs_name, key)][ibin]['random'][:,EB,:])
    #                 label = '$\\chi^2_{{{:}}} = {:.2f}$ ($p={:.1e}$)'.format(len(ell), _chi2, scipy.stats.chi2.sf(_chi2, df=hm.b.get_n_bands()))
    #                 chi2[(self.obs_name, key,ibin)] = _chi2
    #             else:
    #                 label = None
    #             ax.plot(ell, factor*hm.cls[(self.obs_name, key)][ibin]['data'][EB,:], label=label, c='b')
    #             ax.set_title(gamma_label[EB] + ' [bin %i] $\\times$ '%(ibin+1) + key, fontsize=8)
    #             ax.set_xlabel('$\\ell$')
    #             ax.set_ylabel(ylabels[EB])
    #             ax.set_xlim(0, hm.b.lmax)
    #             vmax = max(np.abs(ax.get_ylim()))
    #             ax.set_ylim(-vmax,+vmax)
    #             if showchi2:
    #                 ax.legend(loc=1)
    #     plt.tight_layout()

    #     make_directory(self.config.path_figures+'/'+self.name)
    #     figfile = os.path.join(self.config.path_figures, self.name, 'cls_cross_templates_{}_{}_{}_nside{}.png'.format(self.obs_name, self.config.name, self.mode, self.nside))
    #     plt.savefig(figfile, dpi=300)

    #     if showchi2:
    #         return chi2

    # def _compute_cross_PSF_cls(self, hm, ibin, nrandom=0, save=True):
    #     npix = self.npix
    #     cat = self.cats[ibin]
    #     mask_apo = self.masks_apo[ibin]

    #     logging.info("_compute_cross_PSF_cls: Making field_0")
    #     field_0 = self.get_field(hm, ibin, include_templates=False)

    #     logging.info("_compute_cross_PSF_cls: Making template_fields and wsp_dir")
    #     wsp_dir  = {}
    #     for key, psf_field in self.prog(self.psf_fields.items(), desc='{}._compute_cross_PSF_cls [bin {}]'.format(self.obs_name, ibin)):
    #         wsp_dir[key] = nmt.NmtWorkspace()
    #         wsp_dir[key].compute_coupling_matrix(field_0, psf_field, hm.b)
    #         hm.cls[(self.obs_name, key)][ibin] = {}
    #         hm.cls[(self.obs_name, key)][ibin]['data'] = compute_master(field_0, psf_field, wsp_dir[key])
    #         if nrandom > 0:
    #             hm.cls[(self.obs_name, key)][ibin]['random'] = []

    #     if nrandom > 0:
    #         Nobj = len(cat)
    #         self.get_ipix()

    #         count = np.zeros(npix, dtype=float)
    #         ipix = self.ipix[ibin] #hp.ang2pix(self.nside, (90-cat['dec'])*np.pi/180.0, cat['ra']*np.pi/180.0)
    #         np.add.at(count, ipix, 1.)
    #         bool_mask = (count > 0.)

    #         if hm.nproc==0:
    #             cls_r = []
    #             for i in self.prog(nrandom, desc='{}._compute_cross_PSF_cls [bin {}]'.format(self.obs_name, ibin)):
    #                 cls_r.append(_randrot_cross_PSF_cls(cat['e1'], cat['e2'], ipix, npix, bool_mask, mask_apo, count, hm.purify_e, hm.purify_b, self.psf_fields, wsp_dir))
    #             for key in self.psf_maps.keys():
    #                 hm.cls[(self.obs_name, key)][ibin]['random'] = np.array([_x[key] for _x in cls_r])

    #         else:
    #             args = (cat['e1'], cat['e2'], ipix, npix, bool_mask, mask_apo, count, hm.purify_e, hm.purify_b, self.nside, hm.lmax, hm.nlb)
    #             _multiple_results = [hm.pool.apply_async(_multiproc_randrot_cross_PSF_cls, (len(_x), args, self.psf_maps, self.psf_mask_apo, pos+1)) for pos, _x in enumerate(np.array_split(range(nrandom), hm.nproc)) if len(_x)>0]
    #             cls_r = []
    #             for res in self.prog(_multiple_results, desc='{}._compute_cross_PSF_cls [bin {}]<{}>'.format(self.obs_name, ibin, os.getpid()), position=0):
    #                 cls_r += res.get()

    #             for key in self.psf_maps.keys():
    #                 hm.cls[(self.obs_name, key)][ibin]['random'] = np.array([_x[key] for _x in cls_r])

    #             print("\n")

    # def plot_cross_PSF_cls(self, hm, showchi2=False, EB_shear=0, EB_psf=0, sqrtscale=False):
    #     ntemp = len(list(self.psf_maps.keys()))

    #     fig, axes = plt.subplots(ntemp, self.nzbins, figsize=(4*self.nzbins, 3*ntemp))
    #     ell = hm.b.get_effective_ells()

    #     EB = ['E', 'B']
    #     ylabel = '$\\ell C_\\ell$' # (\\ell+1)
    #     # gamma_label = ['$\\gamma_{\\rm %s}$'%s for s in ['E', 'B']]
    #     which = np.array([[0,1],[2,3]])
    #     title = '$\\gamma_{{\\rm {}}}^{{\\rm gal}} \\ {{\\rm [bin {}]}} \\times \\gamma_{{\\rm {}}}^{{\\rm \\ PSF {}}}$'

    #     which = np.array([[0,1],[2,3]])[EB_shear,EB_psf]

    #     factor = ell #*(ell+1)

    #     chi2 = {}
    #     for i, ibin in enumerate(self.zbins):
    #         for ik, key in enumerate(self.psf_maps.keys()):
    #             ax = axes[ik, i]
    #             ax.axhline(y=0, c='0.8', lw=1)
    #             nrandom = hm.cls[(self.obs_name, key)][ibin]['random'].shape[0]
    #             for j in range(nrandom):
    #                 ax.plot(ell, factor*hm.cls[(self.obs_name, key)][ibin]['random'][j,which,:], c='r', alpha=max(0.01, 1./nrandom))
    #             if showchi2:
    #                 _chi2 = get_chi2_smoothcov(hm.cls[(self.obs_name, key)][ibin]['data'][which,:], hm.cls[(self.obs_name, key)][ibin]['random'][:,which,:])
    #                 label = '$\\chi^2_{{{:}}} = {:.2f}$ ($p={:.1e}$)'.format(len(ell), _chi2, scipy.stats.chi2.sf(_chi2, df=hm.b.get_n_bands()))
    #                 chi2[(self.obs_name, key,ibin)] = _chi2
    #             else:
    #                 label = None
    #             ax.plot(ell, factor*hm.cls[(self.obs_name, key)][ibin]['data'][which,:], label=label, c='b')
    #             ax.set_title(title.format(EB[EB_shear], ibin+1, EB[EB_psf], key), fontsize=12)
    #             ax.set_xlabel('$\\ell$')
    #             ax.set_ylabel(ylabel)
    #             ax.set_xlim(0, hm.b.lmax)
    #             if sqrtscale:
    #                 ax.set_xscale('squareroot')
    #                 ax.set_xticks(np.arange(0,np.sqrt(hm.b.lmax), np.ceil(np.sqrt(hm.b.lmax)/5.))**2)
    #             vmax = max(np.abs(ax.get_ylim()))
    #             ax.set_ylim(-vmax,+vmax)
    #             if showchi2:
    #                 ax.legend(loc=1)
    #     plt.tight_layout()

    #     make_directory(self.config.path_figures+'/'+self.name)
    #     figfile = os.path.join(self.config.path_figures, self.name, 'cls_cross_PSF_{}_{}_{}_nside{}.png'.format(self.obs_name, self.config.name, self.mode, self.nside))
    #     plt.savefig(figfile, dpi=300)

    #     if showchi2:
    #         return chi2


def apply_random_rotation(e1_in, e2_in):
    np.random.seed() # CRITICAL in multiple processes !
    rot_angle = np.random.rand(len(e1_in))*2*np.pi #no need for 2?
    cos = np.cos(rot_angle)
    sin = np.sin(rot_angle)
    e1_out = + e1_in * cos + e2_in * sin
    e2_out = - e1_in * sin + e2_in * cos
    return e1_out, e2_out

# @profile
def _randrot_maps(cat_e1, cat_e2, ipix, npix, bool_mask, count):
    e1_rot, e2_rot = apply_random_rotation(cat_e1, cat_e2)

    e1_map = np.zeros(npix, dtype=float)
    e2_map = np.zeros(npix, dtype=float)

    np.add.at(e1_map, ipix, e1_rot)
    np.add.at(e2_map, ipix, e2_rot)

    e1_map[bool_mask] /= count[bool_mask]
    e2_map[bool_mask] /= count[bool_mask]

    return e1_map, e2_map

def _randrot_maps_fast_sub(args):#, bool_mask, count):
    cat_e1, cat_e2, ipix, npix = args
    e1_rot, e2_rot = apply_random_rotation(cat_e1, cat_e2)

    e1_map = np.zeros(npix, dtype=float)
    e2_map = np.zeros(npix, dtype=float)

    np.add.at(e1_map, ipix, e1_rot)
    np.add.at(e2_map, ipix, e2_rot)
    
    return e1_map, e2_map

def _randrot_maps_fast_sub_proc(args, output):
    res = _randrot_maps_fast_sub(args)
    output.put(res)

# @ray.remote
# def _randrot_maps_fast_sub_ray(e12_map, cat_e1, cat_e2, ipix, npix):
#     e1_rot, e2_rot = apply_random_rotation(cat_e1, cat_e2)

#     np.add.at(e12_map[0], ipix, e1_rot)
#     np.add.at(e12_map[1], ipix, e2_rot)

#     return e12_map

# @profile
def _randrot_maps_fast(cat_e1, cat_e2, ipix, npix, bool_mask, count):
    n = max(1, int(multiprocessing.cpu_count()))
    cat_e1s = np.array_split(cat_e1, n)
    cat_e2s = np.array_split(cat_e2, n)
    ipixs = np.array_split(ipix, n)

    # Pool version
    # pool = multiprocessing.Pool(n)
    # # iterator = zip(cat_e1s, cat_e2s, ipixs)
    # iterator = [(*plop, npix) for plop in zip(cat_e1s, cat_e2s, ipixs)] #use a generator, so that nothing is computed before it's needed :)
    # res = pool.map(_randrot_maps_fast_sub, iterator, chunksize=1)
    # pool.close()
    # pool.join()

    # Process version
    output = multiprocessing.Queue()
    processes = [multiprocessing.Process(target=_randrot_maps_fast_sub_proc, args=((*x,npix), output)) for x in zip(cat_e1s, cat_e2s, ipixs)]
    for p in processes:
        p.start()
    res = [output.get() for p in processes]
    for p in processes:
        p.join()

    res = np.array(res)
    # print(res.shape)
    e1_map = np.sum(res[:,0], axis=0)
    e2_map = np.sum(res[:,1], axis=0)

    # # ray version
    # n = max(1, int(multiprocessing.cpu_count()*4))
    # cat_e1s = np.array_split(cat_e1, n)
    # cat_e2s = np.array_split(cat_e2, n)
    # ipixs = np.array_split(ipix, n)

    # e12_map = np.zeros((2,npix), dtype=float)
    # e12_map_id = ray.put(e12_map)
    # plop = [_randrot_maps_fast_sub_ray.remote(e12_map_id, e1, e2, i, npix) for e1, e2, i in zip(cat_e1s, cat_e2s, ipixs)]
    # print("plop")
    # e12_map = ray.get(e12_map_id) #ray.get(plop)
    # e1_map = e12_map[0]
    # e2_map = e12_map[1]

    e1_map[bool_mask] /= count[bool_mask]
    e2_map[bool_mask] /= count[bool_mask]

    return e1_map, e2_map

# @profile
def _randrot_field(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b):
    e1_map, e2_map = _randrot_maps_fast(cat_e1, cat_e2, ipix, npix, bool_mask, count)
    return nmt.NmtField(mask_apo, [e1_map, e2_map], purify_e=purify_e, purify_b=purify_b)

# @profile
def _randrot_cls(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, wsp):
    field = _randrot_field(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b)
    cls = compute_master(field, field, wsp)
    return cls

# def _randrot_cross_cls(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, template_fields, wsp_dir):
#     cls = {}
#     field = _randrot_field(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b)
#     for tempname, temp in template_fields.items():
#         cls[tempname] = compute_master(field, template_fields[tempname], wsp_dir[tempname])

#     return cls

# def _randrot_cross_PSF_cls(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, psf_fields, wsp_dir):
#     cls = {}
#     field = _randrot_field(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b)
#     for key in psf_fields.keys():
#         cls[key] = compute_master(field, psf_fields[key], wsp_dir[key])

#     return cls


def _multiproc_randrot_cls(nsamples, args, pos):
    cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, wsp_filename = args

    wsp = nmt.NmtWorkspace()
    wsp.read_from(wsp_filename)

    _cls = []
    for i in trange(nsamples, desc="[worker {:4d}]<{}>".format(pos,os.getpid()), position=pos, leave=False):
        _cls.append(_randrot_cls(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, wsp))

    return _cls


# def _multiproc_randrot_cross_cls(nsamples, args1, templates_dir, pos):
#     cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, nside, lmax, nlb = args1

#     b = nmt.NmtBin(nside, nlb=nlb, lmax=lmax)
#     field_0 = nmt.NmtField(mask_apo, [np.zeros_like(mask_apo), np.zeros_like(mask_apo)], purify_e=purify_e, purify_b=purify_b)

#     template_fields = {}
#     wsp_dir  = {}
#     for key, temp in templates_dir.items():
#         mask = np.logical_not((temp == hp.UNSEEN) | (temp == 0.0)) # kinda dangerous...
#         template_fields[key] = nmt.NmtField(mask, [temp])
#         wsp_dir[key] = nmt.NmtWorkspace()
#         wsp_dir[key].compute_coupling_matrix(field_0, template_fields[key], b)

#     cls = []
#     for i in self.prog(nsamples, desc="[worker {:4d}]<{}>".format(pos,os.getpid()), position=pos, leave=False):
#         cls.append(_randrot_cross_cls(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, template_fields, wsp_dir))

#     return cls

# def _multiproc_randrot_cross_PSF_cls(nsamples, args1, psf_maps, psf_mask_apo, pos):
#     cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, nside, lmax, nlb = args1

#     b = nmt.NmtBin(nside, nlb=nlb, lmax=lmax)
#     field_0 = nmt.NmtField(mask_apo, [np.zeros_like(mask_apo), np.zeros_like(mask_apo)], purify_e=purify_e, purify_b=purify_b)

#     psf_fields = {}
#     wsp_dir  = {}
#     for key, psf_map in psf_maps.items():
#         psf_fields[key] =  nmt.NmtField(psf_mask_apo, psf_maps[key], purify_e=purify_e, purify_b=purify_b)
#         wsp_dir[key] = nmt.NmtWorkspace()
#         wsp_dir[key].compute_coupling_matrix(field_0, psf_fields[key], b)

#     cls = []
#     for i in self.prog(nsamples, desc="[worker {:4d}]<{}>".format(pos,os.getpid()), position=pos, leave=False):
#         cls.append(_randrot_cross_PSF_cls(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, psf_fields, wsp_dir))

#     return cls
