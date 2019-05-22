from .observable import *

class Shear(Observable):
    def __init__(self, config, nside, mode, nzbins, data_dir='../data', *args, **kwargs):
        self.obs_name = 'galaxy_shear'
        self.map_names = ['count', 'e1', 'e2']

        super(Shear, self).__init__(config, nside, mode, nzbins, self.obs_name, self.map_names, *args, **kwargs)

        self.data_dir = data_dir

        self.cats = {}

        if mode.startswith('buzzard'):
            self._init_buzzard()

        if mode=='data_sub' or mode=='mastercat':
            self._init_data()

        if mode=='mock':
            self._init_mock()

    def _init_buzzard(self):
        # self.zlims = [(.2, .43), (.43,.63), (.64,.9), (.9, 1.3), (.2,1.3)][:self.nzbins]
        for ibin in tqdm(self.zbins):
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
        for ibin in tqdm(self.zbins):
            filename = os.path.join(self.data_dir, "source_s{}.fits".format(ibin+1))
            _cat = fits.open(filename)[1].data

            cat = {}
            cat['ra'] = _cat['RA']
            cat['dec'] = _cat['DEC']

            cat['e1'] = _cat['g1']
            cat['e2'] = -1.0 * _cat['g2']

            self.cats[ibin] = cat

    def _init_mock(self):
        for ibin in tqdm(self.zbins):
            filename = os.path.join(self.data_dir, "source_s{}.fits".format(ibin+1))
            _cat = fits.open(filename)[1].data

            cat = {}
            cat['ra'] = _cat['RA']
            cat['dec'] = _cat['DEC']

            cat['e1'] = _cat['e1']
            cat['e2'] = _cat['e2']

            self.cats[ibin] = cat

    def make_maps(self, save=True):
        keys = ['e1', 'e2']
        for ibin in tqdm(self.zbins, desc='{}.make_maps'.format(self.obs_name)):
            cat = self.cats[ibin]
            quantities, count, mask = ca.cosmo.make_healpix_map(cat['ra'], cat['dec'],
                                                    quantity=[cat[_x] for _x in keys],
                                                    nside=self.nside, fill_UNSEEN=True, mask=None, weight=None)
            for j, key in enumerate(keys):
                self.maps[ibin][key] = quantities[j]

            self.maps[ibin]['count'] = count
            self.masks[ibin] = mask

        if save:
            self.save_maps()

    def compute_ipix(self):
        if not hasattr(self, 'ipix'):
            self.ipix = {}
            for ibin in tqdm(self.zbins, desc='{}.compute_ipix'.format(self.obs_name)):
                cat = self.cats[ibin]
                self.ipix[ibin] = hp.ang2pix(self.nside, (90-cat['dec'])*np.pi/180.0, cat['ra']*np.pi/180.0)

    def get_field(self, hm, ibin, include_templates=True):
        return nmt.NmtField(self.masks_apo[ibin], [self.maps[ibin]['e1'], self.maps[ibin]['e2']],
                            templates=self._get_templates_array() if include_templates else None,
                            purify_e=hm.purify_e, purify_b=hm.purify_b)

    def get_randomized_fields(self, hm, ibin, nsamples=1):
        bool_mask = (self.maps[ibin]['count'] > 0.)
        self.compute_ipix()
        fields = []
        for i in trange(nsamples):
            e1_map, e2_map = _randrot_maps(self.cats[ibin]['e1'], self.cats[ibin]['e2'], self.ipix[ibin], self.npix, bool_mask, self.maps[ibin]['count'])
            field =  nmt.NmtField(self.masks_apo[ibin], [e1_map, e2_map],
                                    templates=None,
                                    purify_e=hm.purify_e, purify_b=hm.purify_b)
            fields.append(field)

        return fields

    def get_randomized_map(self, ibin):
        raise NotImplementedError

    def _compute_auto_cls(self, hm, ibin, nrandom=0, save=True):
        npix = self.npix

        cat = self.cats[ibin]
        mask_apo = self.masks_apo[ibin]

        wsp = nmt.NmtWorkspace()
        # field_0 = nmt.NmtField(mask_apo, [self.maps[ibin]['e1'], self.maps[ibin]['e2']],
        #                        templates=self.templates,
        #                        purify_e=hm.purify_e, purify_b=hm.purify_b)
        field_0 = self.get_field(hm, ibin)

        wsp.compute_coupling_matrix(field_0, field_0, hm.b)

        cls = {}
        cls['true'] = compute_master(field_0, field_0, wsp)

        if nrandom > 0:
            Nobj = len(cat)
            self.compute_ipix()

            count = np.zeros(npix, dtype=float)
            ipix = self.ipix[ibin] #hp.ang2pix(self.nside, (90-cat['dec'])*np.pi/180.0, cat['ra']*np.pi/180.0)
            np.add.at(count, ipix, 1.)
            bool_mask = (count > 0.)

            _cls = []

            if self.nproc==0:
                for i in trange(nrandom, desc='{}.compute_auto_cls [bin {}]'.format(self.obs_name, ibin)):
                    _cls.append(_randrot_cls(cat['e1'], cat['e2'], ipix, npix, bool_mask, mask_apo, count, hm.purify_e, hm.purify_b, wsp))

            else:
                args = (cat['e1'], cat['e2'], ipix, npix, bool_mask, mask_apo, count, hm.purify_e, hm.purify_b, self.nside, hm.lmax, hm.nlb)
                _multiple_results = [self.pool.apply_async(_multiproc_randrot_cls, (len(_x), args, pos+1)) for pos, _x in enumerate(np.array_split(range(nrandom), self.nproc)) if len(_x)>0]
                for res in tqdm(_multiple_results, desc='{}.compute_auto_cls [bin {}]<{}>'.format(self.obs_name, ibin, os.getpid()), position=0):
                    _cls += res.get()
                print("\n")

            cls['random'] = np.array(_cls)

        return cls

    def plot_auto_cls(self, hm, **kwargs):
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
                cls[k] = {}
                cls[k]['true'] = hm.cls[(self.obs_name, self.obs_name)][(zbin,zbin)]['true'][idx_EB[j]]
                cls[k]['random'] = hm.cls[(self.obs_name, self.obs_name)][(zbin,zbin)]['random'][:,idx_EB[j],:]

        return self.plot_cls(hm, cls, self.nzbins, 3, figname='auto', titles=titles, ylabels=ylabels)

        # cls = hm.cls[(self.obs_name, self.obs_name)]
        # ell = hm.cls['ell']

        # fig, axes = plt.subplots(self.nzbins, 3, figsize=(12, self.nzbins*3))
        # axes = axes.reshape((self.nzbins, 3))
        # titles = ['EE', 'EB', 'BB']

        # chi2 = {}

        # for k in range(3):
        #     chi2[titles[k]] = {}
        #     axes[0,k].set_title(titles[k])
        #     axes[-1,k].set_xlabel('$\ell$')
        #     for iz, i in enumerate(self.zbins):
        #         axes[iz,0].set_ylabel('$C_\\ell$ (bin %i)'%(i+1))
        #         ax = axes[iz, k]
        #         if 'random' in cls[0]:
        #             nrandoms = len(cls[i]['random'])
        #             for j in range(nrandoms):
        #                 ax.plot(ell, cls[i]['random'][j][idx_EB[k]], c='r', alpha=max(0.01,1./nrandoms))
        #             ax.plot(ell, np.mean(cls[i]['random'][:,idx_EB[k],:], axis=0), c='r', ls='--')
        #         if showchi2:
        #             _chi2 = get_chi2_smoothcov(cls[i]['true'][idx_EB[k]], cls[i]['random'][:,idx_EB[k],:])
        #             label = '$\\chi^2_{{{:}}} = {:.2f}$ ($p={:.1e}$)'.format(len(ell), _chi2, scipy.stats.chi2.sf(_chi2, df=hm.b.get_n_bands()))
        #             chi2[titles[k]][i] = _chi2
        #         else:
        #             label=None
        #         ax.plot(ell, cls[i]['true'][idx_EB[k]], c='b', label=label)
        #         if showchi2:
        #             ax.legend()

        # plt.tight_layout()

        # make_directory(self.config.path_figures+'/'+self.name)
        # figfile = os.path.join(self.config.path_figures, self.name, 'cls_auto_{}_{}_{}_nside{}.png'.format(self.obs_name, self.config.name, self.mode, self.nside))
        # plt.savefig(figfile, dpi=300)

        # if showchi2:
        #     return chi2

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
                cls[k]['true'] = np.copy(hm.cls[(self.obs_name, self.obs_name)][(zbin, zbin)]['true'][3])
                cls[k]['random'] = np.copy(hm.cls[(self.obs_name, self.obs_name)][(zbin, zbin)]['random'][:,3,:])
                if remove_Nl:
                    clr_r_m = np.mean(cls[k]['random'], axis=0)
                    cls[k]['true'] -= clr_r_m
                    cls[k]['random'] -= clr_r_m

        return self.plot_cls(hm, cls, 1, self.nzbins, figname='BB', titles=titles, ylabels=ylabels, **kwargs)

        # cls = hm.cls[(self.obs_name, self.obs_name)]
        # ell = hm.cls['ell']

        # fig, axes = plt.subplots(1, self.nzbins, figsize=(self.nzbins*4, 3))
        # idx_EB = [0, 1, 3]
        # titles = ['EE', 'EB', 'BB']

        # chi2 = {}
        # k = 2

        # chi2['BB'] = {}
        # for iz, i in enumerate(self.zbins):
        #     ax = axes[iz]
        #     ax.axhline(y=0, c='0.8', lw=1)
        #     ax.set_xlabel('$\\ell$')
        #     ax.set_ylabel('$C_\\ell ^{\\rm BB}$')
        #     ax.set_title('BB spectrum [bin %i]'%(i+1))
        #     if 'random' in cls[0]:
        #         nrandoms = len(cls[i]['random'])
        #         for j in range(nrandoms):
        #             ax.plot(ell, cls[i]['random'][j][idx_EB[k]], c='r', alpha=max(0.01,1./nrandoms))
        #         ax.plot(ell, np.mean(cls[i]['random'][:,idx_EB[k],:], axis=0), c='r', ls='--')
        #     if showchi2:
        #         _chi2 = get_chi2_smoothcov(cls[i]['true'][idx_EB[k]], cls[i]['random'][:,idx_EB[k],:])
        #         label = '$\\chi^2_{{{:}}} = {:.2f}$ ($p={:.1e}$)'.format(len(ell), _chi2, scipy.stats.chi2.sf(_chi2, df=hm.b.get_n_bands()))
        #         chi2[titles[k]][i] = _chi2
        #     else:
        #         label=None
        #     ax.plot(ell, cls[i]['true'][idx_EB[k]], c='b', label=label)
        #     ax.set_xlim(0, hm.b.lmax)
        #     if showchi2:
        #         ax.legend()

        # plt.tight_layout()

        # make_directory(self.config.path_figures+'/'+self.name)
        # figfile = os.path.join(self.config.path_figures, self.name, 'cls_BBonly_{}_{}_{}_nside{}.png'.format(self.obs_name, self.config.name, self.mode, self.nside))
        # plt.savefig(figfile, dpi=300)

        # if showchi2:
        #     return chi2

    def _compute_cross_template_cls(self, hm, ibin, nrandom=0, save=True):
        npix = self.npix
        cat = self.cats[ibin]
        mask_apo = self.masks_apo[ibin]

        logging.info("_compute_cross_template_cls: Making field_0")
        field_0 = self.get_field(hm, ibin, include_templates=False)

        logging.info("_compute_cross_template_cls: Making template_fields and wsp_dir")
        template_fields = {}
        wsp_dir  = {}
        for tempname, temp in tqdm(self.templates_dir.items(), desc='{}.compute_cross_template_cls [bin {}]'.format(self.obs_name, ibin)):
            mask = np.logical_not((temp == hp.UNSEEN) | (temp == 0.0)) # kinda dangerous...
            template_fields[tempname] = nmt.NmtField(mask, [temp])
            wsp_dir[tempname] = nmt.NmtWorkspace()
            wsp_dir[tempname].compute_coupling_matrix(field_0, template_fields[tempname], hm.b)
            hm.cls[(self.obs_name, tempname)][ibin] = {}
            hm.cls[(self.obs_name, tempname)][ibin]['true'] = compute_master(field_0, template_fields[tempname], wsp_dir[tempname])
            if nrandom > 0:
                hm.cls[(self.obs_name, tempname)][ibin]['random'] = []

        if nrandom > 0:
            Nobj = len(cat)
            self.compute_ipix()

            count = np.zeros(npix, dtype=float)
            ipix = self.ipix[ibin] #hp.ang2pix(self.nside, (90-cat['dec'])*np.pi/180.0, cat['ra']*np.pi/180.0)
            np.add.at(count, ipix, 1.)
            bool_mask = (count > 0.)

            if self.nproc==0:
                cls_r = []
                for i in trange(nrandom, desc='{}.compute_cross_template_cls [bin {}]'.format(self.obs_name, ibin)):
                    cls_r.append(_randrot_cross_cls(cat['e1'], cat['e2'], ipix, npix, bool_mask, mask_apo, count, hm.purify_e, hm.purify_b, template_fields, wsp_dir))
                for tempname in self.templates_dir.keys():
                    hm.cls[(self.obs_name, tempname)][ibin]['random'] = np.array([_x[tempname] for _x in cls_r])

            else:
                args = (cat['e1'], cat['e2'], ipix, npix, bool_mask, mask_apo, count, hm.purify_e, hm.purify_b, self.nside, hm.lmax, hm.nlb)
                _multiple_results = [self.pool.apply_async(_multiproc_randrot_cross_cls, (len(_x), args, self.templates_dir, pos+1)) for pos, _x in enumerate(np.array_split(range(nrandom), self.nproc)) if len(_x)>0]
                cls_r = []
                for res in tqdm(_multiple_results, desc='{}.compute_cross_template_cls [bin {}]<{}>'.format(self.obs_name, ibin, os.getpid()), position=0):
                    cls_r += res.get()

                for tempname in self.templates_dir.keys():
                    hm.cls[(self.obs_name, tempname)][ibin]['random'] = np.array([_x[tempname] for _x in cls_r])

                print("\n")

    def plot_cross_template_cls(self, hm, showchi2=False, EB=0):
        ntemp = len(list(self.templates_dir.keys()))

        fig, axes = plt.subplots(ntemp, self.nzbins, figsize=(4*self.nzbins, 3*ntemp))
        ell = hm.b.get_effective_ells()

        # EB = 0 #0 for E, 1 for B
        ylabels = ['$\\ell (\\ell+1) C_\\ell^{{\\rm syst} \\times \\gamma_{\\rm %s}}$'%s for s in ['E', 'B']]
        gamma_label = ['$\\gamma_{\\rm %s}$'%s for s in ['E', 'B']]

        factor = ell*(ell+1)

        import scipy

        chi2 = {}
        for i, ibin in enumerate(self.zbins):
            for ik, key in enumerate(self.templates_dir.keys()):
                ax = axes[ik, i]
                ax.axhline(y=0, c='0.8', lw=1)
                nrandom = hm.cls[(self.obs_name, key)][ibin]['random'].shape[0]
                for j in range(nrandom):
                    ax.plot(ell, factor*hm.cls[(self.obs_name, key)][ibin]['random'][j,EB,:], c='r', alpha=max(0.01, 1./nrandom))
                if showchi2:
                    _chi2 = get_chi2_smoothcov(hm.cls[(self.obs_name, key)][ibin]['true'][EB,:], hm.cls[(self.obs_name, key)][ibin]['random'][:,EB,:])
                    label = '$\\chi^2_{{{:}}} = {:.2f}$ ($p={:.1e}$)'.format(len(ell), _chi2, scipy.stats.chi2.sf(_chi2, df=hm.b.get_n_bands()))
                    chi2[(self.obs_name, key,ibin)] = _chi2
                else:
                    label = None
                ax.plot(ell, factor*hm.cls[(self.obs_name, key)][ibin]['true'][EB,:], label=label, c='b')
                ax.set_title(gamma_label[EB] + ' [bin %i] $\\times$ '%(ibin+1) + key, fontsize=8)
                ax.set_xlabel('$\\ell$')
                ax.set_ylabel(ylabels[EB])
                ax.set_xlim(0, hm.b.lmax)
                vmax = max(np.abs(ax.get_ylim()))
                ax.set_ylim(-vmax,+vmax)
                if showchi2:
                    ax.legend(loc=1)
        plt.tight_layout()

        make_directory(self.config.path_figures+'/'+self.name)
        figfile = os.path.join(self.config.path_figures, self.name, 'cls_cross_templates_{}_{}_{}_nside{}.png'.format(self.obs_name, self.config.name, self.mode, self.nside))
        plt.savefig(figfile, dpi=300)

        if showchi2:
            return chi2

    def _compute_cross_PSF_cls(self, hm, ibin, nrandom=0, save=True):
        npix = self.npix
        cat = self.cats[ibin]
        mask_apo = self.masks_apo[ibin]

        logging.info("_compute_cross_PSF_cls: Making field_0")
        field_0 = self.get_field(hm, ibin, include_templates=False)

        logging.info("_compute_cross_PSF_cls: Making template_fields and wsp_dir")
        wsp_dir  = {}
        for key, psf_field in tqdm(self.psf_fields.items(), desc='{}._compute_cross_PSF_cls [bin {}]'.format(self.obs_name, ibin)):
            wsp_dir[key] = nmt.NmtWorkspace()
            wsp_dir[key].compute_coupling_matrix(field_0, psf_field, hm.b)
            hm.cls[(self.obs_name, key)][ibin] = {}
            hm.cls[(self.obs_name, key)][ibin]['true'] = compute_master(field_0, psf_field, wsp_dir[key])
            if nrandom > 0:
                hm.cls[(self.obs_name, key)][ibin]['random'] = []

        if nrandom > 0:
            Nobj = len(cat)
            self.compute_ipix()

            count = np.zeros(npix, dtype=float)
            ipix = self.ipix[ibin] #hp.ang2pix(self.nside, (90-cat['dec'])*np.pi/180.0, cat['ra']*np.pi/180.0)
            np.add.at(count, ipix, 1.)
            bool_mask = (count > 0.)

            if self.nproc==0:
                cls_r = []
                for i in trange(nrandom, desc='{}._compute_cross_PSF_cls [bin {}]'.format(self.obs_name, ibin)):
                    cls_r.append(_randrot_cross_PSF_cls(cat['e1'], cat['e2'], ipix, npix, bool_mask, mask_apo, count, hm.purify_e, hm.purify_b, self.psf_fields, wsp_dir))
                for key in self.psf_maps.keys():
                    hm.cls[(self.obs_name, key)][ibin]['random'] = np.array([_x[key] for _x in cls_r])

            else:
                args = (cat['e1'], cat['e2'], ipix, npix, bool_mask, mask_apo, count, hm.purify_e, hm.purify_b, self.nside, hm.lmax, hm.nlb)
                _multiple_results = [self.pool.apply_async(_multiproc_randrot_cross_PSF_cls, (len(_x), args, self.psf_maps, self.psf_mask_apo, pos+1)) for pos, _x in enumerate(np.array_split(range(nrandom), self.nproc)) if len(_x)>0]
                cls_r = []
                for res in tqdm(_multiple_results, desc='{}._compute_cross_PSF_cls [bin {}]<{}>'.format(self.obs_name, ibin, os.getpid()), position=0):
                    cls_r += res.get()

                for key in self.psf_maps.keys():
                    hm.cls[(self.obs_name, key)][ibin]['random'] = np.array([_x[key] for _x in cls_r])

                print("\n")

    def plot_cross_PSF_cls(self, hm, showchi2=False, EB_shear=0, EB_psf=0, sqrtscale=False):
        ntemp = len(list(self.psf_maps.keys()))

        fig, axes = plt.subplots(ntemp, self.nzbins, figsize=(4*self.nzbins, 3*ntemp))
        ell = hm.b.get_effective_ells()

        EB = ['E', 'B']
        ylabel = '$\\ell C_\\ell$' # (\\ell+1)
        # gamma_label = ['$\\gamma_{\\rm %s}$'%s for s in ['E', 'B']]
        which = np.array([[0,1],[2,3]])
        title = '$\\gamma_{{\\rm {}}}^{{\\rm gal}} \\ {{\\rm [bin {}]}} \\times \\gamma_{{\\rm {}}}^{{\\rm \\ PSF {}}}$'

        which = np.array([[0,1],[2,3]])[EB_shear,EB_psf]

        factor = ell #*(ell+1)

        chi2 = {}
        for i, ibin in enumerate(self.zbins):
            for ik, key in enumerate(self.psf_maps.keys()):
                ax = axes[ik, i]
                ax.axhline(y=0, c='0.8', lw=1)
                nrandom = hm.cls[(self.obs_name, key)][ibin]['random'].shape[0]
                for j in range(nrandom):
                    ax.plot(ell, factor*hm.cls[(self.obs_name, key)][ibin]['random'][j,which,:], c='r', alpha=max(0.01, 1./nrandom))
                if showchi2:
                    _chi2 = get_chi2_smoothcov(hm.cls[(self.obs_name, key)][ibin]['true'][which,:], hm.cls[(self.obs_name, key)][ibin]['random'][:,which,:])
                    label = '$\\chi^2_{{{:}}} = {:.2f}$ ($p={:.1e}$)'.format(len(ell), _chi2, scipy.stats.chi2.sf(_chi2, df=hm.b.get_n_bands()))
                    chi2[(self.obs_name, key,ibin)] = _chi2
                else:
                    label = None
                ax.plot(ell, factor*hm.cls[(self.obs_name, key)][ibin]['true'][which,:], label=label, c='b')
                ax.set_title(title.format(EB[EB_shear], ibin+1, EB[EB_psf], key), fontsize=12)
                ax.set_xlabel('$\\ell$')
                ax.set_ylabel(ylabel)
                ax.set_xlim(0, hm.b.lmax)
                if sqrtscale:
                    ax.set_xscale('squareroot')
                    ax.set_xticks(np.arange(0,np.sqrt(hm.b.lmax), np.ceil(np.sqrt(hm.b.lmax)/5.))**2)
                vmax = max(np.abs(ax.get_ylim()))
                ax.set_ylim(-vmax,+vmax)
                if showchi2:
                    ax.legend(loc=1)
        plt.tight_layout()

        make_directory(self.config.path_figures+'/'+self.name)
        figfile = os.path.join(self.config.path_figures, self.name, 'cls_cross_PSF_{}_{}_{}_nside{}.png'.format(self.obs_name, self.config.name, self.mode, self.nside))
        plt.savefig(figfile, dpi=300)

        if showchi2:
            return chi2


def apply_random_rotation(e1_in, e2_in):
    np.random.seed() # CRITICAL in multiple processes !
    rot_angle = np.random.rand(len(e1_in))*2*np.pi #no need for 2?
    cos = np.cos(rot_angle)
    sin = np.sin(rot_angle)
    e1_out = + e1_in * cos + e2_in * sin
    e2_out = - e1_in * sin + e2_in * cos
    return e1_out, e2_out

def _randrot_maps(cat_e1, cat_e2, ipix, npix, bool_mask, count):
    e1_rot, e2_rot = apply_random_rotation(cat_e1, cat_e2)

    e1_map = np.zeros(npix, dtype=float)
    e2_map = np.zeros(npix, dtype=float)

    np.add.at(e1_map, ipix, e1_rot)
    np.add.at(e2_map, ipix, e2_rot)

    e1_map[bool_mask] /= count[bool_mask]
    e2_map[bool_mask] /= count[bool_mask]

    return e1_map, e2_map

def _randrot_field(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b):
    e1_map, e2_map = _randrot_maps(cat_e1, cat_e2, ipix, npix, bool_mask, count)
    return nmt.NmtField(mask_apo, [e1_map, e2_map], purify_e=purify_e, purify_b=purify_b)

def _randrot_cls(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, wsp):
    # e1_rot, e2_rot = apply_random_rotation(cat_e1, cat_e2)
    #
    # e1_map = np.zeros(npix, dtype=float)
    # e2_map = np.zeros(npix, dtype=float)
    #
    # np.add.at(e1_map, ipix, e1_rot)
    # np.add.at(e2_map, ipix, e2_rot)
    #
    # e1_map[bool_mask] /= count[bool_mask]
    # e2_map[bool_mask] /= count[bool_mask]

    # e1_map, e2_map = _randrot_maps(cat_e1, cat_e2, ipix, npix, bool_mask, count)

    # field = nmt.NmtField(mask_apo, [e1_map, e2_map], purify_e=purify_e, purify_b=purify_b)
    field = _randrot_field(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b)

    cls = compute_master(field, field, wsp)

    return cls

def _randrot_cross_cls(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, template_fields, wsp_dir):
    # e1_rot, e2_rot = apply_random_rotation(cat_e1, cat_e2)
    #
    # e1_map = np.zeros(npix, dtype=float)
    # e2_map = np.zeros(npix, dtype=float)
    #
    # np.add.at(e1_map, ipix, e1_rot)
    # np.add.at(e2_map, ipix, e2_rot)
    #
    # e1_map[bool_mask] /= count[bool_mask]
    # e2_map[bool_mask] /= count[bool_mask]

    # e1_map, e2_map = _randrot_maps(cat_e1, cat_e2, ipix, npix, bool_mask, count)

    # field = nmt.NmtField(mask_apo, [e1_map, e2_map], purify_e=purify_e, purify_b=purify_b)

    # cls = compute_master(field, field_t, wsp)

    cls = {}
    field = _randrot_field(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b)
    for tempname, temp in template_fields.items():
        cls[tempname] = compute_master(field, template_fields[tempname], wsp_dir[tempname])

    return cls

def _randrot_cross_PSF_cls(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, psf_fields, wsp_dir):
    cls = {}
    field = _randrot_field(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b)
    for key in psf_fields.keys():
        cls[key] = compute_master(field, psf_fields[key], wsp_dir[key])

    return cls

# def _multiproc_randrot_maps(cat, ipix, npix, bool_mask, count):
#     e1_rot, e2_rot = apply_random_rotation(cat['e1'], cat['e2'])
#
#     e1_map = np.zeros(npix, dtype=float)
#     e2_map = np.zeros(npix, dtype=float)
#
#     np.add.at(e1_map, ipix, e1_rot)
#     np.add.at(e2_map, ipix, e2_rot)
#
#     e1_map[bool_mask] /= count[bool_mask]
#     e2_map[bool_mask] /= count[bool_mask]
#
#     return e1_map, e2_map


def _multiproc_randrot_cls(nsamples, args, pos):
    cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, nside, lmax, nlb = args

    wsp = nmt.NmtWorkspace()
    b = nmt.NmtBin(nside, nlb=nlb, lmax=lmax)
    field_0 = nmt.NmtField(mask_apo, [np.zeros_like(mask_apo), np.zeros_like(mask_apo)], purify_e=purify_e, purify_b=purify_b)
    wsp.compute_coupling_matrix(field_0, field_0, b)

    _cls = []
    for i in trange(nsamples, desc="[worker {:4d}]<{}>".format(pos,os.getpid()), position=pos, leave=False):
        _cls.append(_randrot_cls(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, wsp))

    return _cls


def _multiproc_randrot_cross_cls(nsamples, args1, templates_dir, pos):
    cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, nside, lmax, nlb = args1

    b = nmt.NmtBin(nside, nlb=nlb, lmax=lmax)
    field_0 = nmt.NmtField(mask_apo, [np.zeros_like(mask_apo), np.zeros_like(mask_apo)], purify_e=purify_e, purify_b=purify_b)

    template_fields = {}
    wsp_dir  = {}
    for key, temp in templates_dir.items():
        mask = np.logical_not((temp == hp.UNSEEN) | (temp == 0.0)) # kinda dangerous...
        template_fields[key] = nmt.NmtField(mask, [temp])
        wsp_dir[key] = nmt.NmtWorkspace()
        wsp_dir[key].compute_coupling_matrix(field_0, template_fields[key], b)

    cls = []
    for i in trange(nsamples, desc="[worker {:4d}]<{}>".format(pos,os.getpid()), position=pos, leave=False):
        cls.append(_randrot_cross_cls(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, template_fields, wsp_dir))

    return cls

def _multiproc_randrot_cross_PSF_cls(nsamples, args1, psf_maps, psf_mask_apo, pos):
    cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, nside, lmax, nlb = args1

    b = nmt.NmtBin(nside, nlb=nlb, lmax=lmax)
    field_0 = nmt.NmtField(mask_apo, [np.zeros_like(mask_apo), np.zeros_like(mask_apo)], purify_e=purify_e, purify_b=purify_b)

    psf_fields = {}
    wsp_dir  = {}
    for key, psf_map in psf_maps.items():
        psf_fields[key] =  nmt.NmtField(psf_mask_apo, psf_maps[key], purify_e=purify_e, purify_b=purify_b)
        wsp_dir[key] = nmt.NmtWorkspace()
        wsp_dir[key].compute_coupling_matrix(field_0, psf_fields[key], b)

    cls = []
    for i in trange(nsamples, desc="[worker {:4d}]<{}>".format(pos,os.getpid()), position=pos, leave=False):
        cls.append(_randrot_cross_PSF_cls(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, psf_fields, wsp_dir))

    return cls
