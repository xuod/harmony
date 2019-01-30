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
        for ibin in trange(self.nzbins):
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
        for ibin in trange(self.nzbins):
            filename = os.path.join(self.data_dir, "source_s{}.fits".format(ibin+1))
            _cat = fits.open(filename)[1].data

            cat = {}
            cat['ra'] = _cat['RA']
            cat['dec'] = _cat['DEC']

            cat['e1'] = _cat['g1']
            cat['e2'] = -1.0 * _cat['g2']

            self.cats[ibin] = cat

    def _init_mock(self):
        for ibin in trange(self.nzbins):
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
        for ibin in trange(self.nzbins, desc='Harmony.make_maps'):
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
            for ibin in trange(self.nzbins, desc='Harmony.compute_ipix'):
                cat = self.cats[ibin]
                self.ipix[ibin] = hp.ang2pix(self.nside, (90-cat['dec'])*np.pi/180.0, cat['ra']*np.pi/180.0)

    def get_field(self, hm, ibin, include_templates=True):
        return nmt.NmtField(self.masks_apo[ibin], [self.maps[ibin]['e1'], self.maps[ibin]['e2']],
                            templates=self.templates if include_templates else None,
                            purify_e=hm.purify_e, purify_b=hm.purify_b)

    def get_randomized_fields(self, hm, ibin, nsamples=1):
        bool_mask = (self.maps[ibin]['count'] > 0.)
        fields = []
        for i in range(nsamples):
            e1_map, e2_map = _randrot_maps(self.cats[ibin]['e1'], self.cats[ibin]['e2'], self.ipix[ibin], self.npix, bool_mask)
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
                for i in trange(nrandom, desc='Harmony.compute_cls [bin {}]'.format(ibin)):
                    _cls.append(_randrot_cls(cat['e1'], cat['e2'], ipix, npix, bool_mask, mask_apo, count, hm.purify_e, hm.purify_b, wsp))

            else:
                args = (cat['e1'], cat['e2'], ipix, npix, bool_mask, mask_apo, count, hm.purify_e, hm.purify_b, self.nside, hm.lmax, hm.nlb)
                _multiple_results = [self.pool.apply_async(_multiproc_randrot_cls, (len(_x), args, pos+1)) for pos, _x in enumerate(np.array_split(range(nrandom), self.nproc)) if len(_x)>0]
                for res in tqdm(_multiple_results, desc='Harmony.compute_cls [bin {}]<{}>'.format(ibin, os.getpid()), position=0):
                    _cls += res.get()
                print("\n")

            cls['random'] = np.array(_cls)

        return cls

    def plot_auto_cls(self, hm, showchi2=False):
        cls = hm.cls[(self.obs_name, self.obs_name)]
        ell = hm.cls['ell']

        fig, axes = plt.subplots(self.nzbins, 3, figsize=(12, self.nzbins*3))
        axes = axes.reshape((self.nzbins, 3))
        idx_EB = [0, 1, 3]
        titles = ['EE', 'EB', 'BB']

        chi2 = {}

        for k in range(3):
            chi2[titles[k]] = {}
            axes[0,k].set_title(titles[k])
            axes[-1,k].set_xlabel('$\ell$')
            for i in range(self.nzbins):
                axes[i,0].set_ylabel('$C_\\ell$ (bin %i)'%(i+1))
                ax = axes[i, k]
                if 'random' in cls[0]:
                    nrandoms = len(cls[i]['random'])
                    for j in range(nrandoms):
                        ax.plot(ell, cls[i]['random'][j][idx_EB[k]], c='r', alpha=max(0.01,1./nrandoms))
                    ax.plot(ell, np.mean(cls[i]['random'][:,idx_EB[k],:], axis=0), c='r', ls='--')
                if showchi2:
                    _chi2 = get_chi2_smoothcov(cls[i]['true'][idx_EB[k]], cls[i]['random'][:,idx_EB[k],:])
                    label = '$\\chi^2_{{{:}}} = {:.2f}$ ($p={:.1e}$)'.format(len(ell), _chi2, scipy.stats.chi2.sf(_chi2, df=hm.b.get_n_bands()))
                    chi2[titles[k]][i] = _chi2
                else:
                    label=None
                ax.plot(ell, cls[i]['true'][idx_EB[k]], c='b', label=label)
                if showchi2:
                    ax.legend()

        plt.tight_layout()

        make_directory(self.config.path_figures+'/'+self.name)
        figfile = os.path.join(self.config.path_figures, self.name, 'cls_auto_{}_{}_{}_nside{}.png'.format(self.obs_name, self.config.name, self.mode, self.nside))
        plt.savefig(figfile, dpi=300)

        if showchi2:
            return chi2

    def plot_cls_BB_only(self, hm, showchi2=False):
        cls = hm.cls[(self.obs_name, self.obs_name)]
        ell = hm.cls['ell']

        fig, axes = plt.subplots(1, self.nzbins, figsize=(self.nzbins*4, 3))
        idx_EB = [0, 1, 3]
        titles = ['EE', 'EB', 'BB']

        chi2 = {}
        k = 2

        chi2['BB'] = {}
        for i in range(self.nzbins):
            ax = axes[i]
            ax.axhline(y=0, c='0.8', lw=1)
            ax.set_xlabel('$\\ell$')
            ax.set_ylabel('$C_\\ell ^{\\rm BB}$')
            ax.set_title('BB spectrum [bin %i]'%(i+1))
            if 'random' in cls[0]:
                nrandoms = len(cls[i]['random'])
                for j in range(nrandoms):
                    ax.plot(ell, cls[i]['random'][j][idx_EB[k]], c='r', alpha=max(0.01,1./nrandoms))
                ax.plot(ell, np.mean(cls[i]['random'][:,idx_EB[k],:], axis=0), c='r', ls='--')
            if showchi2:
                _chi2 = get_chi2_smoothcov(cls[i]['true'][idx_EB[k]], cls[i]['random'][:,idx_EB[k],:])
                label = '$\\chi^2_{{{:}}} = {:.2f}$ ($p={:.1e}$)'.format(len(ell), _chi2, scipy.stats.chi2.sf(_chi2, df=hm.b.get_n_bands()))
                chi2[titles[k]][i] = _chi2
            else:
                label=None
            ax.plot(ell, cls[i]['true'][idx_EB[k]], c='b', label=label)
            ax.set_xlim(0, hm.b.lmax)
            if showchi2:
                ax.legend()

        plt.tight_layout()

        make_directory(self.config.path_figures+'/'+self.name)
        figfile = os.path.join(self.config.path_figures, self.name, 'cls_BBonly_{}_{}_{}_nside{}.png'.format(self.obs_name, self.config.name, self.mode, self.nside))
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

def _randrot_maps(cat_e1, cat_e2, ipix, npix, bool_mask):
    e1_rot, e2_rot = apply_random_rotation(cat_e1, cat_e2)

    e1_map = np.zeros(npix, dtype=float)
    e2_map = np.zeros(npix, dtype=float)

    np.add.at(e1_map, ipix, e1_rot)
    np.add.at(e2_map, ipix, e2_rot)

    e1_map[bool_mask] /= count[bool_mask]
    e2_map[bool_mask] /= count[bool_mask]

    return e1_map, e2_map

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

    e1_map, e2_map = _randrot_maps(cat_e1, cat_e2, ipix, npix, bool_mask)

    field = nmt.NmtField(mask_apo, [e1_map, e2_map], purify_e=purify_e, purify_b=purify_b)

    cls = compute_master(field, field, wsp)

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


def _multiproc_randrot_cross_cls(nsamples, args1, args2 pos):
    cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, nside, lmax, nlb = args1


    wsp = nmt.NmtWorkspace()
    b = nmt.NmtBin(nside, nlb=nlb, lmax=lmax)
    field_0 = nmt.NmtField(mask_apo, [np.zeros_like(mask_apo), np.zeros_like(mask_apo)], purify_e=purify_e, purify_b=purify_b)
    wsp.compute_coupling_matrix(field_0, field_0, b)

    _cls = []
    for i in trange(nsamples, desc="[worker {:4d}]<{}>".format(pos,os.getpid()), position=pos, leave=False):
        _cls.append(_randrot_cls(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, wsp))

    return _cls
