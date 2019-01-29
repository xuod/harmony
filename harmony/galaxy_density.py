from .observable import *

class Galaxy(Observable):
    def __init__(self, config, nside, mode, nzbins, data_dir='../data', mask_dir='../masks', *args, **kwargs):

        self.obs_name = 'galaxy_density'
        self.map_names = ['count', 'density', 'completeness']

        super(Galaxy, self).__init__(config, nside, mode, nzbins, self.obs_name, self.map_names, *args, **kwargs)

        self.data_dir = data_dir
        self.mask_dir = mask_dir

        self.cats = {}

        if mode=='redmagic':
            self.init_redmagic(self.nzbins)

    def init_redmagic(self, nzbins):
        self.zbins = [(0.15,0.30), (0.30,0.45), (0.45,0.60), (0.60,0.75), (0.75,0.90)]
        self.zbins = self.zbins[:nzbins]

        catdir = '/global/cscratch1/sd/troxel/cats_des_y3/'
        basename = 'y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_'
        cats_name = ['redmagic_highdens_0.5', 'redmagic_highlum_1.0', 'redmagic_higherlum_1.5'] #
        mask_ext = '_vlim_zmask.fit'
        cats_ext = ['-10.fit', '-04.fit', '-01.fit']

        which_cat_zbins = [0,0,0,1,2]

        for ibin in trange(len(self.zbins), desc='Galaxy.init_redmagic'):
            self.cats[ibin] = fits.open(os.path.join(self.data_dir, 'redmagic_bin{}.fits'.format(ibin)))[1].data
            self.masks[ibin] = hp.read_map(os.path.join(self.mask_dir, basename+cats_name[which_cat_zbins[ibin]]+'_binary_nside%i.fits'%(self.nside)), verbose=False)
            comp = hp.read_map(os.path.join(self.mask_dir, basename+cats_name[which_cat_zbins[ibin]]+'_FRACGOOD_nside%i.fits'%(self.nside)), verbose=False)
            comp[comp == hp.UNSEEN] = 0.0
            self.maps[ibin]['completeness'] = comp

    def make_maps(self, save=True):
        for ibin in trange(self.nzbins, desc='Galaxy.make_maps'):
            cat = self.cats[ibin]
            _, count, _ = ca.cosmo.make_healpix_map(cat['ra'], cat['dec'], None, self.nside,
                                            mask=self.masks[ibin],
                                            weight=None, fill_UNSEEN=False, return_extra=False)

            density = ca.cosmo.count2density(count, mskfrac_map=self.maps[ibin]['completeness'], mask=self.masks[ibin])

            self.maps[ibin]['count'] = count
            self.maps[ibin]['density'] = density

        if save:
            self.save_maps()

    def get_field(self, hm, ibin):
        return nmt.NmtField(self.masks_apo[ibin], [self.maps[ibin]['density']], templates=self.templates, purify_e=hm.purify_e, purify_b=hm.purify_b)

    def _compute_auto_cls(self, hm, ibin, nrandom=0, save=True):
        npix = hp.nside2npix(self.nside)

        mask_apo = self.masks_apo[ibin]

        wsp = nmt.NmtWorkspace()
        field_0 = get_field(self, hm, ibin) #nmt.NmtField(mask_apo, [self.maps[ibin]['density']], templates=self.templates, purify_e=hm.purify_e, purify_b=hm.purify_b)

        wsp.compute_coupling_matrix(field_0, field_0, hm.b)

        cls = {}
        cls['true'] = compute_master(field_0, field_0, wsp)

        if nrandom > 0:
            Nobj = len(self.cats[ibin])

            _cls = []

            for i in trange(nrandom, desc='Galaxy.compute_cls [bin {}]'.format(ibin)):
                count = np.zeros(npix, dtype=float)
                ipix_r = random_pos(self.masks[ibin].astype(float), Nobj)
                np.add.at(count, ipix_r, 1.)

                density = ca.cosmo.count2density(count, mskfrac_map=self.masks[ibin], mask=self.masks[ibin])

                field_r = nmt.NmtField(mask_apo, [density], templates=None, purify_e=hm.purify_e, purify_b=hm.purify_b)

                _cls.append(compute_master(field_r, field_r, wsp))

            cls['random'] = np.array(_cls)

        return cls


        # if save:
        #     self.save_cls()

    # def save_cls(self):
    #     make_directory(self.config.path_output+'/'+self.name)
    #     filename = os.path.join(self.config.path_output, self.name, 'cls_{}_{}_nside{}.pickle'.format(self.config.mode, self.config.name, self.nside))
    #     pickle.dump(cls, open(filename, mode='wb'))
    #
    # def load_cls(self):
    #     filename = os.path.join(self.config.path_output, self.name, 'cls_{}_{}_nside{}.pickle'.format(self.config.mode, self.config.name, self.nside))
    #     try:
    #         cls = pickle.load(open(filename, mode='rb'))
    #     except FileNotFoundError:
    #         print("Cls file does not exists: {}".format(filename))

    def plot_auto_cls(self, hm, showchi2=False, blindyaxis=False):
        cls = hm.cls[(self.obs_name, self.obs_name)]

        fig, axes = plt.subplots(1, self.nzbins, figsize=(self.nzbins*4, 3))

        ell = hm.cls['ell']

        for k in range(self.nzbins):
            ax = axes[k]

            ngals = len(self.cats[k])
            fsky = np.sum(self.maps[k]['completeness']) / hp.nside2npix(self.nside)
            nbar = ngals / (4 * np.pi * fsky)
            ax.axhline(y=1./nbar, c='r', ls=':')

            if 'random' in cls[0]:
                nrandoms = cls[k]['random'].shape[0]
                for j in range(nrandoms):
                    ax.plot(ell, cls[k]['random'][j,0,:], c='r', alpha=max(0.01, 1./nrandoms))
                ax.plot(ell, np.mean(cls[k]['random'][:,0,:], axis=0), c='r', ls='--')

            if 'true' in cls[0]:
                ax.plot(ell, cls[k]['true'][0], c='b')

            ax.set_yscale('log')
            ax.set_xlabel('$\ell$')
            ax.set_ylabel('$C_\\ell$ (bin %i)'%(k+1))
            ax.set_xlim(0)
            if blindyaxis:
                ax.set_yticks([])

        plt.tight_layout()

        make_directory(self.config.path_figures+'/'+self.name)
        figfile = os.path.join(self.config.path_figures, self.name, 'cls_auto_{}_{}_{}_nside{}.png'.format(self.obs_name, self.config.name, self.mode, self.nside))
        plt.savefig(figfile, dpi=300)

def random_pos(completeness, nobj):
    return np.random.choice(len(completeness), size=nobj, replace=True, p=completeness*1./np.sum(completeness))
