import healpy as hp
import pymaster as nmt
from tqdm.auto import tqdm, trange
from astropy.io import fits
import os, sys
import castor as ca
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing
from .utils import prog, make_directory, compute_master
from .observable import Observable
import pickle
import numpy as np
import itertools
import warnings
import twopoint
import numpy as np

class Harmony(object):
    def __init__(self, config, nside, b=None, nlb=None, nproc=0, save_cls=True, save_wsp=True, verbose=True):
        self.config = config
        self.name = config.name
        self.nside = nside

        self.do_save_cls = save_cls
        self.do_save_wsp = save_wsp

        if b is None:
            assert nlb is not None
            self.nlb = nlb
            self.b = nmt.NmtBin(self.nside, nlb=nlb)
        else:
            self.b = b
            
        self.ell = self.b.get_effective_ells()
        self.cls = {}
        self.cls['ell'] = self.ell
        self.wsp = {}

        self.nproc = nproc
        if nproc > 1:
            self.pool = multiprocessing.Pool(nproc)
        
        make_directory(self.config.path_output+'/'+self.name)
        make_directory(self.config.path_wsp+'/'+self.name)

        self.verbose = verbose
        self.prog = prog(verbose)

    def clear_cls_wsp(self):
        self.cls = {}
        self.cls['ell'] = self.ell
        self.wsp = {}

    @staticmethod
    def get_pairs(obs1, obs2=None, auto_only=False):        
        if (obs2==obs1) or obs2 is None: # same observable
            if auto_only:
                pairs = [(i1,i1) for i1 in obs1.zbins]
            else:
                pairs = [(i1,i2) for i,i1 in enumerate(obs1.zbins) for i2 in obs1.zbins[:i+1]]
        else:
            assert not auto_only
            pairs = [(i1,i2) for i1 in obs1.zbins for i2 in obs2.zbins]
        
        return pairs

    def check_obs(self, obs1, obs2):
        key = (obs1.obs_name, obs2.obs_name)
        if key not in self.cls.keys():
            self.cls[key] = {}
        if key not in self.wsp.keys():
            self.wsp[key] = {}    


    #########################
    # Manage wsp and cls        
    
    def get_workspace_filename(self, obs1, obs2, i1, i2):
        if isinstance(obs1, Observable) and isinstance(obs2, Observable):
            suffix = '{}_{}_{}_{}'.format(obs1.obs_name, obs2.obs_name, i1, i2)
        elif isinstance(obs1, str) and isinstance(obs2, str):
            suffix = '{}_{}_{}_{}'.format(obs1, obs2, i1, i2)
        else:
            raise ValueError
        filename = os.path.join(self.config.path_wsp, self.name, 'wsp_{}_nside{}_{}.bin'.format(self.config.name, self.nside, suffix))
        return filename

    def load_workspace(self, obs1, obs2, i1, i2):
        filename = self.get_workspace_filename(obs1, obs2, i1, i2)
        self.check_obs(obs1, obs2)

        if os.path.isfile(filename):
            wsp = nmt.NmtWorkspace()
            wsp.read_from(filename)
            self.wsp[(obs1.obs_name, obs2.obs_name)][(i1,i2)] = wsp
            return wsp
        else:
            raise RuntimeError("Workspace file does not exist: ", filename)

    def load_workspace_bpws(self, obs1, obs2, i1, i2):
        filename = self.get_workspace_filename(obs1, obs2, i1, i2)
        filename_bpws = filename[:-4]+'_bpws.npy'

        if os.path.isfile(filename_bpws):
            bpws = np.load(filename_bpws)
            return bpws
        else:
            raise RuntimeError("Workspace file does not exist: ", filename_bpws)

    def load_all_workspaces(self, obs1, obs2=None, auto_only=False):
        pairs = self.get_pairs(obs1, obs2=obs2, auto_only=auto_only)
        if obs2 is None:
            obs2 = obs1

        for i1,i2 in self.prog(pairs, desc='Harmony.load_all_workspaces [obs1:{}, obs2={}]'.format(obs1.obs_name, obs2.obs_name)):
            self.load_workspace(obs1, obs2, i1, i2)
        
    def _save_workspace(self, wsp, obs1, obs2, i1, i2, bpws_only=False):
        filename = self.get_workspace_filename(obs1, obs2, i1, i2)
        np.save(filename[:-4]+'_bpws.npy', wsp.get_bandpower_windows())
        if not bpws_only:
            wsp.write_to(filename)

    def compute_workspace(self, obs1, obs2, i1, i2, wsp=None, **kwargs):
        self.check_obs(obs1, obs2)

        field1 = obs1.get_field(i1)
        field2 = obs2.get_field(i2)

        if wsp is None:
            wsp = nmt.NmtWorkspace()
        wsp.compute_coupling_matrix(field1, field2, self.b, **kwargs)
        
        return wsp

    def prepare_workspace(self, obs1, obs2, i1, i2, **kwargs):
        wsp = self.compute_workspace(obs1, obs2, i1, i2, **kwargs)

        self.wsp[(obs1.obs_name, obs2.obs_name)][(i1,i2)] = wsp

        if self.do_save_wsp:
            self._save_workspace(wsp, obs1, obs2, i1, i2, bpws_only=(self.do_save_wsp=='bpws_only'))
        
        return wsp
    
    def prepare_all_workspaces(self, obs1, obs2=None, auto_only=False, **kwargs):
        pairs = self.get_pairs(obs1, obs2=obs2, auto_only=auto_only)
        if obs2 is None:
            obs2 = obs1

        for i1,i2 in self.prog(pairs, desc='Harmony.prepare_all_workspaces [obs1:{}, obs2={}]'.format(obs1.obs_name, obs2.obs_name)):
            # self.wsp[(obs1.obs_name, obs2.obs_name)][(i1,i2)] = self.prepare_workspace(obs1, obs2, i1, i2)
            self.prepare_workspace(obs1, obs2, i1, i2, **kwargs)

    def get_workspace(self, obs1, obs2, i1, i2):
        if isinstance(obs1, Observable):
            obs1_name = obs1.obs_name
            obs2_name = obs2.obs_name
        else:
            obs1_name = obs1
            obs2_name = obs2

        # WRONG ! Even if obs1=obs2, EB and BE are also inverted between bins i1 and i2
        # try:
        #     wsp = self.wsp[(obs1_name, obs2_name)][(i1,i2)]
        # except KeyError:
        #     if obs1 == obs2:
        #         try:
        #             wsp = self.wsp[(obs2_name, obs1_name)][(i2,i1)]
        #         except:
        #             print("[get_workspace] workspace for observables [{},{}] with bins [{},{}] does not exist.".format(obs1_name, obs2_name, i1, i2))
        #             wsp = None
        #     else:
        #         wsp = None # for different observables with spin 2, the spectra are EE, EB, BE and BB, so EB and BE would be inverted...

        wsp = self.wsp[(obs1_name, obs2_name)][(i1,i2)]
        
        if wsp is not None:
            return wsp
        else:
            raise KeyError

    def get_workspace_bpws(self, obs1, obs2, i1, i2):
        try:
            wsp = self.get_workspace(obs1, obs2, i1, i2)
            return wsp.get_bandpower_windows()
        except KeyError:
            bpws = self.load_workspace_bpws(obs1, obs2, i1, i2)
            return bpws

    def get_workspace_binning_decoupling_matrix(self, obs1, obs2, i1, i2, save=True, icl=0):
        print("[get_workspace_binning_decoupling_matrix] Warning: this does not account for f_ell")
        wsp = self.get_workspace(obs1, obs2, i1, i2)

        # nell = self.b.lmax+1
        # coupling = wsp.get_coupling_matrix()[icl*nell:(icl+1)*nell,:][:,icl*nell:(icl+1)*nell]
        # WRONG, see NaMaster doc : The assumed ordering of power spectra is such that the l-th element of the i-th power spectrum be stored with index l * n_cls + i.
        n_cls =  max(1,obs1.spin)*max(1,obs2.spin)
        print("[get_workspace_binning_decoupling_matrix] Using {} spectra for obs1 and ob2 with spins {} and {}.".format(n_cls, obs1.spin, obs2.spin))
        coupling = wsp.get_coupling_matrix()[icl::n_cls,:][:,icl::n_cls]

        binning = np.zeros((self.b.get_n_bands(),self.b.lmax+1))
        binning_w = np.zeros((self.b.get_n_bands(),self.b.lmax+1))
        for i in range(self.b.get_n_bands()):
            binning[i,self.b.get_ell_list(i)] = 1.
            binning_w[i,self.b.get_ell_list(i)] = self.b.get_weight_list(i)

        binned_coupling = np.dot(binning_w, np.dot(coupling, binning.T))
        inv_binned_coupling = np.linalg.inv(binned_coupling)
        binning_decoupling_matrix = np.dot(inv_binned_coupling, binning_w)

        if save:
            filename = self.get_workspace_filename(obs1, obs2, i1, i2)
            np.save(filename[:-4]+'_binning_decoupling_matrix.npy', binning_decoupling_matrix)
        
        return binning_decoupling_matrix, inv_binned_coupling
        
    def save_cls(self, ext=None):
        make_directory(self.config.path_output+'/'+self.name)
        filename = os.path.join(self.config.path_output, self.name, 'cls_{}_nside{}.pickle'.format(self.config.name, self.nside))
        if ext is not None:
            filename = filename.replace('.pickle', '_'+str(ext)+'.pickle')
        pickle.dump(self.cls, open(filename, mode='wb'))

    def load_cls(self, print_summary=True, ext=None, keep_cls=True):
        filename = os.path.join(self.config.path_output, self.name, 'cls_{}_nside{}.pickle'.format(self.config.name, self.nside))
        if ext is not None:
            filename = filename.replace('.pickle', '_'+str(ext)+'.pickle')
        loaded_cls = pickle.load(open(filename, mode='rb'))

        if keep_cls:
            assert np.allclose(self.ell, loaded_cls['ell'])
            self.cls = loaded_cls

        if print_summary:
            print("Loaded Cl's info:")
            print('  - Multipole bins ({}) = '.format(len(loaded_cls['ell'])), loaded_cls['ell'])
            for obs_key in loaded_cls.keys():
                if obs_key != 'ell':
                    print("  - Observables", obs_key)
                    for bin_key in loaded_cls[obs_key].keys():
                        print("     - Redshift bins", bin_key)
                        for tr_key, cl in loaded_cls[obs_key][bin_key].items():
                            print("          -", tr_key, cl.shape)

        return loaded_cls

    #########################
    # Compute cls   

    def compute_cls(self, obs1, i1, obs2=None, i2=None, save_cls=None, wsp=None):
        if obs2 is None:
            # same_obs = True
            obs2 = obs1
        if i2 is None:
            i2=i1

        self.check_obs(obs1, obs2)

        field1 = obs1.get_field(i1)
        field2 = obs2.get_field(i2)

        if wsp is None:
            wsp = self.get_workspace(obs1, obs2, i1, i2)

        _cls = compute_master(field1, field2, wsp)

        if (i1, i2) not in self.cls[(obs1.obs_name, obs2.obs_name)].keys():
            self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)] = {}

        self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)]['data'] = _cls

        if save_cls or self.do_save_cls:
            self.save_cls()
        
        return self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)]

    def _add_to_random(self, obs1, obs2, i1, i2, clr_new):
        if 'random' in self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)].keys():
            clr_old = self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)]['random']
            if clr_old is not None:
                assert isinstance(clr_old, np.ndarray)
                assert clr_old.shape[1:] == clr_new.shape[1:]
                print("Adding to existing randoms.")
                self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)]['random'] = np.concatenate([clr_old, clr_new], axis=0)
            else:
                self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)]['random'] = clr_new
        else:
            self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)]['random'] = clr_new

    def compute_random_cls(self, obs1, i1, obs2, i2, random_obs1, random_obs2, nrandom=1, auto_cls=False, save_cls=None, wsp=None, add_to_random=True):
        assert random_obs1 or random_obs2
        assert nrandom>0
        if auto_cls:
            assert obs1==obs2 and i1==i2

        self.check_obs(obs1, obs2)
        if (i1, i2) not in self.cls[(obs1.obs_name, obs2.obs_name)].keys():
            self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)] = {}

        # This would use a looot of memory
        # # Observable 1
        # if random_obs1:
        #     fields1 = obs1.make_randomized_fields(self, i1, nrandom=nrandom)
        # else:
        #     fields1 = [obs1.get_field(self, i1)] * nrandom
        # # Observable 2
        # if random_obs2:
        #     fields2 = obs2.make_randomized_fields(self, i2, nrandom=nrandom)
        # else:
        #     fields2 = [obs2.get_field(self, i2)] * nrandom

        if wsp is None:
            wsp = self.get_workspace(obs1, obs2, i1, i2)

        _cls = []
        for _ in self.prog(nrandom, desc='Harmony.compute_random_cls [obs1:{}, obs2={}]'.format(obs1.obs_name, obs2.obs_name)):
            # Observable 1
            if random_obs1:
                field1 = obs1.make_randomized_fields(i1, nrandom=1)[0]
            else:
                field1 = obs1.get_field(i1)

            # Observable 2
            if auto_cls:
                field2 = field1
            else:
                if random_obs2:
                    field2 = obs2.make_randomized_fields(i2, nrandom=1)[0]
                else:
                    field2 = obs2.get_field(i2)

            _cls.append(compute_master(field1, field2, wsp))

        if add_to_random:
            self._add_to_random(obs1, obs2, i1, i2, np.array(_cls))

        if save_cls or self.do_save_cls:
            self.save_cls()
        
        return np.array(_cls)


    def compute_all_cls(self, obs1, obs2=None, save_cls=None, auto_only=False):
        pairs = self.get_pairs(obs1, obs2=obs2, auto_only=auto_only)
        if obs2 is None:
            obs2 = obs1

        for i1,i2 in self.prog(pairs, desc='Harmony.compute_all_cls [obs1:{}, obs2={}]'.format(obs1.obs_name, obs2.obs_name)):
            self.compute_cls(obs1=obs1, i1=i1, obs2=obs2, i2=i2, save_cls=save_cls)

    def compute_random_all_cls(self, obs1, obs2, random_obs1, random_obs2, nrandom=1, share_randoms=True, auto_cls=False, save_cls=None, auto_only=False):
        assert random_obs1 or random_obs2
        assert nrandom>0
        if auto_cls:
            assert obs1==obs2

        self.check_obs(obs1, obs2)
        
        pairs = self.get_pairs(obs1, obs2=obs2, auto_only=auto_only)
        pairs_i1 = [x[0] for x in pairs]
        pairs_i2 = [x[1] for x in pairs]

        _cls = {}
        for x in pairs:
            _cls[x] = []
            if x not in self.cls[(obs1.obs_name, obs2.obs_name)].keys():
                self.cls[(obs1.obs_name, obs2.obs_name)][x] = {}

        if share_randoms:
            for _ in self.prog(nrandom, desc='Harmony.compute_random_all_cls [obs1:{}, obs2={}]'.format(obs1.obs_name, obs2.obs_name)):
                # Observable 1
                if random_obs1:
                    fields1 = {i1:obs1.make_randomized_fields(i1, nrandom=1)[0] for i1 in pairs_i1}
                else:
                    fields1 = {i1:obs1.get_field(i1) for i1 in pairs_i1}

                # Observable 2
                if auto_cls:
                    fields2 = fields1
                else:
                    if random_obs2:
                        fields2 = {i2:obs2.make_randomized_fields(i2, nrandom=1)[0] for i2 in pairs_i2}
                    else:
                        fields2 = {i2:obs2.get_field(i2) for i2 in pairs_i2}

                for i1,i2 in pairs:
                    wsp = self.get_workspace(obs1, obs2, i1, i2)
                    _cls[i1,i2].append(compute_master(fields1[i1], fields2[i2], wsp))
            
            for i1,i2 in pairs:
                self._add_to_random(obs1, obs2, i1, i2, np.array(_cls[i1,i2]))
                # self.cls[(obs1.obs_name, obs2.obs_name)][x]['random'] = np.array(_cls[x])
            
            if save_cls or self.do_save_cls:
                self.save_cls()

        else:
            for i1,i2 in self.prog(pairs, desc='Harmony.compute_random_all_cls [obs1:{}, obs2={}]'.format(obs1.obs_name, obs2.obs_name)):
                self.compute_random_cls(obs1, i1, obs2, i2, random_obs1, random_obs2, nrandom, auto_cls=auto_cls, save_cls=save_cls)

    def compute_random_auto_cls(self, obs, nrandom=1, save_cls=None):
        self.compute_random_all_cls(obs, obs, True, True, nrandom=nrandom, share_randoms=False, auto_cls=True, save_cls=save_cls, auto_only=True)

    def compute_noise_auto_cls(self, obs, save_cls=None):
        obs.compute_noise_auto_cls(self, save_cls=save_cls)

    def compute_full_auto_cls(self, obs, nrandom=0, from_scratch=False, save_cls=None, save_maps=True, plot_maps=False):
        if from_scratch:
            self.cls = {}
            self.cls['ell'] = self.ell
            self.wsp = {}
            obs.make_maps(save=save_maps)
        else:
            try:
                obs.load_maps()
            except:
                obs.make_maps(save=save_maps)
        
        if plot_maps:
            obs.plot_maps()

        obs.make_masks_apo()
        obs.make_fields(self)

        if from_scratch: # otherwise wsp are loaded when computing cls.
            self.prepare_all_workspaces(obs)

        self.compute_all_cls(obs, save_cls=save_cls)
        if nrandom>0:
            self.compute_random_auto_cls(obs, nrandom=nrandom, save_cls=save_cls)


    #########################
    # Create data fits files   

    def bin_cl_theory(self, cl_in, obs1, obs2, i1, i2, fix_pixwin=False, bpws=None):
        if bpws is None:
            bpws = self.get_workspace_bpws(obs1, obs2, i1, i2)
        if isinstance(cl_in, dict):
            cl = cl_in[(obs1.obs_name, obs2.obs_name)][(i1,i2)]
        else:
            cl = cl_in
        if cl.ndim == 1:
            temp = np.zeros((bpws.shape[-2],bpws.shape[-1]))
            temp[0,:] = cl[:bpws.shape[-1]]
            cl = temp        

        # return wsp.decouple_cell(wsp.couple_cell(cl_in))
        if fix_pixwin:
            f_ell = hp.pixwin(self.nside)[:bpws.shape[-1]]**2
        else:
            f_ell = np.ones_like(cl[:,:bpws.shape[-1]])

        return np.tensordot(bpws, cl[:,:bpws.shape[-1]]*f_ell, axes=2)
    
    def get_twopoint_filename(self):
        return os.path.join(self.config.path_output, self.name, 'twopoint_cls_{}_nside{}.fits'.format(self.config.name, self.nside))

    def load_twopoint(self):
        filename = self.get_twopoint_filename()
        twopoint.TwoPointFile.from_fits(filename)

    def twopoint_build_spec(self, obs1, obs2, name, kernels, types, spin_idx=0):
        """
        eg name='shear_cl', kernels=('nz_source','nz_source'), types=('GEF','GEF')
        """
        bins1 = []
        bins2 = []
        angbin = []
        value = []
        for (i1,i2), C_ell in self.cls[(obs1.obs_name, obs2.obs_name)].keys():
            cl_temp = C_ell['data'][spin_idx]
            if 'random' in C_ell.keys():
                cl_temp -= np.mean(C_ell['data'][:,spin_idx], axis=0)
            n = len(C_ell)
            bins1 += [i1+1]*n
            bins2 += [i2+1]*n
            angbin += list(np.arange(n))
            value += list(cl_temp)

        spec = twopoint.SpectrumMeasurement(name=name,bins=(np.array(bins1),np.array(bins2)),
                types=types, kernels=kernels, windows='SAMPLE',
                angular_bin=np.array(angbin), value=np.array(value),
                angle_min=np.array([self.b.get_ell_list(_i)[0] for _i in range(self.b.get_n_bands())]),
                angle_max=np.array([self.b.get_ell_list(_i)[-1] for _i in range(self.b.get_n_bands())])
                )
            
        spec.reorder_canonical()

        make_directory(self.config.path_output+'/'+self.name)
        filename = os.path.join(self.config.path_output, self.name, 'twopointspec_{}_nside{}_{}_{}_{}_{}.pickle'.format(self.config.name, self.nside, obs1.obs_name, obs2.obs_name, i1, i2))
        spec.to_fits(filename)

        return spec

    def get_cl(self, obs1, obs2, i1, i2, debias=True, which=None):
        if isinstance(obs1, Observable):
            obs1_name = obs1.obs_name
            obs2_name = obs2.obs_name
        else:
            obs1_name = obs1
            obs2_name = obs2

        if which is not None:
            assert which in ['data', 'random']
            return np.copy(self.cls[(obs1_name, obs2_name)][(i1,i2)][which])
        else:
            out = np.copy(self.cls[(obs1_name, obs2_name)][(i1,i2)]['data'])
            if obs1==obs2 and i1==i2 and debias:
                out -= np.mean(self.cls[(obs1_name, obs2_name)][(i1,i2)]['random'], axis=0)            
            return out

    def _compute_gaussian_covariance_block(self, obsa1, obsa2, obsb1, obsb2, a1, a2, b1, b2, C_ell, add_noise_to_C_ell=True, improved_NKA=False):

        def spin_help(obs1, obs2):
            return max(1,obs1.spin)*max(1,obs2.spin)
        n_ell = self.b.get_n_bands()

        def get_C_ell(obs1, obs2, i1, i2, add_noise=add_noise_to_C_ell, improved_NKA=improved_NKA):
            if (obs1, obs2, i1, i2) in self.C_ell_cov.keys():
                return self.C_ell_cov[obs1, obs2, i1, i2]
            else:
                res = np.copy(C_ell[(obs1.obs_name,obs2.obs_name)][(i1,i2)])
                if add_noise and obs1==obs2 and i1==i2:
                    NL = self.b.unbin_cell(self.cls[(obs1.obs_name,obs2.obs_name)][(i1,i2)]['random'].mean(axis=0))
                    lowest_ell = self.b.get_ell_list(0)[0]
                    for i in range(NL.shape[0]):
                        NL[i,:lowest_ell] = NL[i,lowest_ell]
                    res += NL

                if improved_NKA:
                    print("[_compute_gaussian_covariance_block] using improved NKA on {},{} [{},{}]".format(obs1.obs_name, obs2.obs_name, i1, i2))
                    try:
                        wsp = self.get_workspace(obs1, obs2, i1, i2)
                    except KeyError:
                        try:
                            wsp = self.load_workspace(obs1, obs2, i1, i2)
                        except RuntimeError:
                            print("[_compute_gaussian_covariance_block] preparing workspace for this pair...")
                            wsp = self.prepare_workspace(obs1, obs2, i1, i2)
                    coupling = wsp.get_coupling_matrix()
                    re_coupling = np.reshape(coupling, newshape=(spin_help(obs1, obs2), self.b.lmax+1, spin_help(obs1, obs2), self.b.lmax+1), order='F')
                    cl_tilde = np.tensordot(re_coupling, res) / np.mean(obs1.masks_apo[i1] * obs2.masks_apo[i2])
                    assert cl_tilde.shape == res.shape
                    res = cl_tilde

                self.C_ell_cov[obs1, obs2, i1, i2] = np.copy(res)

                return res

        # cw=nmt.NmtCovarianceWorkspace()

        suffix = ''
        suffix += obsa1.obs_name+str(a1)+'_'
        suffix += obsa2.obs_name+str(a2)+'_'
        suffix += obsb1.obs_name+str(b1)+'_'
        suffix += obsb2.obs_name+str(b2)+'.pickle'
        filename = os.path.join(self.config.path_output, self.name, 'cw_{}_nside{}_.pickle'.format(self.config.name, self.nside) + suffix)
        if os.path.exists(filename):
            self.cw.read_from(filename)
        else:
            self.cw.compute_coupling_coefficients(obsa1.get_field(a1),
                                             obsa2.get_field(a2),
                                             obsb1.get_field(b1),
                                             obsb2.get_field(b2)) #<- This is the time-consuming operation

            self.cw.write_to(filename)

        cov_temp = nmt.gaussian_covariance(self.cw,
                        obsa1.spin, obsa2.spin, obsb1.spin, obsb2.spin,
                        get_C_ell(obsa1,obsb1,a1,b1),
                        get_C_ell(obsa1,obsb2,a1,b2),
                        get_C_ell(obsa2,obsb1,a2,b1),
                        get_C_ell(obsa2,obsb2,a2,b2),
                        self.get_workspace(obsa1, obsa2, a1, a2),
                        self.get_workspace(obsb1, obsb2, b1, b2))
        
        return cov_temp.reshape([n_ell, spin_help(obsa1, obsa2), n_ell, spin_help(obsb1, obsb2)])#[:, 0, :, 0]
    
    @staticmethod
    def bin_helper(obs, i):
        return obs.zbins.index(i)
    
    def _twopoint_add_bpws(self, obs_pairs, names, fix_pixwin, spin_idx=0):
        filename = self.get_twopoint_filename()
        hdus = fits.open(filename)
        
        # Add bpws
        for name, (obs1,obs2) in zip(names, obs_pairs):
            for (i1,i2) in self.cls[(obs1.obs_name, obs2.obs_name)].keys():
                f_ell = np.ones(self.b.lmax+1)
                if fix_pixwin:
                    f_ell = 1./hp.pixwin(self.nside)[:self.b.lmax+1]**2
                bwps = self.load_workspace_bpws(obs1, obs2, i1, i2)[spin_idx,:,spin_idx,:] / f_ell
                hdus.append(fits.ImageHDU(bwps, name='bpws_'+name+'_{}_{}'.format(self.bin_helper(obs1,i1)+1,self.bin_helper(obs2,i2)+1)))
        
        # Add true ell limits
        ell_lims_min = np.array([self.b.get_ell_list(_i)[0] for _i in range(self.b.get_n_bands())])
        ell_lims_max = np.array([self.b.get_ell_list(_i)[-1] for _i in range(self.b.get_n_bands())])
        angle_edges = [ell_lims_min[0]-0.5]+list(ell_lims_max+0.5)

        ell_lims_min = fits.Column(array=ell_lims_min, format='I', name='ell_lims_min')
        ell_lims_max = fits.Column(array=ell_lims_max, format='I', name='ell_lims_max')
        hdus.append(fits.BinTableHDU.from_columns([ell_lims_min, ell_lims_max], name='ell_lims'))

        # Add ell ranges used in binned spectrum within the cosmosis 2pt likelihood
        for name in names:
            _ntiles = int(len(self.cls[(obs1.obs_name, obs2.obs_name)].keys()))
            angle_min = fits.Column(name='ANGLEMIN', format='D', array=np.tile(angle_edges[:-1], _ntiles))
            angle_max = fits.Column(name='ANGLEMAX', format='D', array=np.tile(angle_edges[1:], _ntiles))
            hdus[name] = fits.BinTableHDU.from_columns(header=hdus[name].header, columns=hdus[name].columns+fits.ColDefs([angle_min, angle_max]))

        hdus.writeto(filename, overwrite=True)

    def build_twopoint(self, obs_pairs, names, kernels, C_ell, fix_pixwin, spin_idx=0, use_C_ell_as_data=False, add_noise_to_C_ell=True, overwrite=False, clobber=False, improved_NKA=False):
        """
        names is a list of names for each obs pair
        kernels is the list of redshift distribution to be used
        C_ell is a dictionary of theoretical spectra with the same organization as the measured cls [(obs1.obs_name, obs2.obs_name)][(i1,i2)][spin_size,n_ell]
        """
        n_ell = self.b.get_n_bands()

        builder = twopoint.SpectrumCovarianceBuilder()

        # bin_pairs = np.array(self.cls[(obs1.obs_name, obs2.obs_name)].keys())
        # # # sort by bin1, then bin2
        # bin_pairs.view('i8,i8').sort(order=['f0','f1'], axis=0)
        # length = len(bin_pairs) * n

        # Loop over obs pairs, then their redshift bin pais present in cls
        total_size = 0
        ell_b = self.b.get_effective_ells()
        for (obs1, obs2) in self.prog(obs_pairs, desc='Harmony.build_twopoint [data vector]'):
            for (i1,i2) in self.prog(self.cls[(obs1.obs_name, obs2.obs_name)].keys(), desc='[{},{}]'.format(obs1.obs_name, obs2.obs_name)):
                if use_C_ell_as_data:
                    values = self.bin_cl_theory(C_ell, obs1, obs2, i1, i2, fix_pixwin=fix_pixwin)
                else:
                    values = self.get_cl(obs1, obs2, i1, i2, debias=True)
                for i in range(n_ell):
                    builder.add_data_point(kernel1=obs1.kernel,
                                   kernel2=obs2.kernel,
                                   type1=obs1.type,
                                   type2=obs2.type,
                                   bin1=self.bin_helper(obs1,i1)+1,
                                   bin2=self.bin_helper(obs2,i2)+1,
                                   ang=ell_b[i],
                                   angbin=i,
                                   value=values[spin_idx,i])
                    total_size += 1
        covmat = np.zeros((total_size, total_size))

        # Double loop in same order as above
        self.cw = nmt.NmtCovarianceWorkspace()
        self.C_ell_cov = {}
        idxa = 0
        for (obsa1, obsa2) in self.prog(obs_pairs, desc='Harmony.build_twopoint [covariance]'):
            for (a1,a2) in self.prog(self.cls[(obsa1.obs_name, obsa2.obs_name)].keys(), desc='[{},{}]'.format(obsa1.obs_name, obsa2.obs_name)):
                idxb = 0
                for (obsb1, obsb2) in obs_pairs:
                    for (b1,b2) in self.cls[(obsb1.obs_name, obsb2.obs_name)].keys():
                        covmat[idxa:idxa+n_ell,idxb:idxb+n_ell] = self._compute_gaussian_covariance_block(obsa1, obsa2, obsb1, obsb2, a1, a2, b1, b2, C_ell, add_noise_to_C_ell=add_noise_to_C_ell, improved_NKA=improved_NKA)[:,spin_idx,:,spin_idx]
                        idxb += n_ell
                idxa += n_ell

        builder.set_names(names)
        spectra, covmat_info = builder.generate(covmat, None)
        
        twopointfile = twopoint.TwoPointFile(spectra, kernels, windows='SAMPLE', covmat_info=covmat_info)
        # twopointfile.reorder_canonical()
        
        filename = self.get_twopoint_filename()
        twopointfile.to_fits(filename, overwrite, clobber)

        self._twopoint_add_bpws(obs_pairs, [names[(obs1.kernel, obs2.kernel, obs1.type, obs2.type)] for obs1,obs2 in obs_pairs], fix_pixwin, spin_idx=spin_idx)


