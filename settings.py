import os
from pyDOE import *

def init():
    home = os.environ['HOME']

    global dir_base
    # dir_base= home + "/Documents/python/var_ph_dist_search/13logk_hgs28.6/"
    # dir_base= home + "/Documents/python/var_ph_dist_search/13logk_mineral/"
    # dir_base= home + "/Documents/python/var_ph_dist_search/14logk_mineral/"
    # dir_base= home + "/Documents/python/new_cases/var_ph/14logk_mineral/"
    dir_base= home + "/Documents/python/new_cases/var_ph/13logk_hgs16.4/"
    # dir_base= home + "/Documents/python/new_cases/var_ph/13logk_mineral/"
    # dir_base= home + "/Documents/python/new_cases/var_ph/13logk_min_hghs2/"
    # dir_base= home + "/Documents/python/new_cases/var_ph/w_mineral/1logk_hgs/"
    # dir_base= home + "/Documents/python/new_cases/var_ph/w_mineral/1logk_mineral/"
    # dir_base= home + "/Documents/python/new_cases/var_ph/w_mineral/1logk_hghs2-/"
    # dir_base= home + "/Documents/python/new_cases/var_ph/w_mineral/1logk_hgs2-2/"
    # dir_base= home + "/Documents/python/new_cases/var_ph/w_mineral/1logk_hghs2/"
    # dir_base= home + "/Documents/python/new_cases/var_ph/w_mineral/2logk_min_hghs2/"
    # dir_base= home + "/Documents/python/new_cases/var_ph/w_mineral/2logk_hghs2_hghs2-/"
    # dir_base= home + "/Documents/python/new_cases/var_ph/w_mineral/2logk_hghs2_hgs2-2/"
    # dir_base= home + "/Documents/python/new_cases/var_ph/w_mineral/2logk_min_hghs2-/"
    # dir_base= home + "/Documents/python/new_cases/var_ph/w_mineral/2logk_min_hgs2-2/"
    # dir_base= home + "/Documents/python/new_cases/var_ph/w_o_mineral/1logk_hghs2/"
    # dir_base= home + "/Documents/python/new_cases/var_ph/w_o_mineral/1logk_hghs2-/"
    # dir_base= home + "/Documents/python/new_cases/var_ph/w_o_mineral/1logk_hgs2-2/"

    global logk_basedir
    logk_basedir= "/Users/qng/Documents/python/"

    global basecase_dir
    # basecase_dir = home + "/Documents/SBR_Mercury/phreeqc/basecases/hg_hs_eq_metacinnabar/"
    # basecase_dir = home + "/Documents/SBR_Mercury/phreeqc/basecases/hg-h2o-cl-hs/"
    # basecase_dir = home + "/Documents/SBR_Mercury/phreeqc/new_basecases/hg_hs_eq_metacinnabar/"
    basecase_dir = home + "/Documents/SBR_Mercury/phreeqc/new_basecases/hg-h2o-cl-hs_hgs16.4/"

    global var_ph_dirbase
    # var_ph_dirbase = home + "/Documents/SBR_Mercury/phreeqc/basecases/var_ph/fine_0.2inc_mineral/"
    # var_ph_dirbase = home + "/Documents/SBR_Mercury/phreeqc/basecases/var_ph/fine_0.2inc/"
    # var_ph_dirbase = home + "/Documents/SBR_Mercury/phreeqc/new_basecases/var_ph/fine_0.2inc_mineral/"
    var_ph_dirbase = home + "/Documents/SBR_Mercury/phreeqc/new_basecases/var_ph/fine_0.2inc_hgs16.4/"

    global new_dir
    new_dir = []

    global lb
    lb = 4.0
    # lb = 4.1

    global ub
    ub = 10.1
    # ub = 10.0

    global inc
    # inc = 0.1
    inc = 0.2
    # inc = 1.0

    global logk_samples
    logk_samples = np.load(logk_basedir+'new14logk_1000samples.npy')
    # logk_samples = np.load(logk_basedir+'14logk_1000samples.npy')
    # logk_samples = np.load(logk_basedir+'14logk_samples.npy')
    # logk_samples = np.load(logk_basedir+'13logk_samples.npy')
    # logk_samples = np.load(dir_base + 'logk_samples.npy')


def lhc_sampling():
    global dir_base
    dir_base= "/Users/qng/Documents/python/var_ph_dist_search/"

    global sample_size
    # sample_size = 6000
    sample_size = 6500

    global logk
    # logk = lhs(12, samples=sample_size)
    logk = lhs(13, samples=sample_size)

