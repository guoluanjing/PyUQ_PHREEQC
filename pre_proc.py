from scipy.stats.distributions import norm
import settings
import numpy as np

settings.lhc_sampling()


# # sp = [HgOH,Hg(OH)2,Hg(OH)3,HgCl+,HgCl2,HgCl3-,HgCl4-2,HgOHCl,Hg(SH)2,HgS2H-,HgS2-2,(HgSH)+]
# means = [-3.4, -5.98, -21.1, 7.31, 14.0, 14.925, 15.535, 4.27, 39.1, 32.5, 23.2, 20.01]
# stdvs = [0.08, 0.06, 0.3, 0.04, 0.07, 0.09, 0.12, 0.35, 0.806, 0.806, 0.806, 0.56]

# sp = [HgOH,Hg(OH)2,Hg(OH)3,HgCl+,HgCl2,HgCl3-,HgCl4-2,HgOHCl,Hg(SH)2,HgS2H-,HgS2-2,(HgSH)+,HgS]
# means = [-3.4, -5.98, -21.1, 7.31, 14.0, 14.925, 15.535, 4.27, 39.1, 32.5, 23.2, 20.01, 28.6]
means = [-3.4, -5.98, -21.1, 7.31, 14.0, 14.925, 15.535, 4.27, 39.1, 32.5, 23.2, 20.01, 16.4]
stdvs = [0.08, 0.06, 0.3, 0.04, 0.07, 0.09, 0.12, 0.35, 0.806, 0.806, 0.806, 0.56, 0.806]

logk = settings.logk
for i in xrange(len(logk[0])):
    logk[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(logk[:, i])

np.save(settings.dir_base+'logk_samples.npy', logk)