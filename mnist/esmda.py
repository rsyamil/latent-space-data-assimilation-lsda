import numpy as np
import string
import numpy.matlib as npmatlib

class ESMDA:
    def __init__(self, ref, d_obs, proxy, zm_test, zd_test, name=[]):
        self.name = name
        
        self.ref = ref
        self.d_obs = d_obs
        self.proxy = proxy
        
        self.zm_test = zm_test
        self.zd_test = zd_test
        
        self.zd_pert = []
        self.R = []
        
        self.Ensemble = []
        self.Ensemble_d = []
        
    def computeR(self):
    
        nens, nd = self.zm_test.shape[0], self.d_obs.shape[0]
        obsmatrix = npmatlib.repmat(self.d_obs, 1, nens)
        
        zd_obs_ens = self.proxy.d2zd.predict(obsmatrix.T)
        
        robs = 0.10 * np.ones((self.d_obs.shape[0], 1)) 
        errmatrix = np.multiply(obsmatrix, npmatlib.repmat(robs, 1, nens))
        errmatrix = np.multiply(errmatrix, np.random.normal(0, 1, (nd, nens)))
        obsmatrix = obsmatrix + errmatrix
        
        self.zd_pert = self.proxy.d2zd.predict(obsmatrix.T)
        zd_errmatrix = self.zd_pert.T - zd_obs_ens.T
        
        self.R = np.diag(np.diag(np.divide(np.matmul(zd_errmatrix, zd_errmatrix.T), (nens - 1))))
        
    def assimilate(self, N_a, alpha):
        
        self.computeR()
        
        self.Ensemble = [None] * N_a
        self.Ensemble[0] = self.zm_test[:, :].T
        
        self.Ensemble_d = [None] * N_a
        self.Ensemble_d[0] = self.zd_test[:, :].T 
        
        for step in range(N_a - 1):
            if step != 0:
                self.Ensemble_d[step] = self.proxy.zm2zd.predict(self.Ensemble[step].T).T
             
            self.Ensemble[step + 1] = self.update(self.Ensemble[step], self.Ensemble_d[step], alpha[step], self.R, self.zd_pert.T)

        self.Ensemble_d[N_a - 1] = self.proxy.zm2zd.predict(self.Ensemble[N_a - 1].T).T   
        
        return self.Ensemble, self.Ensemble_d
        
    def update(self, propens, dprd, alpha, R, obsmatrix):
        '''
        Update ensemble of properties
        :seed:      random number seed
        :propens:   ensemble to update (i.e. vectorized)
        :dprd:      data sim. from propens
        :alpha:     magnitude of update
        :R:         to perturb data
        :obsmatrix: perturbed observed data        
        '''
        np.random.seed(np.random.randint(0, 1e3))
        nens = propens.shape[1]
        
        CD = np.diag(alpha*R)
        CDD = np.cov(dprd) 
        
        NA = np.multiply(np.ones(shape=(nens, nens)), (1.0/nens))
        DA = np.matmul(propens, np.eye(nens) - NA)
        DF = np.matmul(dprd, np.eye(nens) - NA)
        CMD = np.divide(np.matmul(DA, DF.T), (nens - 1))

        propens = propens + np.matmul(np.matmul(CMD, np.linalg.pinv(CDD + CD)), (obsmatrix - dprd))

        return propens
