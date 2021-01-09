import numpy as np
import string
import numpy.matlib as npmatlib

class ESMDA:
    def __init__(self, ref, d_obs, sim, maxs, m_test, d_test, name=[]):
        self.name = name
        
        self.ref = ref
        self.d_obs = d_obs
        
        self.sim = sim
        self.maxs = maxs
        
        self.m_test = m_test
        self.d_test = d_test
        
        self.d_pert = []
        self.R = []
        
        self.Ensemble = []
        self.Ensemble_d = []
        
    def computeR(self):
    
        nens, nd = self.m_test.shape[0], self.d_obs.shape[0]            #10,000 x 128
        obsmatrix = npmatlib.repmat(self.d_obs, 1, nens)                #128 x 10,000
        obsmatrix_ = obsmatrix                                          #128 x 10,000
        
        robs = 0.10 * np.ones((self.d_obs.shape[0], 1))                             #128 x 1
        errmatrix = np.multiply(obsmatrix, npmatlib.repmat(robs, 1, nens))          #128 x 10,000
        errmatrix = np.multiply(errmatrix, np.random.normal(0, 1, (nd, nens)))      #128 x 10,000
        obsmatrix = obsmatrix + errmatrix                                           #128 x 10,000
        
        self.d_pert = obsmatrix                               #128 x 10,000
        d_errmatrix = self.d_pert - obsmatrix_                #128 x 10,000
        
        self.R = np.diag(np.diag(np.divide(np.matmul(d_errmatrix, d_errmatrix.T), (nens - 1))))
        
    def simulator(self, ms):
    
        #discretize the images
        ms = np.where(ms<0.5, 0, 1)
        
        #ms : 10,000 x 784
        d_dim = self.sim.shape[-1]
        ds = np.zeros([ms.shape[0], d_dim])
        
        for i in range(ms.shape[0]):
            ds[i:i+1, :] = (ms[i:i+1, :])@self.sim
        ds = ds/self.maxs

        return ds
        
    def assimilate(self, N_a, alpha):
        
        self.computeR()
        
        self.Ensemble = [None] * N_a
        self.Ensemble[0] = self.m_test[:, :].T             #784 x 10,000  
        
        self.Ensemble_d = [None] * N_a
        self.Ensemble_d[0] = self.d_test[:, :].T           #128 x 10,000 
        
        for step in range(N_a - 1):
            if step != 0:
                self.Ensemble_d[step] = self.simulator(self.Ensemble[step].T).T
             
            self.Ensemble[step + 1] = self.update(self.Ensemble[step], self.Ensemble_d[step], alpha[step], self.R, self.d_pert)

        self.Ensemble_d[N_a - 1] = self.simulator(self.Ensemble[N_a - 1].T).T   
        
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
