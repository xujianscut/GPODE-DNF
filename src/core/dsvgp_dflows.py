from src.misc.param import Param
from src.misc import transforms
from src.core.kernels import RBF

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

jitter = 1e-5


def sample_normal(shape, seed=None):
    # sample from standard Normal with a given shape
    if seed is not None:
        rng = np.random.RandomState(seed)
        return torch.tensor(rng.normal(size=shape).astype(np.float32))
    else:
        return torch.tensor(np.random.normal(size=shape).astype(np.float32))


def sample_uniform(shape, seed=None):
    # random Uniform sample of a given shape
    if seed is not None:
        rng = np.random.RandomState(seed)
        return torch.tensor(rng.uniform(low=0.0, high=1.0, size=shape).astype(np.float32))
    else:
        return torch.tensor(np.random.uniform(low=0.0, high=1.0, size=shape).astype(np.float32))

    
class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.s = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Tanh(),)
        self.u = nn.Linear(1,dim)
     
        
        self.b = torch.zeros(1)

    def forward(self, x):
        
        f = x +self.u(self.s(x))

        
        psi = (1 - torch.tanh(self.s(x)) ** 2) * self.u.weight
        #det = torch.eye(500) + torch.matmul(psi, self.s[0].weight)
        #log_det = torch.log(torch.abs(torch.det(det)))
        det = 1 + torch.matmul( self.s[0].weight,psi)
        log_det = torch.log(torch.abs(det))

        return f, log_det

class PlanarFlowModel(nn.Module):
    def __init__(self, dim, num_flows):
        super().__init__()

        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(num_flows)])

    def forward(self, x):
        log_det = 0
        for flow in self.flows:
            x, ld = flow(x)
            log_det += ld
        return x, log_det

class PlanarFlow1(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.s = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Tanh(),nn.Linear(1,dim))
        
     
        
        

    def forward(self, x):
        
        f = x + self.s(x)

        
        

        return f
        
        
class PlanarFlowModel1(nn.Module):
    def __init__(self, dim, num_flows):
        super().__init__()

        self.flows = nn.ModuleList([PlanarFlow1(dim) for _ in range(num_flows)])

    def forward(self, x):
        
        for flow in self.flows:
            x = flow(x)
            
        return x

    
class RealNVP(nn.Module):
    def __init__(self, in_dim):
        super(RealNVP, self).__init__()
        self.mask = torch.arange(in_dim) % 2
        self.s = nn.Sequential(
            nn.Linear(in_dim // 2 , 64),
            nn.Tanh(),
            nn.Linear(64, in_dim // 2)
        )
        self.t = nn.Sequential(
            nn.Linear(in_dim // 2 , 40),
            nn.Tanh(),
            nn.Linear(40, in_dim // 2)
        )

    def forward(self, x):
        x1 = x[:, self.mask == 0]
        x2 = x[:, self.mask == 1]
       
        s = self.s(x1)
        t = self.t(x1)
        y2 = x2
        y1 = x1 * torch.exp(s) + t
        y = torch.cat([y1, y2], dim=1)
        return y

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.realnvp1 = RealNVP(500)
        self.realnvp2 = RealNVP(500)
        self.realnvp3 = RealNVP(500)
        self.realnvp4 = RealNVP(500)

    def forward(self, x):
        x = self.realnvp1(x)
        x = self.realnvp2(x)
        x = self.realnvp3(x)
        x = self.realnvp4(x)
        
        
        return x    


    

class DSVGP_Layer(torch.nn.Module):
    """
    A layer class implementing decoupled sampling of SVGP posterior
     @InProceedings{pmlr-v119-wilson20a,
                    title = {Efficiently sampling functions from {G}aussian process posteriors},
                    author = {Wilson, James and Borovitskiy, Viacheslav and Terenin, Alexander and Mostowsky, Peter and Deisenroth, Marc},
                    booktitle = {Proceedings of the 37th International Conference on Machine Learning},
                    pages = {10292--10302},
                    year = {2020},
                    editor = {Hal Daumé III and Aarti Singh},
                    volume = {119},
                    series = {Proceedings of Machine Learning Research},
                     publisher = {PMLR},
                     pdf = {http://proceedings.mlr.press/v119/wilson20a/wilson20a.pdf}
                     }
    """

    def __init__(self,D_in, D_out, M, S, q_diag=False, dimwise=True):
        """
        @param D_in: Number of input dimensions
        @param D_out: Number of output dimensions
        @param M: Number of inducing points
        @param S: Number of features to consider for Fourier feature maps
        @param q_diag: Diagonal approximation for inducing posterior
        @param dimwise: If True, different kernel parameters are given to output dimensions
        """
        super(DSVGP_Layer, self).__init__()

        self.kern = RBF(D_in, D_out, dimwise)
        self.q_diag = q_diag
        self.dimwise = dimwise

        self.D_out = D_out
        self.D_in = D_in
        self.M = M
        self.S = S
        
        self.net1=PlanarFlowModel1(5,3)
        self.net_inf=PlanarFlowModel(500,1)
        self.b=torch.nn.Parameter(torch.randn([100,5])*0.01)

        
        
        
        
        
        self.inducing_loc = Param(np.random.normal(size=(M, D_in)), name='Inducing locations')  # (M,D_in)
     
        
        
    def sample_inducing(self):
        """
        Generate a sample from the inducing posterior q(u) ~ N(m, S)
        @return: inducing sample (M,D) tensor
        """
        
   
   
        epsilon = sample_normal(shape=(self.M, self.D_out), seed=None)  # (M, D_out)
        epsilon = epsilon.view(-1,self.M*self.D_out)
        
        u_sample, detlog= self.net_inf(epsilon)
        u_sample = u_sample.reshape(self.M, self.D_out)+self.b
        return u_sample,detlog  # (M, D_out)
    

   


    def build_cache(self):
        """
        Builds a cache of computations that uniquely define a sample from posterior process
        1. Generate and fix parameters of Fourier features
        2. Generate and fix inducing posterior sample
        3. Intermediate computations based on the inducing sample for pathwise update
        """
        # generate parameters required for the Fourier feature maps
        self.rff_weights = sample_normal((self.S, self.D_out))  # (S,D_out)
        self.rff_omega = self.kern.sample_freq(self.S)  # (D_in,S) or (D_in,S,D_out)
        phase_shape = (1, self.S, self.D_out) if self.dimwise else (1, self.S)
        self.rff_phase = sample_uniform(phase_shape) * 2 * np.pi  # (S,D_out)

        # generate sample from the inducing posterior
        inducing_val,_ = self.sample_inducing()  # (M,D)

        # compute th term nu = k(Z,Z)^{-1}(u-f(Z)) in whitened form of inducing variables
        
        Ku = self.kern.K(self.inducing_loc())  # (M,M) or (D,M,M)
        
        
        
        
        
        Lu = torch.cholesky(Ku + torch.eye(self.M) * jitter)  # (M,M) or (D,M,M)
        u_prior = self.rff_forward(self.inducing_loc())  # (M,D)

        if not self.dimwise:
            nu = torch.triangular_solve(u_prior, Lu, upper=False)[0]  # (M,D)
            nu = torch.triangular_solve((inducing_val - nu),
                                        Lu.T, upper=True)[0]  # (M,D)
        else:
            nu = torch.triangular_solve(u_prior.T.unsqueeze(2), Lu, upper=False)[0]  # (D,M,1)
            
            #nu = torch.triangular_solve(u_prior, Lu, upper=False)[0]
            
            nu = torch.triangular_solve((inducing_val.T.unsqueeze(2) - nu),
                                        Lu.permute(0, 2, 1), upper=True)[0]  # (D,M,1)
           



            #nu = torch.triangular_solve((inducing_val - nu),
                                        #Lu.T, upper=True)[0]  # (M,D)
                
        self.nu = nu  # (D,M)
        

    def rff_forward(self, x):
        """
        Calculates samples from the GP prior with random Fourier Features
        @param x: input tensor (N,D)
        @return: function values (N,D_out)
        """
        # compute feature map
        xo = torch.einsum('nd,dfk->nfk' if self.dimwise else 'nd,df->nf', x, self.rff_omega)  # (N,S) or (N,S,D_in)
        phi_ = torch.cos(xo + self.rff_phase)  # (N,S) or (N,S,D_in)
        phi = phi_ * torch.sqrt(self.kern.variance / self.S)  # (N,S) or (N,S,D_in)

        # compute function values
        f = torch.einsum('nfk,fk->nk' if self.dimwise else 'nf,fd->nd', phi, self.rff_weights)  # (N,D_out)
        return f  # (N,D_out)
    
   
    
    ''' 
    
    def build_conditional(self, x, full_cov=False):
        """
        Calculates conditional distribution q(f(x)) = N(m(x), Sigma(x))
            where  m(x) = k(x,Z)k(Z,Z)^{-1}u, k(x,x)
                   Sigma(x) = k(x,Z)k(Z,Z)^{-1}(S-K(Z,Z))k(Z,Z)^{-1}k(Z,x))

        @param x: input tensor (N,D)
        @param full_cov: if True, returns full Sigma(x) else returns only diagonal
        @return: m(x), Sigma(x)
        """
        Ku = self.kern.K(self.inducing_loc())  # (M,M) or (D,M,M)
        Lu = torch.cholesky(Ku + torch.eye(self.M) * jitter)  # (M,M) or (D,M,M)
        Kuf = self.kern.K(self.inducing_loc(), x)  # (M,N) or (D,M,N)
        A = torch.triangular_solve(Kuf, Lu, upper=False)[0]  # (M,M)@(M,N) --> (M,N) or (D,M,M)@(D,M,N) --> (D,M,N)

        Us_sqrt = self.Us_sqrt(self.inducing_loc.optvar).T[:, :, None] if self.q_diag else self.Us_sqrt()  # (D,M,1) or (D,M,M)
        SK = (Us_sqrt @ Us_sqrt.permute(0, 2, 1)) - torch.eye(Ku.shape[1]).unsqueeze(0)  # (D,M,M)
        B = torch.einsum('dme, den->dmn' if self.dimwise else 'dmi, in->dmn', SK, A)  # (D,M,N)

        if full_cov:
            delta_cov = torch.einsum('dme, dmn->den' if self.dimwise else 'me, dmn->den', A, B)  # (D,M,N)
            Kff = self.kern.K(x) if self.dimwise else self.kern.K(x).unsqueeze(0)  # (1,N,N) or (D,N,N)
        else:
            delta_cov = ((A if self.dimwise else A.unsqueeze(0)) * B).sum(1)  # (D,N)
            if self.dimwise:
                Kff = torch.diagonal(self.kern.K(x), dim1=1, dim2=2)  # (N,)
            else:
                Kff = torch.diagonal(self.kern.K(x), dim1=0, dim2=1)  # (D,N)

        var = Kff + delta_cov
        mean = torch.einsum('dmn, md->nd' if self.dimwise else 'mn, md->nd', A, self.Um(self.inducing_loc.optvar))
        return mean, var.T  # (N,D) , (N,D) or (N,N,D)
    '''
    
    def forward(self, t, x):
        """
        Compute sample from the SVGP posterior using decoupeld sampling approach
        Involves two steps:
            1. Generate sample from the prior :: rff_forward
            2. Compute pathwise updates using samples from inducing posterior :: build_cache

        @param t: time value, usually None as we define time-invariant ODEs
        @param x: input tensor in (N,D)
        @return: f(x) where f is a sample from GP posterior
        """
        # generate a prior sample using rff
        f_prior = self.rff_forward(x)  # (N,D))
       

        # compute pathwise updates
        if not self.dimwise:
            Kuf = self.kern.K(self.inducing_loc(), x)  # (M,N)
            f_update = torch.einsum('md, mn -> nd', self.nu, Kuf)  # (N,D)
        else:
            Kuf = self.kern.K(self.inducing_loc(), x)  # (D,M,N)
            
            f_update = torch.einsum('dm, dmn -> nd', self.nu.squeeze(2), Kuf)  # (6, 5)


        dx1=f_prior + f_update
        dx = self.net1(dx1)
        

        return dx  # (N,D)

    def kl(self):
        """
        KL(q(u) || p(u)) with β-annealing to avoid posterior NF collapse.
        Engineering fixes vs original:
          - torch.stack (preserves autograd; original torch.tensor stripped grad)
          - torch.logdet via Cholesky (original used torch.det → wrong magnitude)
          - β-annealing: kl_beta ramps from 0→1 over kl_warmup steps so the
            posterior NF has time to settle into a useful regime before KL kicks in.
          - 4 MC samples (was 64) — sufficient since SGD already adds noise.
        """
        if not hasattr(self, "_kl_step"):
            self._kl_step = 0
            self.kl_warmup = 400
        self._kl_step += 1
        beta = min(1.0, self._kl_step / float(self.kl_warmup))

        Ku = self.kern.K(self.inducing_loc())
        Ku_j = Ku + torch.eye(self.M) * jitter
        Lu = torch.cholesky(Ku_j)
        if self.dimwise:
            logdet_K = 2.0 * torch.diagonal(Lu, dim1=-2, dim2=-1).log().sum()
        else:
            logdet_K = 2.0 * torch.diagonal(Lu).log().sum() * self.D_out

        kl_list = []
        for _ in range(4):
            u, log_detv = self.sample_inducing()
            log_det_flow = log_detv.sum()
            if self.dimwise:
                u_perm = u.T.unsqueeze(-1)
                alpha = torch.triangular_solve(u_perm, Lu, upper=False)[0]
                quad = (alpha ** 2).sum()
            else:
                alpha = torch.triangular_solve(u, Lu, upper=False)[0]
                quad = (alpha ** 2).sum()
            const = 0.5 * self.M * self.D_out
            kl_s = -log_det_flow + 0.5 * logdet_K + 0.5 * quad - const
            kl_list.append(kl_s)
        output = torch.stack(kl_list).mean() * beta
        
               
        
        
        
        
        
        
        return output
