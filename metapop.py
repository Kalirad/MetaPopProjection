"""
A matrix stage-structured model to simulate the population dynamics of P. pacificus
over a metapopulation.
"""

__author__ = 'Ata Kalirad'

__version__ = '1.0'


import scipy as scp
import pandas as pd
import copy as cp
from tqdm import tqdm 
from itertools import cycle
import numpy as np
from scipy.signal import correlate2d

# global
fec_pars = {'A': {'OP50': [22.65, 68.45, 57.05, 33.4,  4.97], 'Novo': [11.66, 62.53, 47.13, 13.94,  0.72]},
            'C': {'OP50': [19.8 , 60.3 , 43.02, 19.9,  6.6], 'Novo': [16.88, 80.77, 77.7 , 16.28,  1.4]}}

mf_prob = {'A': {'OP50': 0.11, 'Novo': 0.83}, 'C': {'OP50': 1.0, 'Novo': 1.0}}

class Population(object):
    """Population object.
    """
    
    def __init__(self, n0, consumption=0.0, r0=1000, α=False):
        """Initialize the population object with an initial population vector.

        Args:
            n0 (numpy.ndarray): The initial composition of the population. 
            consumption (float, optional): Per capita consumption (excluding egg and dauer larvae). Defaults to 0.0.
            r0 (int, optional): Initial resource. Defaults to 1000.
        """
        self.pop = n0
        self.resource = r0
        self.α = α
        self.consumption = consumption
        self.diet_stat_pre = None
    
    @property
    def comb_f_mat(self):
        """Generate the fecundity block matrix.

        Returns:
            numpy.matrix: The block matrix containing the fecundity matrices. 
        """
        if self._diet != self.diet_stat_pre or not hasattr(self, 'f_mat_A'):
            self.f_mat_A = self.gen_fec_matrix('A', self.diet, 0.0415)
            self.f_mat_C = self.gen_fec_matrix('C', self.diet, 0.0415)
        return np.block([[self.f_mat_A, np.zeros((10,10))],
                        [np.zeros((10,10)), self.f_mat_C]])
        
    @property
    def diet(self):
        return self._diet
    
    @diet.setter
    def diet(self, d):
        if self.diet_stat_pre is None:
            self.diet_stat_pre = d
        else:
            self.diet_stat_pre = self._diet
        self._diet = d
        
    @property
    def u_mat(self):
        u_mat_A = self.generate_matrix(self.resource, self.diet, 'A')
        u_mat_C = self.generate_matrix(self.resource, self.diet, 'C')
        return (u_mat_A, u_mat_C)
    
    @property
    def comb_u_mat(self):
        """Generate the transition block matrix.

        Returns:
            numpy.matrix
        """
        u_A, u_C = self.u_mat
        comb_u_mat = np.block([[u_A, np.zeros((10,10))],
                               [np.zeros((10,10)), u_C]])
        return comb_u_mat
            
    @property
    def fund_mat(self):
        """Generate the fundamental matrices for the two strains in the population.

        Returns:
            tuple: strain A and strain c fundamental matrices 
        """
        u_A, u_C = self.u_mat
        return (np.linalg.inv(np.identity(10) - u_A), np.linalg.inv(np.identity(10) - u_C))
    
    @property
    def growth_rate(self):
        """Calculate the per generation growth rate (R0).

        Returns:
            tuple: leading Eigenvalues of A and strain c
        """
        fund_m = self.fund_mat
        return (np.linalg.eig(self.f_mat_A*fund_m[0])[0][0], np.linalg.eig(self.f_mat_C*fund_m[0])[0][0])
        
    @staticmethod
    def generate_matrix(r, diet, strain, r_min=100):
        """Generate the transition matrix.

        Args:
            r (float): Current resource level
            diet (str): Current diet
            strain (str): strain of interest
            r_min (int, optional): The minimum level of resource before switching to low resource condition. Defaults to 100.

        Returns:
            numpy.matrix
        """
        U = np.identity(10) * 0
        γ_JE  = lambda r: 0.0415 if r > r_min else 0.0
        γ_DJ = lambda r: 0.0415 if r < r_min else 0.0
        γ_YD = lambda r: 0.0415 if r > r_min else 0.0

        γ_YJ = lambda r: 0.0415 if r > r_min else 0.0
        if diet=='Novo':
            if strain == 'C':
                γ_RAY = 0.14
            else:
                γ_RAY = 0.12
        else:
            γ_RAY = 0.1
        γ_RR = 0.0415
        γ_OR = 0.0415
        δ_E = lambda r: 1 if r > r_min else 0.9
        δ_J = lambda r: 1 if r > r_min else 0.9
        δ_Y = lambda r: 1 if r > r_min else 0.9
        δ_RA = lambda r: 1 if r > r_min else 0.9
        δ_OA = lambda r: 0.995 if r > r_min else 0.9
        δ_D = 1
        U[0][0] = δ_E(r) * (1 - γ_JE(r))
        # juvenile ra(r)tes
        U[1][0] = δ_E(r) * γ_JE(r)
        U[1][1] = δ_J(r) * (1 - γ_YJ(r)) * (1 - γ_DJ(r))
        # dauer rates
        U[2][1] = δ_J(r) * γ_DJ(r)
        U[2][2] = δ_D * (1 - γ_YD(r))
        # young adults
        U[3][1] = δ_J(r) * γ_YJ(r)
        U[3][2] = δ_D * γ_YD(r)
        U[3][3] = δ_Y(r) * (1 - γ_RAY)
        # reproducing adults
        U[4][3] = δ_Y(r) * γ_RAY
        U[4][4] = δ_RA(r) * (1 - γ_RR)
        U[5][4] = δ_RA(r) * γ_RR
        U[5][5] = δ_RA(r) * (1 - γ_RR)
        U[6][5] = δ_RA(r) * γ_RR
        U[6][6] = δ_RA(r) * (1 - γ_RR)
        U[7][6] = δ_RA(r) * γ_RR
        U[7][7] = δ_RA(r) * (1 - γ_RR)
        U[8][7] = δ_RA(r) * γ_RR
        U[8][8] = δ_RA(r) * (1 - γ_OR)
        # old adults
        U[9][8] = δ_RA(r) * γ_OR
        U[9][9] = δ_OA(r)

        return np.matrix(U)
    
    @staticmethod
    def gen_fec_matrix(strain, food_type, rate):
        """Generate fecundity matrix.

        Args:
            strain (str): A (RSC017) or C (RS5405)
            food_type (str): The bacterial diet (Novo or OP50)

        Returns:
            numpy.matrix: fecundity values for each breeding developmental stage.
        """
        F = np.identity(10) * 0
        count = 0
        for i in np.arange(4, 9, 1):
            F[0][i] = rate*fec_pars[strain][food_type][count]
            count += 1
        return np.matrix(F)
    
    def predation(self):
        """Calculate the effect of predation on abundance of juveniles and dauer larvae. 
        """
        # C killing A
        juvenile_a = self.pop[1][0] 
        dauer_a = self.pop[2][0]
        predators_c = np.sum(self.pop[13:20])
        dauer_a -= self.α * mf_prob['C'][self.diet] * dauer_a * predators_c
        juvenile_a -= self.α * mf_prob['C'][self.diet] * juvenile_a * predators_c
        if dauer_a < 0:
            dauer_a = 0
        if juvenile_a < 0:
            juvenile_a = 0
        # A killing C
        juvenile_c = self.pop[11][0]
        dauer_c = self.pop[12][0]
        predators_a = np.sum(self.pop[3:10])
        dauer_c -= self.α * mf_prob['A'][self.diet] * dauer_c * predators_a
        juvenile_c -= self.α * mf_prob['A'][self.diet] * juvenile_c * predators_a
        if dauer_c < 0:
            dauer_c = 0
        if juvenile_c < 0:
            juvenile_c = 0
        self.pop[1][0]  = juvenile_a 
        self.pop[2][0]  = dauer_a
        self.pop[11][0] = juvenile_c
        self.pop[12][0] = dauer_c
    
    def take_a_step(self):
        """Calculate the population composition at t+1.
        """
        self.pop = np.array(np.matmul(self.comb_u_mat + self.comb_f_mat, self.pop))
        if self.α > 0:
            self.predation()
        if self.consumption > 0.0:
            total_conusmers = self.pop[1][0] + np.sum(self.pop[3:10]) + self.pop[11][0] + np.sum(self.pop[13:20])
            self.resource -= self.consumption * total_conusmers
            if self.resource < 0:
                self.resource = 0
            
            
class MetaPopulation(object):
    
    kernel = np.array([[0, 1, 0],
                       [1,-4, 1],
                       [0, 1, 0]])
    
    def __init__(self, dim, pred_rate, c, r_rate=0, diff_boundary='symm'):
        """Initialize the MetaPopulation object.

        Args:
            dim (int): the dimension of the lattice.
            pred_rate (float): the predation rate.
            c (float): the consumption rate.
            r_rate (int, optional): Dispersion rate. Defaults to 0.
            diff_boundary (str, optional): The diffusion pattern (for Periodic boundary condition, set to 'wrap'). Defaults to 'symm'.
        """
        self.dim = dim
        self.pred_rate = pred_rate
        self.cons_rate = c
        self.r_rate = r_rate
        self.diff_boundary = diff_boundary
        self.index = [(i,j) for i in range(dim) for j in range(dim)]
        self.fill_pop()
        
    def set_diet_comp(self, style):
        """Distribute the resources across according to the style ('OP50', 'Novo', 'quad_1', 'quad_2', 'rand').
        """
        if style == 'OP50':
            diet_comp = {i:'OP50' for i in self.index}
        if style == 'Novo':
            diet_comp = {i:'Novo' for i in self.index}
        if style == 'quad_1':
            ind = [(i,j) for i in range(self.dim) if i < self.dim/2 for j in range(self.dim//2)] + [(i,j) for i in range(self.dim) if i >= self.dim/2 for j in np.arange(self.dim//2 , self.dim)]
            for i in self.index:
                diet_comp = {i:'OP50' if i in ind else 'Novo' for i in self.index}
        if style == 'quad_2':
            ind = [(i,j) for i in range(self.dim) if i < self.dim/2 for j in range(self.dim//2)] + [(i,j) for i in range(self.dim) if i >= self.dim/2 for j in np.arange(self.dim//2 , self.dim)]
            for i in self.index:
                diet_comp = {i:'Novo' if i in ind else 'OP50' for i in self.index}   
        if style == 'rand':
            ind = np.random.randint(0, len(self.index), size=len(self.index)//2)
            Novo_ind = [self.index[i] for i in ind]
            diet_comp = {i:'Novo' if i in Novo_ind else 'OP50' for i in self.index}
        for i in self.index:
            self.metapop[i].diet = diet_comp[i]
        
    def fill_pop(self):
        """Fill the MetaPopulation with empty population objects.
        """
        n0_empty = np.array([[0],  [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],  [0], [0], [0], [0], [0], [0], [0], [0], [0]])
        self.metapop = {i:Population(n0_empty, consumption=self.cons_rate, α=self.pred_rate) for i in self.index}
        
    def add_pop(self, loc, strain):
        """Add Population to specific location on the lattice. 

        Args:
            loc (tuple): The location on the lattice.
            strain (str): 'A' or 'C'.
        """
        if strain == 'A':
            n0_A = np.array([[0], [50], [0], [0], [0], [0], [0], [0], [0], [0], [0],  [0], [0], [0], [0], [0], [0], [0], [0], [0]])
            self.metapop[loc] = Population(n0_A, consumption=self.cons_rate, α=self.pred_rate)
        else:
            n0_C = np.array([[0],  [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [50], [0], [0], [0], [0], [0], [0], [0], [0]])
            self.metapop[loc] = Population(n0_C, consumption=self.cons_rate, α=self.pred_rate)
            
    def reset_food(self, r0=1000):
        """Set the food in all subpopulations to r0.
        """
        for i in self.index:
            self.metapop[i].resource = r0

    @property
    def daur_dist(self):
        """The number of dauer larvae on the MetaPopulation.
        """
        dist_A = np.zeros((self.dim, self.dim))
        dist_C = np.zeros((self.dim, self.dim))
        for i in self.index:
            dist_A[i[0]][i[1]] = self.metapop[i].pop[2][0]
            dist_C[i[0]][i[1]] = self.metapop[i].pop[12][0]
        return dist_A, dist_C
    
    @property
    def r_dist(self):
        """Get the number of resource in each subpopulation.
        """
        r_patt = np.zeros((self.dim, self.dim))
        for i in self.index:
            r_patt[i[0]][i[1]] = self.metapop[i].resource
        return r_patt
    
    @property
    def ra_dist(self):
        """Get the number of reproducing adults.
        """
        dist_A = np.zeros((self.dim, self.dim))
        dist_C = np.zeros((self.dim, self.dim))
        for i in self.index:
            dist_A[i[0]][i[1]] = np.sum(self.metapop[i].pop[4:9])
            dist_C[i[0]][i[1]] = np.sum(self.metapop[i].pop[14:19])
        return dist_A, dist_C
    
    def diffuse_dauer(self):
        """Diffuse dauer larvae across the MetPopulation based on the kernel.
        """
        a, c = self.daur_dist
        total = np.sum(a) + np.sum(c)
        if total >= 0:
            total_a = np.sum(a)
            total_c = np.sum(c)
            if total_a > 0:
                f_a = np.divide(a, total)
                diff_state_a = self.r_rate * correlate2d(f_a, self.kernel, mode='same', boundary=self.diff_boundary)
                f_a += diff_state_a
                f_a = total_a * f_a
                for i in self.index:
                    self.metapop[i].pop[2][0]  = f_a[i[0]][i[1]]
            if total_c > 0:
                f_c = np.divide(c, total)
                diff_state_c = self.r_rate * correlate2d(f_c, self.kernel, mode='same', boundary=self.diff_boundary)
                f_c += diff_state_c
                f_c = total_c * f_c
                for i in self.index:
                    self.metapop[i].pop[12][0] = f_c[i[0]][i[1]]

    def simulate_pops_one_step(self):
        """Calculate the composition of the MetaPopulation at t+1.
        """
        for i in self.index:
            self.metapop[i].take_a_step()

