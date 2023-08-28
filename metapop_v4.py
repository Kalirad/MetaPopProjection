"""
A matrix stage-structured model to simulate the population dynamics of two 
P. pacificus strains over a metapopulation. Strain A (RS017), is a plastic strain
with respect to its mouth-from morphology and strain C (RS5405) is a non-plastic
strain with respect to the same trait.
"""

__author__ = 'Ata Kalirad'

__version__ = '3.0'


import numpy as np
from scipy.signal import correlate2d

fp_precision = 6

class Population(object):
    """Population object.
    """
    
    def __init__(self, n0, fec_pars, mf_prob, consumption=0.002, r0=500, α=False):
        """Initialize the population object with an initial population vector.

        Args:
            n0 (numpy.ndarray): The initial composition of the population. 
            consumption (float, optional): Per capita consumption (excluding egg and dauer larvae). Defaults to 0.0.
            r0 (int, optional): Initial resource. Defaults to 1000.
        """
        self.pop = n0
        self.fec_pars = fec_pars
        self.mf_prob = mf_prob
        self.init_resource = r0
        self.resource = r0
        self.starv = False
        self.α = α
        self.consumption = consumption
        self.diet_stat_pre = None
        self.pred_data = []

    def replenish(self):
        self.resource =  self.init_resource
        self.starv = False
    
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
    def n_consumers(self):
        return self.pop[1][0] + np.sum(self.pop[3:10]) + self.pop[11][0] + np.sum(self.pop[13:20])

        
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
        n_conusmers = self.n_consumers
        u_mat_A = self.generate_matrix(self.diet, 'A', self.starv, n_conusmers)
        u_mat_C = self.generate_matrix(self.diet, 'C', self.starv, n_conusmers)
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
    def generate_matrix(diet, strain, starv, size, sigma=6e-5):
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
        γ_JE  = lambda starv: 0.0415 if not starv else 0.0
        γ_DJ = lambda starv: 0.0415 if starv else 0.0
        γ_YD = lambda starv: 0.0415 if not starv else 0.0
        γ_YJ = lambda starv: 0.0415 if not starv else 0.0
        if diet=='Novo':
            if strain == 'C':
                γ_RAY = 0.14
            else:
                γ_RAY = 0.12
        else:
            γ_RAY = 0.1
        γ_RR = 0.0415
        γ_OR = 0.0415
        # δ_E = lambda starv: 1 if not starv else 0.9
        # δ_J = lambda starv: 1 if not starv else 0.9
        # δ_Y = lambda starv: 1 if not starv else 0.9
        # δ_RA = lambda starv: 1 if not starv else 0.9
        # δ_OA = lambda starv: 0.995 if not starv else 0.9

        δ_E = lambda size: np.exp(-sigma*size)
        δ_J = lambda size: np.exp(-sigma*size)
        δ_Y = lambda size: np.exp(-sigma*size)
        δ_RA = lambda size: np.exp(-sigma*size)
        δ_OA = lambda size: np.exp(-sigma*size)
        δ_D = 1
        U[0][0] = δ_E(size) * (1 - γ_JE(starv))
        # juvenile ra(starv)tes
        U[1][0] = δ_E(size) * γ_JE(starv)
        U[1][1] = δ_J(size) * (1 - γ_YJ(starv)) * (1 - γ_DJ(starv))
        # dauer rates
        U[2][1] = δ_J(size) * γ_DJ(starv)
        U[2][2] = δ_D * (1 - γ_YD(starv))
        # young adults
        U[3][1] = δ_J(size) * γ_YJ(starv)
        U[3][2] = δ_D * γ_YD(starv)
        U[3][3] = δ_Y(size) * (1 - γ_RAY)
        # reproducing adults
        U[4][3] = δ_Y(size) * γ_RAY
        U[4][4] = δ_RA(size) * (1 - γ_RR)
        U[5][4] = δ_RA(size) * γ_RR
        U[5][5] = δ_RA(size) * (1 - γ_RR)
        U[6][5] = δ_RA(size) * γ_RR
        U[6][6] = δ_RA(size) * (1 - γ_RR)
        U[7][6] = δ_RA(size) * γ_RR
        U[7][7] = δ_RA(size) * (1 - γ_RR)
        U[8][7] = δ_RA(size) * γ_RR
        U[8][8] = δ_RA(size) * (1 - γ_OR)
        # old adults
        U[9][8] = δ_RA(size) * γ_OR
        U[9][9] = δ_OA(size)

        return np.matrix(U)
    
    def gen_fec_matrix(self, strain, food_type, rate):
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
            F[0][i] = rate*self.fec_pars[strain][food_type][count]
            count += 1
        return np.matrix(F)
    
    def predation(self):
        """Calculate the effect of predation on abundance of juveniles and dauer larvae. 
        """
        # C killing A
        juvenile_a = self.pop[1][0] 
        dauer_a = self.pop[2][0]
        predators_c = np.sum(self.pop[13:19]) * self.mf_prob['C'][self.diet]
        if self.α == 2:
            a = 0.2
            h = 0.15
            dauer_a_killed = np.divide(a* dauer_a, 1 + a*h* dauer_a) *  predators_c
            juvenile_a_killed = np.divide(a* juvenile_a, 1 + a*h* juvenile_a) *  predators_c
            dauer_a -= dauer_a_killed
            juvenile_a -= juvenile_a_killed
        else:
            dauer_a_killed = self.α * dauer_a * predators_c
            juvenile_a_killed = self.α * juvenile_a * predators_c
            dauer_a -= dauer_a_killed
            juvenile_a -= juvenile_a_killed
        if dauer_a < 0:
            dauer_a = 0
        if juvenile_a < 0:
            juvenile_a = 0
        # A killing C
        juvenile_c = self.pop[11][0]
        dauer_c = self.pop[12][0]
        predators_a = np.sum(self.pop[3:9]) * self.mf_prob['A'][self.diet]
        if self.α == 2:
            a = 0.2
            h = 0.15
            dauer_c_killed = np.divide(a* dauer_c, 1 + a*h* dauer_c) *  predators_a
            juvenile_c_killed = np.divide(a* juvenile_c, 1 + a*h* juvenile_c) *  predators_a
            dauer_c -= dauer_c_killed
            juvenile_c -= juvenile_c_killed
        else:
            dauer_c -= self.α *  dauer_c * predators_a
            juvenile_c -= self.α *  juvenile_c * predators_a
        if dauer_c < 0:
            dauer_c = 0
        if juvenile_c < 0:
            juvenile_c = 0
        self.pop[1][0]  = round(juvenile_a, fp_precision) 
        self.pop[2][0]  = round(dauer_a, fp_precision)
        self.pop[11][0] = round(juvenile_c, fp_precision)
        self.pop[12][0] = round(dauer_c, fp_precision)
        if self.α == 2:
            self.pred_data.append((juvenile_a_killed + dauer_a_killed, juvenile_c_killed + dauer_c_killed))
        
    
    def take_a_step(self):
        """Calculate the population composition at t+1.
        """
        self.pop = np.matmul(self.comb_u_mat + self.comb_f_mat, self.pop)
        self.pop = np.matrix.round(self.pop, fp_precision)
        if self.α > 0:
            self.predation()
        if self.consumption > 0.0:
            total_conusmers = self.n_consumers
            r_min = total_conusmers * self.consumption
            if self.resource < r_min:
                self.starv = True
            else:
                self.resource -= self.consumption * total_conusmers
                self.resource = round(self.resource, fp_precision)
            
            
class MetaPopulation(object):
    
    kernel = np.array([[0, 1, 0],
                       [1,-4, 1],
                       [0, 1, 0]])

    def __init__(self, dim, fec_pars, mf_prob, r_rate, pred_rate=1e-3, c=0.002):
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
        self.fec_pars = fec_pars
        self.mf_prob = mf_prob
        self.r_rate = r_rate
        self.index = [(i,j) for i in range(dim) for j in range(dim)]
        self.fill_pop()
        self.init_pos = []
        
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
            ind = np.random.binomial(1, 0.5, size=len(self.index))
            diet_comp = {j:'Novo' if i else 'OP50' for i,j in zip(ind, self.index)}
        for i in self.index:
            self.metapop[i].diet = diet_comp[i]
        
    def fill_pop(self):
        """Fill the MetaPopulation with empty population objects.
        """
        n0_empty = np.array([[0],  [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],  [0], [0], [0], [0], [0], [0], [0], [0], [0]])
        self.metapop = {i:Population(n0_empty, self.fec_pars, self.mf_prob, consumption=self.cons_rate, α=self.pred_rate) for i in self.index}
        
    def add_pop(self, loc, strain):
        """Add Population to specific location on the lattice. 

        Args:
            loc (tuple): The location on the lattice.
            strain (str): 'A' or 'C'.
        """
        if strain == 'A':
            n0_A = np.array([[0], [0], [50], [0], [0], [0], [0], [0], [0], [0], [0],  [0], [0], [0], [0], [0], [0], [0], [0], [0]])
            self.metapop[loc] = Population(n0_A, self.fec_pars, self.mf_prob, consumption=self.cons_rate, α=self.pred_rate)
            self.init_pos.append(('P',loc[0], loc[1]))
        else:
            n0_C = np.array([[0],  [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [50], [0], [0], [0], [0], [0], [0], [0]])
            self.metapop[loc] = Population(n0_C, self.fec_pars, self.mf_prob, consumption=self.cons_rate, α=self.pred_rate)
            self.init_pos.append(('NP',loc[0], loc[1]))

    def add_pop_rand(self, strains):
        n_strains = len(strains)
        spots = np.random.choice(range(len(self.index)), size=n_strains, replace=False)
        for i in range(n_strains):
            if strains[i] == 'A':
                n0_A = np.array([[0], [0], [50], [0], [0], [0], [0], [0], [0], [0], [0],  [0], [0], [0], [0], [0], [0], [0], [0], [0]])
                self.metapop[self.index[spots[i]]] = Population(n0_A, self.fec_pars, self.mf_prob, consumption=self.cons_rate, α=self.pred_rate)
                self.init_pos.append(('P', self.index[spots[i]][0], self.index[spots[i]][1]))
            else:
                n0_C = np.array([[0],  [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [50], [0], [0], [0], [0], [0], [0], [0]])
                self.metapop[self.index[spots[i]]] = Population(n0_C, self.fec_pars, self.mf_prob, consumption=self.cons_rate, α=self.pred_rate)
                self.init_pos.append(('NP',  self.index[spots[i]][0], self.index[spots[i]][1]))

    def reset_food(self):
        """Set the food in all subpopulations to r0.
        """
        for i in self.index:
            self.metapop[i].replenish()

    @property
    def daur_dist(self):
        """The number of dauer larvae on the MetaPopulation.
        """
        dist_A = np.zeros((self.dim, self.dim))
        dist_C = np.zeros((self.dim, self.dim))
        for i in self.index:
            dist_A[i[0]][i[1]] = round(self.metapop[i].pop[2][0], fp_precision)
            dist_C[i[0]][i[1]] = round(self.metapop[i].pop[12][0], fp_precision)
        return dist_A, dist_C
    
    @property
    def r_dist(self):
        """Get the number of resource in each subpopulation.
        """
        r_patt = np.zeros((self.dim, self.dim))
        for i in self.index:
            r_patt[i[0]][i[1]] = round(self.metapop[i].resource, fp_precision)
        return r_patt
    
    @property
    def ra_dist(self):
        """Get the number of reproducing adults.
        """
        dist_A = np.zeros((self.dim, self.dim))
        dist_C = np.zeros((self.dim, self.dim))
        for i in self.index:
            dist_A[i[0]][i[1]] = round(np.sum(self.metapop[i].pop[4:9]), fp_precision)
            dist_C[i[0]][i[1]] = round(np.sum(self.metapop[i].pop[14:19]), fp_precision)
        return dist_A, dist_C
    
    @staticmethod
    def migrate(lattice, migration_rate):
        rows, cols = lattice.shape
        new_lattice = np.copy(lattice)
        for i in range(rows):
            for j in range(cols):
                dauer = lattice[i, j]
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        new_i, new_j = i + dx, j + dy
                        if 0 <= new_i < rows and 0 <= new_j < cols:
                            disp_dauer = round(dauer * migration_rate, fp_precision)
                            new_lattice[i, j] -= disp_dauer
                            new_lattice[new_i, new_j] += disp_dauer
        return new_lattice

    def disperse_dauer(self):
        a, b = self.daur_dist
        new_dauer_a = self.migrate(a, self.r_rate[0])
        new_dauer_b = self.migrate(b, self.r_rate[1])
        for i in self.index:
            self.metapop[i].pop[2][0]  = new_dauer_a[i[0]][i[1]]
            self.metapop[i].pop[12][0]  = new_dauer_b[i[0]][i[1]]

    def simulate_pops_one_step(self):
        """Calculate the composition of the MetaPopulation at t+1.
        """
        for i in self.index:
            self.metapop[i].take_a_step()