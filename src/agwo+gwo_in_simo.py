import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.linalg as la

# ===================== MATLAB SINC FUNCTION =====================
def sinc_matlab(x):
    """
    MATLAB's sinc function: sin(x)/x
    """
    x = np.asarray(x)
    result = np.ones_like(x, dtype=np.float64)
    mask = x != 0
    result[mask] = np.sin(x[mask]) / x[mask]
    return result

# ===================== WATER-FILLING FUNCTION =====================
def water_filling(P_total, sigma2, a):
    """
    WF: water-filling solution
    P_total: total transmit power
    sigma2: noise power  
    a: singular value vector of MIMO channel
    PA_WF: power allocation solution for each stream using WF
    """
    a = np.array(a).flatten()
    n = len(a)
    
    P_sum = 0
    ii = 0
    
    for ii in range(n - 1):
        P_sum += (ii + 1) * (sigma2 / (a[ii + 1]**2) - sigma2 / (a[ii]**2))
        if P_sum >= P_total:
            break
    
    ii += 1
    
    if P_sum >= P_total:
        PA_WF = sigma2 / (a[ii]**2) - sigma2 / (a[:ii]**2) - (P_sum - P_total) / ii
        PA_WF_full = np.zeros(n)
        PA_WF_full[:ii] = PA_WF
        return PA_WF_full
    else:
        PA_WF = sigma2 / (a[ii]**2) - sigma2 / (a[:ii+1]**2) + (P_total - P_sum) / (ii + 1)
        return PA_WF

# ===================== ADAPTIVE GWO ALGORITHM (AGWO) =====================
def initialization(SearchAgents_no, dim, ub, lb):
    """
    Initialize positions of search agents
    """
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(SearchAgents_no):
        Positions[i, :] = np.random.rand(dim) * (ub - lb) + lb
    return Positions

class AdaptiveGWOCoefficient:
    """Adaptive coefficient manager for AGWO"""
    def __init__(self, Max_iter):
        self.Max_iter = Max_iter
        self.convergence_history = []
        self.improvement_rate = []
        self.diversity_history = []
        
    def calculate_improvement_rate(self, current_score, previous_score):
        """Calculate improvement rate"""
        if previous_score > 0:
            improvement = (previous_score - current_score) / previous_score
        else:
            improvement = 0
        return max(0, improvement)
    
    def calculate_diversity(self, positions):
        """Calculate population diversity"""
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        return np.mean(distances)
    
    def update_adaptive_coefficient(self, t, current_score, positions):
        """Update adaptive coefficient 'a' based on convergence progress"""
        # Store history
        self.convergence_history.append(current_score)
        
        if len(self.convergence_history) >= 2:
            improvement = self.calculate_improvement_rate(
                current_score, self.convergence_history[-2]
            )
            self.improvement_rate.append(improvement)
        
        # Calculate diversity
        diversity = self.calculate_diversity(positions)
        self.diversity_history.append(diversity)
        
        # Base decay
        base_a = 2 - t * (2 / self.Max_iter)
        
        # Adaptive adjustment based on convergence progress
        if len(self.improvement_rate) >= 5:
            avg_improvement = np.mean(self.improvement_rate[-5:])
            avg_diversity = np.mean(self.diversity_history[-5:])
            
            # 🔥 ADAPTIVE LOGIC:
            # If improvement is slow → explore more (increase a)
            # If improvement is fast → exploit more (decrease a)
            
            if avg_improvement < 0.001:  # Slow improvement
                # Increase exploration by 20%
                adaptive_factor = 1.2
                adjustment = 0.3 * (1 - avg_diversity)  # Encourage exploration
            elif avg_improvement > 0.01:  # Fast improvement
                # Increase exploitation by 15%
                adaptive_factor = 0.85
                adjustment = -0.2 * avg_diversity  # Encourage exploitation
            else:  # Moderate improvement
                adaptive_factor = 1.0
                adjustment = 0
            
            # Apply adaptive adjustment
            base_a = base_a * adaptive_factor + adjustment
            
            # Ensure a stays in reasonable bounds [0.1, 2.5]
            base_a = max(0.1, min(2.5, base_a))
        
        return base_a
    
    def get_adaptive_weights(self, t, fitness_values):
        """Get adaptive weights for alpha, beta, delta"""
        if len(fitness_values) < 3:
            return 0.5, 0.3, 0.2
        
        # Calculate fitness statistics
        fitness_std = np.std(fitness_values)
        fitness_mean = np.mean(fitness_values)
        diversity = min(fitness_std / (fitness_mean + 1e-10), 1.0)
        
        # Adaptive weighting based on iteration progress
        progress = t / self.Max_iter
        
        if progress < 0.3:  # Early stage: more exploration
            w_alpha = 0.4 + 0.2 * diversity
            w_beta = 0.35 + 0.1 * diversity
            w_delta = 0.25 - 0.3 * diversity
        elif progress < 0.7:  # Middle stage: balanced
            w_alpha = 0.5 + 0.1 * diversity
            w_beta = 0.3 + 0.1 * diversity
            w_delta = 0.2 - 0.2 * diversity
        else:  # Late stage: more exploitation
            w_alpha = 0.7 - 0.2 * diversity
            w_beta = 0.2 + 0.1 * diversity
            w_delta = 0.1 + 0.1 * diversity
        
        # Normalize
        total = w_alpha + w_beta + w_delta
        return w_alpha/total, w_beta/total, w_delta/total

def AGWO(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    """
    Adaptive Grey Wolf Optimizer 🔥
    Best balance between exploration & exploitation
    """
    # initialize alpha, beta, and delta_pos
    Alpha_pos = np.zeros(dim)
    Alpha_score = float('inf')
    
    Beta_pos = np.zeros(dim)
    Beta_score = float('inf')
    
    Delta_pos = np.zeros(dim)
    Delta_score = float('inf')
    
    # Initialize positions
    Positions = initialization(SearchAgents_no, dim, ub, lb)
    
    # Initialize adaptive coefficient manager
    adaptive_coeff = AdaptiveGWOCoefficient(Max_iter)
    
    Convergence_curve = np.zeros(Max_iter)
    
    # Main loop
    for t in range(Max_iter):
        fitness_values = []
        
        # Calculate fitness for all agents
        for i in range(SearchAgents_no):
            # Boundary handling
            for d in range(dim):
                if Positions[i, d] > ub[d]:
                    Positions[i, d] = ub[d] - (Positions[i, d] - ub[d]) * np.random.rand()
                elif Positions[i, d] < lb[d]:
                    Positions[i, d] = lb[d] + (lb[d] - Positions[i, d]) * np.random.rand()
            
            fitness = fobj(Positions[i, :])
            fitness_values.append(fitness)
            
            # Update leaders
            if fitness < Alpha_score:
                Delta_score = Beta_score
                Delta_pos = Beta_pos.copy()
                
                Beta_score = Alpha_score
                Beta_pos = Alpha_pos.copy()
                
                Alpha_score = fitness
                Alpha_pos = Positions[i, :].copy()
            elif fitness < Beta_score:
                Delta_score = Beta_score
                Delta_pos = Beta_pos.copy()
                
                Beta_score = fitness
                Beta_pos = Positions[i, :].copy()
            elif fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()
        
        # Get adaptive coefficient
        a = adaptive_coeff.update_adaptive_coefficient(t, Alpha_score, Positions)
        
        # Get adaptive weights
        w_alpha, w_beta, w_delta = adaptive_coeff.get_adaptive_weights(t, fitness_values)
        
        # Update positions with adaptive mechanism
        for i in range(SearchAgents_no):
            # Generate random coefficients
            r1_alpha = np.random.rand(dim)
            r2_alpha = np.random.rand(dim)
            r1_beta = np.random.rand(dim)
            r2_beta = np.random.rand(dim)
            r1_delta = np.random.rand(dim)
            r2_delta = np.random.rand(dim)
            
            # Calculate A and C coefficients
            A_alpha = 2 * a * r1_alpha - a
            C_alpha = 2 * r2_alpha
            
            A_beta = 2 * a * r1_beta - a
            C_beta = 2 * r2_beta
            
            A_delta = 2 * a * r1_delta - a
            C_delta = 2 * r2_delta
            
            # Calculate distances
            D_alpha = np.abs(C_alpha * Alpha_pos - Positions[i, :])
            D_beta = np.abs(C_beta * Beta_pos - Positions[i, :])
            D_delta = np.abs(C_delta * Delta_pos - Positions[i, :])
            
            # Calculate new positions
            X_alpha = Alpha_pos - A_alpha * D_alpha
            X_beta = Beta_pos - A_beta * D_beta
            X_delta = Delta_pos - A_delta * D_delta
            
            # Apply adaptive weights
            new_position = w_alpha * X_alpha + w_beta * X_beta + w_delta * X_delta
            
            # Add small random perturbation for exploration
            if t < Max_iter * 0.8:  # Only in early/mid stages
                perturbation = 0.1 * (1 - t/Max_iter) * np.random.randn(dim)
                new_position += perturbation
            
            # Update position
            Positions[i, :] = new_position
        
        Convergence_curve[t] = Alpha_score
        
        # Display progress
        if t % 10 == 0:
            print(f"      Iter {t:3d}: NMSE = {Alpha_score:.6f}, a = {a:.3f}")
        
        # Early stopping check
        if t > 50:
            recent_improvement = np.mean(np.abs(np.diff(Convergence_curve[t-10:t])))
            if recent_improvement < 1e-8:
                print(f"      Early convergence at iteration {t}")
                Convergence_curve = Convergence_curve[:t+1]
                break
    
    return Alpha_score, Alpha_pos, Convergence_curve

# ===================== SIM-MIMO SYSTEM WITH AGWO =====================
class SIM_MIMO_AGWO:
    def __init__(self, M, N, S, L, K, lamda, d_element_spacing, 
                 d_layer_spacing_transmit, d_layer_spacing_receive, 
                 N_max=10, pathloss=1.0):
        """
        Initialize SIM-MIMO system
        """
        self.M = M
        self.N = N
        self.S = S
        self.L = L
        self.K = K
        self.lamda = lamda
        self.d_element_spacing = d_element_spacing
        self.d_layer_spacing_transmit = d_layer_spacing_transmit
        self.d_layer_spacing_receive = d_layer_spacing_receive
        self.N_max = N_max
        self.pathloss = pathloss
        
        # Generate system matrices
        self.generate_matrices()
        
    def generate_matrices(self):
        """Generate all SIM matrices"""
        # Initialize matrices
        self.W_T = np.zeros((self.M, self.M), dtype=np.complex128)
        self.Corr_T = np.zeros((self.M, self.M))
        self.U_R = np.zeros((self.N, self.N), dtype=np.complex128)
        self.Corr_R = np.zeros((self.N, self.N))
        self.W_T_1 = np.zeros((self.M, self.S), dtype=np.complex128)
        self.U_R_1 = np.zeros((self.S, self.N), dtype=np.complex128)
        
        # Calculate indices
        indices_M = np.arange(self.M)
        m_z_M = indices_M // self.N_max + 1
        m_x_M = indices_M % self.N_max + 1
        
        indices_N = np.arange(self.N)
        m_z_N = indices_N // self.N_max + 1
        m_x_N = indices_N % self.N_max + 1
        
        # Calculate TX-SIM matrices
        for mm1 in range(self.M):
            for mm2 in range(self.M):
                d_temp = np.sqrt((m_x_M[mm1] - m_x_M[mm2])**2 + 
                               (m_z_M[mm1] - m_z_M[mm2])**2) * self.d_element_spacing
                d_temp2 = np.sqrt(self.d_layer_spacing_transmit**2 + d_temp**2)
                
                self.W_T[mm2, mm1] = (self.lamda/(4*np.pi*d_temp2) * 
                                    np.exp(-1j*2*np.pi*d_temp2/self.lamda))
                self.Corr_T[mm2, mm1] = sinc_matlab(2*d_temp/self.lamda)
        
        # Calculate RX-SIM matrices
        for nn1 in range(self.N):
            for nn2 in range(self.N):
                d_temp = np.sqrt((m_x_N[nn1] - m_x_N[nn2])**2 + 
                               (m_z_N[nn1] - m_z_N[nn2])**2) * self.d_element_spacing
                d_temp2 = np.sqrt(self.d_layer_spacing_receive**2 + d_temp**2)
                
                self.U_R[nn2, nn1] = (self.lamda/(4*np.pi*d_temp2) * 
                                    np.exp(-1j*2*np.pi*d_temp2/self.lamda))
                self.Corr_R[nn2, nn1] = sinc_matlab(2*d_temp/self.lamda)
        
        # Calculate channel matrices
        for mm in range(self.M):
            for nn in range(self.S):
                d_transmit = np.sqrt(
                    self.d_layer_spacing_transmit**2 + 
                    ((m_x_M[mm] - (1 + self.N_max)/2) * self.d_element_spacing)**2 +
                    ((m_z_M[mm] - (1 + self.N_max)/2) * self.d_element_spacing - 
                     (nn - (1 + self.S)/2) * self.lamda/2)**2
                )
                self.W_T_1[mm, nn] = (self.lamda/(4*np.pi*d_transmit) * 
                                    np.exp(-1j*2*np.pi*d_transmit/self.lamda))
        
        for mm in range(self.N):
            for nn in range(self.S):
                d_receive = np.sqrt(
                    self.d_layer_spacing_receive**2 +
                    ((m_x_N[mm] - (1 + self.N_max)/2) * self.d_element_spacing)**2 +
                    ((m_z_N[mm] - (1 + self.N_max)/2) * self.d_element_spacing - 
                     (nn - (1 + self.S)/2) * self.lamda/2)**2
                )
                self.U_R_1[nn, mm] = (self.lamda/(4*np.pi*d_receive) * 
                                    np.exp(-1j*2*np.pi*d_receive/self.lamda))
    
    def vector_to_phase_shifts(self, vec):
        """Convert optimization vector to phase shift matrices"""
        # First M*L elements are real parts, next M*L are imaginary parts for TX
        tx_real = vec[:self.M * self.L].reshape(self.M, self.L)
        tx_imag = vec[self.M * self.L:2 * self.M * self.L].reshape(self.M, self.L)
        phase_transmit = tx_real + 1j * tx_imag
        
        # Next N*K elements are real parts, last N*K are imaginary parts for RX
        rx_real = vec[2 * self.M * self.L:2 * self.M * self.L + self.N * self.K].reshape(self.N, self.K)
        rx_imag = vec[2 * self.M * self.L + self.N * self.K:].reshape(self.N, self.K)
        phase_receive = rx_real + 1j * rx_imag
        
        # Normalize to unit magnitude
        eps = 1e-12
        phase_transmit = phase_transmit / (np.abs(phase_transmit) + eps)
        phase_receive = phase_receive / (np.abs(phase_receive) + eps)
        
        return phase_transmit, phase_receive
    
    def calculate_sim_response(self, phase_transmit, phase_receive):
        """Calculate SIM response for given phase shifts"""
        # TX-SIM response
        P = np.diag(phase_transmit[:, 0]) @ self.W_T_1
        for l in range(1, self.L):
            P = np.diag(phase_transmit[:, l]) @ self.W_T @ P
        
        # RX-SIM response
        Q = self.U_R_1 @ np.diag(phase_receive[:, 0])
        for k in range(1, self.K):
            Q = Q @ self.U_R @ np.diag(phase_receive[:, k])
        
        return P, Q
    
    def fitness_function(self, vec, G, H_true, Norm_H):
        """Fitness function for AGWO (minimize NMSE)"""
        # Convert vector to phase shifts
        phase_transmit, phase_receive = self.vector_to_phase_shifts(vec)
        
        # Calculate SIM response
        P, Q = self.calculate_sim_response(phase_transmit, phase_receive)
        
        # Calculate end-to-end channel
        H_SIM = Q @ G @ P
        H_SIM_vec = H_SIM.flatten('F')
        H_true_vec = H_true.flatten('F')
        
        # Calculate compensation factor
        Factor = np.linalg.pinv(H_SIM_vec.reshape(-1, 1).T @ H_SIM_vec.reshape(-1, 1)) @ \
                H_SIM_vec.reshape(-1, 1).T @ H_true_vec.reshape(-1, 1)
        Factor = Factor[0, 0]
        
        # Calculate NMSE
        nmse = np.linalg.norm(Factor * H_SIM - H_true)**2 / Norm_H
        
        return nmse
    
    def optimize_phase_shifts_agwo(self, G, H_true, Norm_H, PA_WF, 
                                   SearchAgents_no=30, Max_iter=100):
        """Optimize phase shifts using Adaptive GWO"""
        # Optimization dimensions
        dim = 2 * (self.M * self.L + self.N * self.K)
        
        # Bounds
        lb = -1 * np.ones(dim)
        ub = 1 * np.ones(dim)
        
        # Create fitness function with current channel
        def fobj(vec):
            return self.fitness_function(vec, G, H_true, Norm_H)
        
        print(f"    AGWO Optimization: dim={dim}, agents={SearchAgents_no}, max_iter={Max_iter}")
        
        # Run Adaptive GWO
        start_time = time.time()
        Alpha_score, Alpha_pos, Convergence_curve = AGWO(
            SearchAgents_no, Max_iter, lb, ub, dim, fobj
        )
        opt_time = time.time() - start_time
        
        # Convert best solution to phase shifts
        best_phase_transmit, best_phase_receive = self.vector_to_phase_shifts(Alpha_pos)
        
        return best_phase_transmit, best_phase_receive, Alpha_score, Convergence_curve, opt_time
    
    def calculate_capacity(self, phase_transmit, phase_receive, G, PA_WF, Sigma2, H_true):
        """Calculate capacity for given phase shifts"""
        # Calculate SIM response
        P, Q = self.calculate_sim_response(phase_transmit, phase_receive)
        
        # Calculate end-to-end channel
        H_SIM = Q @ G @ P
        H_SIM_vec = H_SIM.flatten('F')
        H_true_vec = H_true.flatten('F')
        
        # Calculate compensation factor
        Factor = np.linalg.pinv(H_SIM_vec.reshape(-1, 1).T @ H_SIM_vec.reshape(-1, 1)) @ \
                H_SIM_vec.reshape(-1, 1).T @ H_true_vec.reshape(-1, 1)
        Factor = Factor[0, 0]
        
        # Calculate capacity
        C_single_stream = np.zeros(self.S)
        for pp in range(self.S):
            interference = np.sum(np.abs(Factor * H_SIM[pp, :])**2 * PA_WF) - \
                          PA_WF[pp] * np.abs(Factor * H_SIM[pp, pp])**2
            C_single_stream[pp] = np.log2(1 + 
                PA_WF[pp] * np.abs(Factor * H_SIM[pp, pp])**2 / 
                (Sigma2 + interference)
            )
        
        return np.sum(C_single_stream), Factor, H_SIM

# ===================== COMPARISON: STANDARD GWO =====================
def GWO_standard(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    """
    Standard Grey Wolf Optimizer (for comparison)
    """
    Alpha_pos = np.zeros(dim)
    Alpha_score = float('inf')
    
    Beta_pos = np.zeros(dim)
    Beta_score = float('inf')
    
    Delta_pos = np.zeros(dim)
    Delta_score = float('inf')
    
    Positions = initialization(SearchAgents_no, dim, ub, lb)
    Convergence_curve = np.zeros(Max_iter)
    
    # Main loop
    for t in range(Max_iter):
        for i in range(SearchAgents_no):
            # Boundary handling
            for d in range(dim):
                if Positions[i, d] > ub[d]:
                    Positions[i, d] = ub[d]
                elif Positions[i, d] < lb[d]:
                    Positions[i, d] = lb[d]
            
            fitness = fobj(Positions[i, :])
            
            # Update leaders
            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Positions[i, :].copy()
            
            if fitness > Alpha_score and fitness < Beta_score:
                Beta_score = fitness
                Beta_pos = Positions[i, :].copy()
            
            if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()
        
        # Standard linear decay
        a = 2 - t * (2 / Max_iter)
        
        # Update positions
        for i in range(SearchAgents_no):
            for j in range(dim):
                r1 = np.random.rand()
                r2 = np.random.rand()
                
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha
                
                r1 = np.random.rand()
                r2 = np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta
                
                r1 = np.random.rand()
                r2 = np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta
                
                Positions[i, j] = (X1 + X2 + X3) / 3
        
        Convergence_curve[t] = Alpha_score
    
    return Alpha_score, Alpha_pos, Convergence_curve

# ===================== MAIN SIMULATION =====================
def main():
    # Parameters setup
    Thickness = 0.05
    Pt = 10**(20/10)
    Sigma2 = 10**(-110/10)
    c = 3e8
    f0 = 28e9
    lamda = c/f0
    N_max = 10
    
    # Pathloss calculation
    PL = -20*np.log10(4*np.pi/lamda) - 35*np.log10(250)
    pathloss = 10**(PL/10)
    
    # SIM parameters (adjust for faster testing)
    M = 30
    N = 30
    d_element_spacing = lamda/2
    S = 3
    MonteCarlo = 5
    Max_L = 10
    K = 3  # From your MATLAB code
    
    # AGWO parameters
    SearchAgents_no = 20
    Max_iter = 50
    
    print("="*70)
    print("SIM-MIMO SIMULATION WITH ADAPTIVE GWO (AGWO) 🔥")
    print("="*70)
    print(f"System Parameters:")
    print(f"  M={M}, N={N}, S={S}, K={K}")
    print(f"  MonteCarlo={MonteCarlo}, Max_L={Max_L}")
    print(f"AGWO Parameters:")
    print(f"  SearchAgents_no={SearchAgents_no}, Max_iter={Max_iter}")
    print("="*70)
    print()
    
    np.random.seed(42)
    
    # Initialize arrays for AGWO
    NMSE_agwo = np.zeros(Max_L)
    Capacity_agwo = np.zeros(Max_L)
    Time_agwo = np.zeros(Max_L)
    
    # Initialize arrays for standard GWO (for comparison)
    NMSE_std = np.zeros(Max_L)
    Capacity_std = np.zeros(Max_L)
    Time_std = np.zeros(Max_L)
    
    # Store convergence curves
    conv_curves_agwo = []
    conv_curves_std = []
    
    # Main loop
    for ii in range(Max_L):
        L = ii + 1
        print(f"\n{'='*60}")
        print(f"Processing L={L}, K={K}")
        print(f"{'='*60}")
        
        # Calculate spacing
        d_layer_spacing_transmit = Thickness/L
        d_layer_spacing_receive = Thickness/K
        
        # Create system
        sim_system = SIM_MIMO_AGWO(M, N, S, L, K, lamda, d_element_spacing,
                                  d_layer_spacing_transmit, d_layer_spacing_receive,
                                  N_max, pathloss)
        
        # Initialize arrays for Monte Carlo
        NMSE_vals_agwo = np.zeros(MonteCarlo)
        Capacity_vals_agwo = np.zeros(MonteCarlo)
        Time_vals_agwo = np.zeros(MonteCarlo)
        
        NMSE_vals_std = np.zeros(MonteCarlo)
        Capacity_vals_std = np.zeros(MonteCarlo)
        Time_vals_std = np.zeros(MonteCarlo)
        
        for jj in range(MonteCarlo):
            print(f"\n  Monte Carlo {jj+1}/{MonteCarlo}")
            
            # Generate channel
            G_independent = np.sqrt(1/2) * (np.random.randn(N, M) + 
                                           1j*np.random.randn(N, M))
            
            Corr_R_sqrt = la.sqrtm(sim_system.Corr_R + 1e-8*np.eye(N))
            Corr_T_sqrt = la.sqrtm(sim_system.Corr_T + 1e-8*np.eye(M))
            G = np.sqrt(pathloss) * Corr_R_sqrt @ G_independent @ Corr_T_sqrt
            
            # SVD
            U, G_svd, Vh = la.svd(G, full_matrices=False)
            H_true = np.diag(G_svd[:S])
            H_true_vec = H_true.flatten('F')
            Norm_H = np.linalg.norm(H_true_vec)**2
            
            h_diag = np.diag(H_true)
            PA_WF = water_filling(Pt, Sigma2, h_diag)
            
            # =========== ADAPTIVE GWO ===========
            print("    Running ADAPTIVE GWO...")
            phase_transmit_agwo, phase_receive_agwo, nmse_agwo, conv_agwo, time_agwo = \
                sim_system.optimize_phase_shifts_agwo(
                    G, H_true, Norm_H, PA_WF, SearchAgents_no, Max_iter
                )
            
            capacity_agwo, Factor_agwo, H_SIM_agwo = sim_system.calculate_capacity(
                phase_transmit_agwo, phase_receive_agwo, G, PA_WF, Sigma2, H_true
            )
            
            NMSE_vals_agwo[jj] = nmse_agwo
            Capacity_vals_agwo[jj] = capacity_agwo
            Time_vals_agwo[jj] = time_agwo
            
            if jj == 0:
                conv_curves_agwo.append(conv_agwo)
            
            print(f"    AGWO: NMSE={nmse_agwo:.6f}, Capacity={capacity_agwo:.4f}, Time={time_agwo:.2f}s")
            
            # =========== STANDARD GWO (for comparison) ===========
            print("    Running STANDARD GWO...")
            
            def fobj_std(vec):
                return sim_system.fitness_function(vec, G, H_true, Norm_H)
            
            start_time = time.time()
            Alpha_score_std, Alpha_pos_std, conv_std = GWO_standard(
                SearchAgents_no, Max_iter, -np.ones(2*(M*L + N*K)), 
                np.ones(2*(M*L + N*K)), 2*(M*L + N*K), fobj_std
            )
            time_std = time.time() - start_time
            
            phase_transmit_std, phase_receive_std = sim_system.vector_to_phase_shifts(Alpha_pos_std)
            capacity_std, Factor_std, H_SIM_std = sim_system.calculate_capacity(
                phase_transmit_std, phase_receive_std, G, PA_WF, Sigma2, H_true
            )
            
            NMSE_vals_std[jj] = Alpha_score_std
            Capacity_vals_std[jj] = capacity_std
            Time_vals_std[jj] = time_std
            
            if jj == 0:
                conv_curves_std.append(conv_std)
            
            print(f"    STD GWO: NMSE={Alpha_score_std:.6f}, Capacity={capacity_std:.4f}, Time={time_std:.2f}s")
            
            # Improvement comparison
            improvement = (Alpha_score_std - nmse_agwo) / Alpha_score_std * 100
            print(f"    AGWO improvement: {improvement:.2f}% better NMSE")
        
        # Calculate averages
        NMSE_agwo[ii] = np.mean(NMSE_vals_agwo)
        Capacity_agwo[ii] = np.mean(Capacity_vals_agwo)
        Time_agwo[ii] = np.mean(Time_vals_agwo)
        
        NMSE_std[ii] = np.mean(NMSE_vals_std)
        Capacity_std[ii] = np.mean(Capacity_vals_std)
        Time_std[ii] = np.mean(Time_vals_std)
        
        print(f"\n  Summary L={L}:")
        print(f"    AGWO:  NMSE={NMSE_agwo[ii]:.6f}, Capacity={Capacity_agwo[ii]:.4f}, Time={Time_agwo[ii]:.2f}s")
        print(f"    STD:   NMSE={NMSE_std[ii]:.6f}, Capacity={Capacity_std[ii]:.4f}, Time={Time_std[ii]:.2f}s")
    
    # ===================== VISUALIZATION =====================
    print("\n" + "="*70)
    print("RESULTS VISUALIZATION")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: NMSE Comparison
    axes[0, 0].plot(range(1, Max_L + 1), NMSE_agwo, 'o-', linewidth=2, markersize=8, label='AGWO')
    axes[0, 0].plot(range(1, Max_L + 1), NMSE_std, 's--', linewidth=2, markersize=8, label='Standard GWO')
    axes[0, 0].set_xlabel('Number of TX-SIM Layers (L)', fontsize=12)
    axes[0, 0].set_ylabel('Average NMSE', fontsize=12)
    axes[0, 0].set_title('NMSE Comparison: AGWO vs Standard GWO 🔥', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0.5, Max_L + 0.5)
    
    # Plot 2: Capacity Comparison
    axes[0, 1].plot(range(1, Max_L + 1), Capacity_agwo, 'o-', linewidth=2, markersize=8, 
                    color='green', label='AGWO')
    axes[0, 1].plot(range(1, Max_L + 1), Capacity_std, 's--', linewidth=2, markersize=8, 
                    color='orange', label='Standard GWO')
    axes[0, 1].set_xlabel('Number of TX-SIM Layers (L)', fontsize=12)
    axes[0, 1].set_ylabel('Average Capacity (bps/Hz)', fontsize=12)
    axes[0, 1].set_title('Capacity Comparison', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0.5, Max_L + 0.5)
    
    # Plot 3: Time Comparison
    x = np.arange(1, Max_L + 1)
    width = 0.35
    axes[0, 2].bar(x - width/2, Time_agwo, width, label='AGWO', alpha=0.7)
    axes[0, 2].bar(x + width/2, Time_std, width, label='Standard GWO', alpha=0.7)
    axes[0, 2].set_xlabel('Number of TX-SIM Layers (L)', fontsize=12)
    axes[0, 2].set_ylabel('Average Time (s)', fontsize=12)
    axes[0, 2].set_title('Computation Time Comparison', fontsize=14)
    axes[0, 2].set_xticks(x)
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    axes[0, 2].legend()
    
    # Plot 4: Convergence Curves (L=3 example)
    if len(conv_curves_agwo) >= 3:
        L_show = 3
        axes[1, 0].plot(conv_curves_agwo[L_show-1], 'b-', linewidth=2, label='AGWO')
        axes[1, 0].plot(conv_curves_std[L_show-1], 'r--', linewidth=2, label='Standard GWO')
        axes[1, 0].set_xlabel('Iteration', fontsize=12)
        axes[1, 0].set_ylabel('NMSE', fontsize=12)
        axes[1, 0].set_title(f'Convergence Curves (L={L_show})', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')
    
    # Plot 5: Improvement Percentage
    improvement_pct = (NMSE_std - NMSE_agwo) / NMSE_std * 100
    axes[1, 1].bar(range(1, Max_L + 1), improvement_pct, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Number of TX-SIM Layers (L)', fontsize=12)
    axes[1, 1].set_ylabel('Improvement (%)', fontsize=12)
    axes[1, 1].set_title('AGWO Improvement over Standard GWO', fontsize=14)
    axes[1, 1].set_xticks(range(1, Max_L + 1))
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, val in enumerate(improvement_pct):
        axes[1, 1].text(i+1, val, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 6: Performance vs Complexity Trade-off
    axes[1, 2].scatter(Time_agwo, NMSE_agwo, s=150, alpha=0.7, 
                       c=range(1, Max_L + 1), cmap='viridis', label='AGWO')
    axes[1, 2].scatter(Time_std, NMSE_std, s=150, alpha=0.7, marker='s',
                       c=range(1, Max_L + 1), cmap='viridis', label='Standard GWO')
    axes[1, 2].set_xlabel('Time (s)', fontsize=12)
    axes[1, 2].set_ylabel('NMSE', fontsize=12)
    axes[1, 2].set_title('Performance-Complexity Trade-off', fontsize=14)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('agwo_vs_standard_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ===================== SAVE RESULTS =====================
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    results = {
        'L_values': list(range(1, Max_L + 1)),
        'NMSE_agwo': NMSE_agwo.tolist(),
        'Capacity_agwo': Capacity_agwo.tolist(),
        'Time_agwo': Time_agwo.tolist(),
        'NMSE_std': NMSE_std.tolist(),
        'Capacity_std': Capacity_std.tolist(),
        'Time_std': Time_std.tolist(),
        'Improvement_pct': improvement_pct.tolist(),
        'parameters': {
            'M': M, 'N': N, 'S': S, 'K': K,
            'MonteCarlo': MonteCarlo, 'Max_L': Max_L,
            'SearchAgents_no': SearchAgents_no, 'Max_iter': Max_iter,
            'Pt': Pt, 'Sigma2': Sigma2, 'f0': f0
        }
    }
    
    import json
    with open('agwo_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    np.savez('agwo_comparison.npz',
             NMSE_agwo=NMSE_agwo, Capacity_agwo=Capacity_agwo, Time_agwo=Time_agwo,
             NMSE_std=NMSE_std, Capacity_std=Capacity_std, Time_std=Time_std)
    
    print("Results saved to:")
    print("  agwo_comparison_results.json")
    print("  agwo_comparison.npz")
    print("  agwo_vs_standard_comparison.png")
    
    # ===================== FINAL SUMMARY =====================
    print("\n" + "="*70)
    print("FINAL SUMMARY: AGWO vs STANDARD GWO 🔥")
    print("="*70)
    print(f"{'L':>3} {'NMSE(AGWO)':<12} {'NMSE(STD)':<12} {'Improve%':<10} {'Cap(AGWO)':<12} {'Cap(STD)':<12}")
    print("-"*70)
    
    for i in range(Max_L):
        improve = improvement_pct[i]
        print(f"{i+1:>3} {NMSE_agwo[i]:<12.6f} {NMSE_std[i]:<12.6f} {improve:<10.2f} "
              f"{Capacity_agwo[i]:<12.4f} {Capacity_std[i]:<12.4f}")
    
    print("-"*70)
    avg_improvement = np.mean(improvement_pct)
    print(f"\nAverage Improvement: {avg_improvement:.2f}%")
    
    # Find best configurations
    best_nmse_agwo_idx = np.argmin(NMSE_agwo)
    best_cap_agwo_idx = np.argmax(Capacity_agwo)
    best_nmse_std_idx = np.argmin(NMSE_std)
    best_cap_std_idx = np.argmax(Capacity_std)
    
    print(f"\nBest AGWO NMSE: {NMSE_agwo[best_nmse_agwo_idx]:.6f} at L={best_nmse_agwo_idx+1}")
    print(f"Best AGWO Capacity: {Capacity_agwo[best_cap_agwo_idx]:.4f} at L={best_cap_agwo_idx+1}")
    print(f"Best STD NMSE: {NMSE_std[best_nmse_std_idx]:.6f} at L={best_nmse_std_idx+1}")
    print(f"Best STD Capacity: {Capacity_std[best_cap_std_idx]:.4f} at L={best_cap_std_idx+1}")
    
    print("\n🔥 ADAPTIVE GWO FEATURES:")
    print("  - Dynamic coefficient 'a' based on convergence progress")
    print("  - Adaptive weighting of alpha, beta, delta wolves")
    print("  - Exploration increased when improvement is slow")
    print("  - Exploitation increased when improvement is fast")
    print("  - Better balance between exploration and exploitation")
    print("="*70)

if __name__ == "__main__":
    main()