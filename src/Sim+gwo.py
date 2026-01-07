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

# ===================== GWO ALGORITHM =====================
def initialization(SearchAgents_no, dim, ub, lb):
    """
    Initialize positions of search agents
    """
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(SearchAgents_no):
        Positions[i, :] = np.random.rand(dim) * (ub - lb) + lb
    return Positions

def GWO(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    """
    Grey Wolf Optimizer
    Returns: Alpha_score, Alpha_pos, Convergence_curve
    """
    # initialize alpha, beta, and delta_pos
    Alpha_pos = np.zeros(dim)
    Alpha_score = float('inf')  # for minimization problems
    
    Beta_pos = np.zeros(dim)
    Beta_score = float('inf')
    
    Delta_pos = np.zeros(dim)
    Delta_score = float('inf')
    
    # Initialize the positions of search agents
    Positions = initialization(SearchAgents_no, dim, ub, lb)
    
    Convergence_curve = np.zeros(Max_iter)
    
    # Main loop
    for l in range(Max_iter):
        for i in range(SearchAgents_no):
            # Return back the search agents that go beyond the boundaries
            Flag4ub = Positions[i, :] > ub
            Flag4lb = Positions[i, :] < lb
            Positions[i, :] = (Positions[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb
            
            # Calculate objective function for each search agent
            fitness = fobj(Positions[i, :])
            
            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Alpha_score = fitness  # Update alpha
                Alpha_pos = Positions[i, :].copy()
            
            if fitness > Alpha_score and fitness < Beta_score:
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()
            
            if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()
        
        a = 2 - l * (2 / Max_iter)  # a decreases linearly from 2 to 0
        
        # Update the Position of search agents including omegas
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
        
        Convergence_curve[l] = Alpha_score
    
    return Alpha_score, Alpha_pos, Convergence_curve

# ===================== SIM-MIMO WITH GWO OPTIMIZATION =====================
class SIM_MIMO_GWO:
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
        
        # Current phase shifts
        self.phase_transmit = None
        self.phase_receive = None
        self.current_G = None
        self.current_H_true = None
        self.current_Norm_H = None
        self.current_PA_WF = None
        
    def generate_matrices(self):
        """Generate all SIM matrices"""
        # Initialize matrices
        self.W_T = np.zeros((self.M, self.M), dtype=np.complex128)
        self.Corr_T = np.zeros((self.M, self.M))
        self.U_R = np.zeros((self.N, self.N), dtype=np.complex128)
        self.Corr_R = np.zeros((self.N, self.N))
        self.W_T_1 = np.zeros((self.M, self.S), dtype=np.complex128)
        self.U_R_1 = np.zeros((self.S, self.N), dtype=np.complex128)
        
        # Calculate TX-SIM matrices
        for mm1 in range(self.M):
            m_z = (mm1 // self.N_max) + 1
            m_x = (mm1 % self.N_max) + 1
            
            for mm2 in range(self.M):
                n_z = (mm2 // self.N_max) + 1
                n_x = (mm2 % self.N_max) + 1
                
                d_temp = np.sqrt((m_x - n_x)**2 + (m_z - n_z)**2) * self.d_element_spacing
                d_temp2 = np.sqrt(self.d_layer_spacing_transmit**2 + d_temp**2)
                
                self.W_T[mm2, mm1] = (self.lamda/(4*np.pi*d_temp2) * 
                                    np.exp(-1j*2*np.pi*d_temp2/self.lamda))
                self.Corr_T[mm2, mm1] = sinc_matlab(2*d_temp/self.lamda)
        
        # Calculate RX-SIM matrices
        for nn1 in range(self.N):
            m_z = (nn1 // self.N_max) + 1
            m_x = (nn1 % self.N_max) + 1
            
            for nn2 in range(self.N):
                n_z = (nn2 // self.N_max) + 1
                n_x = (nn2 % self.N_max) + 1
                
                d_temp = np.sqrt((m_x - n_x)**2 + (m_z - n_z)**2) * self.d_element_spacing
                d_temp2 = np.sqrt(self.d_layer_spacing_receive**2 + d_temp**2)
                
                self.U_R[nn2, nn1] = (self.lamda/(4*np.pi*d_temp2) * 
                                    np.exp(-1j*2*np.pi*d_temp2/self.lamda))
                self.Corr_R[nn2, nn1] = sinc_matlab(2*d_temp/self.lamda)
        
        # Calculate channel matrices
        for mm in range(self.M):
            m_z = (mm // self.N_max) + 1
            m_x = (mm % self.N_max) + 1
            
            for nn in range(self.S):
                d_transmit = np.sqrt(
                    self.d_layer_spacing_transmit**2 + 
                    ((m_x - (1 + self.N_max)/2) * self.d_element_spacing)**2 +
                    ((m_z - (1 + self.N_max)/2) * self.d_element_spacing - 
                     (nn - (1 + self.S)/2) * self.lamda/2)**2
                )
                self.W_T_1[mm, nn] = (self.lamda/(4*np.pi*d_transmit) * 
                                    np.exp(-1j*2*np.pi*d_transmit/self.lamda))
        
        for mm in range(self.N):
            m_z = (mm // self.N_max) + 1
            m_x = (mm % self.N_max) + 1
            
            for nn in range(self.S):
                d_receive = np.sqrt(
                    self.d_layer_spacing_receive**2 +
                    ((m_x - (1 + self.N_max)/2) * self.d_element_spacing)**2 +
                    ((m_z - (1 + self.N_max)/2) * self.d_element_spacing - 
                     (nn - (1 + self.S)/2) * self.lamda/2)**2
                )
                self.U_R_1[nn, mm] = (self.lamda/(4*np.pi*d_receive) * 
                                    np.exp(-1j*2*np.pi*d_receive/self.lamda))
    
    def set_channel(self, G, H_true, Norm_H, PA_WF):
        """Set current channel realization"""
        self.current_G = G
        self.current_H_true = H_true
        self.current_Norm_H = Norm_H
        self.current_PA_WF = PA_WF
    
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
        
        # Normalize to unit magnitude (add epsilon to avoid division by zero)
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
    
    def fitness_function(self, vec):
        """Fitness function for GWO (minimize NMSE)"""
        # Convert vector to phase shifts
        phase_transmit, phase_receive = self.vector_to_phase_shifts(vec)
        
        # Calculate SIM response
        P, Q = self.calculate_sim_response(phase_transmit, phase_receive)
        
        # Calculate end-to-end channel
        H_SIM = Q @ self.current_G @ P
        H_SIM_vec = H_SIM.flatten('F')
        H_true_vec = self.current_H_true.flatten('F')
        
        # Calculate compensation factor
        Factor = np.linalg.pinv(H_SIM_vec.reshape(-1, 1).T @ H_SIM_vec.reshape(-1, 1)) @ \
                H_SIM_vec.reshape(-1, 1).T @ H_true_vec.reshape(-1, 1)
        Factor = Factor[0, 0]
        
        # Calculate NMSE
        nmse = np.linalg.norm(Factor * H_SIM - self.current_H_true)**2 / self.current_Norm_H
        
        return nmse
    
    def optimize_phase_shifts_gwo(self, G, H_true, Norm_H, PA_WF, 
                                 SearchAgents_no=30, Max_iter=100):
        """Optimize phase shifts using GWO"""
        # Set current channel
        self.set_channel(G, H_true, Norm_H, PA_WF)
        
        # Optimization dimensions: real and imaginary parts for all phase shifts
        dim = 2 * (self.M * self.L + self.N * self.K)
        
        # Bounds: real and imag parts between -1 and 1
        lb = -1 * np.ones(dim)
        ub = 1 * np.ones(dim)
        
        # Run GWO optimization
        Alpha_score, Alpha_pos, Convergence_curve = GWO(
            SearchAgents_no, Max_iter, lb, ub, dim, self.fitness_function
        )
        
        # Convert best solution to phase shifts
        best_phase_transmit, best_phase_receive = self.vector_to_phase_shifts(Alpha_pos)
        
        return best_phase_transmit, best_phase_receive, Alpha_score, Convergence_curve
    
    def calculate_capacity(self, phase_transmit, phase_receive, G, PA_WF, Sigma2):
        """Calculate capacity for given phase shifts"""
        # Calculate SIM response
        P, Q = self.calculate_sim_response(phase_transmit, phase_receive)
        
        # Calculate end-to-end channel
        H_SIM = Q @ G @ P
        H_SIM_vec = H_SIM.flatten('F')
        H_true_vec = self.current_H_true.flatten('F')
        
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
    
    # SIM parameters
    M = 50
    N = 50
    d_element_spacing = lamda/2
    S = 4
    MonteCarlo = 10
    Max_L = 10
    K = 1
    
    # GWO parameters
    SearchAgents_no = 30
    Max_iter = 100
    
    # Initialize arrays
    NMSE_average = np.zeros(Max_L)
    Capacity_average = np.zeros(Max_L)
    Time_average = np.zeros(Max_L)
    
    print("="*70)
    print("SIM-MIMO SIMULATION WITH GWO OPTIMIZATION")
    print("="*70)
    print(f"System Parameters:")
    print(f"  M={M}, N={N}, S={S}, K={K}")
    print(f"  MonteCarlo={MonteCarlo}, Max_L={Max_L}")
    print(f"GWO Parameters:")
    print(f"  SearchAgents_no={SearchAgents_no}, Max_iter={Max_iter}")
    print("="*70)
    print()
    
    np.random.seed(1)
    
    # Store convergence curves for each L
    convergence_curves = []
    
    # Main loop over different numbers of TX-SIM layers
    for ii in range(Max_L):
        L = ii + 1
        start_time = time.time()
        
        # Calculate inter-layer spacing
        d_layer_spacing_transmit = Thickness/L
        d_layer_spacing_receive = Thickness/K
        
        print(f"\nProcessing L={L}, K={K}...")
        
        # Create SIM-MIMO system
        sim_system = SIM_MIMO_GWO(M, N, S, L, K, lamda, d_element_spacing,
                                 d_layer_spacing_transmit, d_layer_spacing_receive,
                                 N_max, pathloss)
        
        NMSE_vals = np.zeros(MonteCarlo)
        Capacity_vals = np.zeros(MonteCarlo)
        
        for jj in range(MonteCarlo):
            print(f"  Monte Carlo run {jj+1}/{MonteCarlo}")
            
            # Generate HMIMO channel
            G_independent = np.sqrt(1/2) * (np.random.randn(N, M) + 
                                           1j*np.random.randn(N, M))
            
            # Calculate correlation matrix square roots
            Corr_R_sqrt = la.sqrtm(sim_system.Corr_R + 1e-8*np.eye(N))
            Corr_T_sqrt = la.sqrtm(sim_system.Corr_T + 1e-8*np.eye(M))
            
            G = np.sqrt(pathloss) * Corr_R_sqrt @ G_independent @ Corr_T_sqrt
            
            # SVD of HMIMO channel
            U, G_svd, Vh = la.svd(G, full_matrices=False)
            H_true = np.diag(G_svd[:S])
            H_true_vec = H_true.flatten('F')
            Norm_H = np.linalg.norm(H_true_vec)**2
            
            h_diag = np.diag(H_true)
            PA_WF = water_filling(Pt, Sigma2, h_diag)
            
            # Optimize phase shifts using GWO
            print("    Running GWO optimization...")
            phase_transmit, phase_receive, nmse, conv_curve = sim_system.optimize_phase_shifts_gwo(
                G, H_true, Norm_H, PA_WF, SearchAgents_no, Max_iter
            )
            
            # Store convergence curve for first Monte Carlo run
            if jj == 0:
                convergence_curves.append(conv_curve)
            
            # Calculate capacity with optimized phase shifts
            capacity, Factor, H_SIM = sim_system.calculate_capacity(
                phase_transmit, phase_receive, G, PA_WF, Sigma2
            )
            
            NMSE_vals[jj] = nmse
            Capacity_vals[jj] = capacity
            
            print(f"    NMSE: {nmse:.6f}, Capacity: {capacity:.4f} bps/Hz")
        
        NMSE_average[ii] = np.mean(NMSE_vals)
        Capacity_average[ii] = np.mean(Capacity_vals)
        Time_average[ii] = (time.time() - start_time) / MonteCarlo
        
        print(f"\n  L={L}: Average NMSE = {NMSE_average[ii]:.6f}")
        print(f"        Average Capacity = {Capacity_average[ii]:.4f} bps/Hz")
        print(f"        Average Time per MC run = {Time_average[ii]:.2f}s")
    
    # ===================== VISUALIZATION =====================
    print("\n" + "="*70)
    print("RESULTS VISUALIZATION")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: NMSE vs L
    axes[0, 0].plot(range(1, Max_L + 1), NMSE_average, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of TX-SIM Layers (L)', fontsize=12)
    axes[0, 0].set_ylabel('Average NMSE', fontsize=12)
    axes[0, 0].set_title('NMSE vs L ', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0.5, Max_L + 0.5)
    
    # Plot 2: Capacity vs L
    axes[0, 1].plot(range(1, Max_L + 1), Capacity_average, 's-', linewidth=2, markersize=8, color='green')
    axes[0, 1].set_xlabel('Number of TX-SIM Layers (L)', fontsize=12)
    axes[0, 1].set_ylabel('Average Capacity (bps/Hz)', fontsize=12)
    axes[0, 1].set_title('Capacity vs L ', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0.5, Max_L + 0.5)
    
    # Plot 3: Execution Time
    axes[0, 2].bar(range(1, Max_L + 1), Time_average, alpha=0.7)
    axes[0, 2].set_xlabel('Number of TX-SIM Layers (L)', fontsize=12)
    axes[0, 2].set_ylabel('Average Time per MC run (s)', fontsize=12)
    axes[0, 2].set_title('Execution Time vs L', fontsize=14)
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    axes[0, 2].set_xlim(0.5, Max_L + 0.5)
    
    # Plot 4: GWO Convergence Curves for different L
    for i, curve in enumerate(convergence_curves):
        axes[1, 0].plot(curve, label=f'L={i+1}', alpha=0.7)
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('NMSE', fontsize=12)
    axes[1, 0].set_title('GWO Convergence Curves', fontsize=14)
    axes[1, 0].legend(loc='upper right', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Plot 5: Performance Trade-off
    axes[1, 1].scatter(NMSE_average, Capacity_average, s=100, alpha=0.7)
    for i in range(Max_L):
        axes[1, 1].annotate(f'L={i+1}', (NMSE_average[i], Capacity_average[i]), 
                           fontsize=9, ha='center', va='bottom')
    axes[1, 1].set_xlabel('NMSE', fontsize=12)
    axes[1, 1].set_ylabel('Capacity (bps/Hz)', fontsize=12)
    axes[1, 1].set_title('Capacity vs NMSE Trade-off', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Final channel matrix (for L=Max_L)
    if len(convergence_curves) > 0:
        # Create a dummy SIM system for L=Max_L
        sim_final = SIM_MIMO_GWO(M, N, S, Max_L, K, lamda, d_element_spacing,
                                Thickness/Max_L, Thickness/K, N_max, pathloss)
        
        # Generate a sample channel
        G_independent = np.sqrt(1/2) * (np.random.randn(N, M) + 1j*np.random.randn(N, M))
        Corr_R_sqrt = la.sqrtm(sim_final.Corr_R + 1e-8*np.eye(N))
        Corr_T_sqrt = la.sqrtm(sim_final.Corr_T + 1e-8*np.eye(M))
        G = np.sqrt(pathloss) * Corr_R_sqrt @ G_independent @ Corr_T_sqrt
        
        # Create random phase shifts for visualization
        phase_transmit = np.random.randn(M, Max_L) + 1j*np.random.randn(M, Max_L)
        phase_receive = np.random.randn(N, K) + 1j*np.random.randn(N, K)
        phase_transmit = phase_transmit / np.abs(phase_transmit)
        phase_receive = phase_receive / np.abs(phase_receive)
        
        # Calculate response
        P, Q = sim_final.calculate_sim_response(phase_transmit, phase_receive)
        H_SIM = Q @ G @ P
        
        im = axes[1, 2].imshow(np.abs(H_SIM), aspect='auto', cmap='hot')
        axes[1, 2].set_xlabel('Transmit Streams', fontsize=12)
        axes[1, 2].set_ylabel('Receive Streams', fontsize=12)
        axes[1, 2].set_title(f'Channel Matrix |H_SIM| (L={Max_L})', fontsize=14)
        plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('sim_mimo_gwo_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ===================== SAVE RESULTS =====================
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    results = {
        'L_values': list(range(1, Max_L + 1)),
        'NMSE_average': NMSE_average.tolist(),
        'Capacity_average': Capacity_average.tolist(),
        'Time_average': Time_average.tolist(),
        'parameters': {
            'M': M, 'N': N, 'S': S, 'K': K,
            'MonteCarlo': MonteCarlo, 'Max_L': Max_L,
            'SearchAgents_no': SearchAgents_no, 'Max_iter': Max_iter,
            'Pt': Pt, 'Sigma2': Sigma2, 'f0': f0
        }
    }
    
    import json
    with open('sim_mimo_gwo_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    np.save('NMSE_K_10.npy', NMSE_average)
    np.save('Capacity_K_10.npy', Capacity_average)
    
    print("Results saved to:")
    print("  sim_mimo_gwo_results.json")
    print("  NMSE_K_10.npy")
    print("  Capacity_K_10.npy")
    print("  sim_mimo_gwo_results.png")
    
    # ===================== FINAL SUMMARY =====================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'L':>3} {'NMSE':<15} {'Capacity':<15} {'Time(s)':<12}")
    print("-"*70)
    
    for i in range(Max_L):
        print(f"{i+1:>3} {NMSE_average[i]:<15.6f} {Capacity_average[i]:<15.4f} "
              f"{Time_average[i]:<12.2f}")
    
    print("-"*70)
    print(f"Best NMSE: {np.min(NMSE_average):.6f} at L={np.argmin(NMSE_average)+1}")
    print(f"Best Capacity: {np.max(Capacity_average):.4f} bps/Hz at L={np.argmax(Capacity_average)+1}")
    print("="*70)

if __name__ == "__main__":
    main()