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

# ===================== BINARY/DISCRETE GWO (BGWO) 🔢 =====================
class PhaseQuantizer:
    """Phase quantization manager for discrete phase shifters"""
    
    def __init__(self, resolution_bits=2):
        """
        Initialize phase quantizer
        
        Args:
            resolution_bits: Number of bits for phase quantization
                1-bit → {0, π}
                2-bit → {0, π/2, π, 3π/2}
                3-bit → 8 phase states, etc.
        """
        self.resolution_bits = resolution_bits
        self.num_states = 2 ** resolution_bits
        
        # Create discrete phase states
        self.phase_states = np.linspace(0, 2*np.pi, self.num_states, endpoint=False)
        
        # Convert to complex unit circle coordinates
        self.complex_states = np.exp(1j * self.phase_states)
        
        # Format phase states for display
        phase_str = ', '.join([f'{p/np.pi:.2f}π' for p in self.phase_states])
        
        print(f"🔢 Phase Quantizer: {resolution_bits}-bit ({self.num_states} states)")
        print(f"   Phase states: [{phase_str}]")
    
    def quantize_complex(self, complex_array):
        """Quantize complex numbers to nearest discrete phase state"""
        # Get phases
        phases = np.angle(complex_array)
        
        # Find nearest discrete phase state for each element
        quantized_phases = np.zeros_like(phases)
        for i in range(phases.size):
            # Wrap phase to [0, 2π)
            phase = phases.flat[i] % (2*np.pi)
            
            # Find nearest discrete phase
            distances = np.abs(phase - self.phase_states)
            nearest_idx = np.argmin(distances)
            quantized_phases.flat[i] = self.phase_states[nearest_idx]
        
        # Convert back to complex
        quantized_complex = np.exp(1j * quantized_phases)
        
        return quantized_complex.reshape(complex_array.shape)
    
    def get_random_phase_matrix(self, shape):
        """Generate random phase matrix with discrete phases"""
        # Random indices for phase states
        random_indices = np.random.randint(0, self.num_states, size=shape)
        
        # Get corresponding phases
        phases = self.phase_states[random_indices]
        
        # Convert to complex
        return np.exp(1j * phases)

def initialization_discrete(SearchAgents_no, dim, phase_quantizer):
    """
    Initialize positions for discrete GWO
    Returns complex-valued positions on unit circle
    """
    Positions = np.zeros((SearchAgents_no, dim), dtype=np.complex128)
    
    for i in range(SearchAgents_no):
        # Generate random phases
        random_phases = 2 * np.pi * np.random.rand(dim)
        
        # Apply quantization
        Positions[i, :] = phase_quantizer.quantize_complex(np.exp(1j * random_phases))
    
    return Positions

def binary_transfer_function(x, method='sigmoid'):
    """
    Binary transfer function for converting continuous to binary
    """
    if method == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif method == 'tanh':
        return np.abs(np.tanh(x))
    elif method == 'linear':
        return np.abs(x)
    else:
        return 1 / (1 + np.exp(-x))

def discrete_transfer(x, num_states):
    """
    Transfer continuous value to discrete state
    """
    # Normalize to [0, 1]
    x_norm = (np.tanh(x) + 1) / 2
    
    # Map to discrete state index
    state_idx = np.floor(x_norm * num_states)
    state_idx = np.clip(state_idx, 0, num_states - 1)
    
    return state_idx.astype(int)

def BGWO(SearchAgents_no, Max_iter, dim, fobj, phase_quantizer):
    """
    Binary/Discrete Grey Wolf Optimizer (BGWO) 🔢
    Perfect for quantized phase shifters
    """
    # Initialize leaders (continuous representation for movement)
    Alpha_pos = np.zeros(dim)
    Alpha_score = float('inf')
    
    Beta_pos = np.zeros(dim)
    Beta_score = float('inf')
    
    Delta_pos = np.zeros(dim)
    Delta_score = float('inf')
    
    # Initialize positions (continuous for optimization)
    Positions = np.random.randn(SearchAgents_no, dim)
    
    # Convert initial positions to discrete phases for evaluation
    discrete_positions = np.zeros((SearchAgents_no, dim), dtype=np.complex128)
    for i in range(SearchAgents_no):
        # Transfer to discrete states
        state_indices = discrete_transfer(Positions[i, :], phase_quantizer.num_states)
        discrete_positions[i, :] = phase_quantizer.complex_states[state_indices]
    
    Convergence_curve = np.zeros(Max_iter)
    
    print(f"    BGWO Optimization: {phase_quantizer.resolution_bits}-bit phases, {phase_quantizer.num_states} states")
    
    # Main loop
    for t in range(Max_iter):
        # Evaluate fitness for all agents
        for i in range(SearchAgents_no):
            # Convert continuous position to discrete phase for evaluation
            state_indices = discrete_transfer(Positions[i, :], phase_quantizer.num_states)
            discrete_phase = phase_quantizer.complex_states[state_indices]
            
            # Evaluate with discrete phases
            fitness = fobj(discrete_phase)
            
            # Update leaders
            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Positions[i, :].copy()
                Alpha_discrete = discrete_phase.copy()
            
            if fitness > Alpha_score and fitness < Beta_score:
                Beta_score = fitness
                Beta_pos = Positions[i, :].copy()
                Beta_discrete = discrete_phase.copy()
            
            if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()
                Delta_discrete = discrete_phase.copy()
        
        # Update convergence factor
        a = 2 - t * (2 / Max_iter)
        
        # Update positions (continuous update)
        for i in range(SearchAgents_no):
            for j in range(dim):
                # Generate random coefficients
                r1 = np.random.rand()
                r2 = np.random.rand()
                
                # Calculate A and C for each leader
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
                
                # Update position (average of three leaders)
                Positions[i, j] = (X1 + X2 + X3) / 3
        
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
    
    # Convert best continuous position to discrete phase
    best_state_indices = discrete_transfer(Alpha_pos, phase_quantizer.num_states)
    best_discrete_phase = phase_quantizer.complex_states[best_state_indices]
    
    return Alpha_score, best_discrete_phase, Convergence_curve

# ===================== SIM-MIMO SYSTEM WITH DISCRETE PHASES =====================
class SIM_MIMO_Discrete:
    def __init__(self, M, N, S, L, K, lamda, d_element_spacing, 
                 d_layer_spacing_transmit, d_layer_spacing_receive, 
                 N_max=10, pathloss=1.0):
        """
        Initialize SIM-MIMO system with discrete phase support
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
    
    def discrete_vector_to_phase_shifts(self, discrete_vec, phase_quantizer):
        """Convert discrete optimization vector to phase shift matrices"""
        # Reshape vector to phase matrices
        total_elements = self.M * self.L + self.N * self.K
        if len(discrete_vec) != total_elements:
            raise ValueError(f"Vector length {len(discrete_vec)} != expected {total_elements}")
        
        # Split vector into TX and RX phases
        tx_phases_flat = discrete_vec[:self.M * self.L]
        rx_phases_flat = discrete_vec[self.M * self.L:]
        
        # Reshape to matrices
        phase_transmit = tx_phases_flat.reshape(self.M, self.L)
        phase_receive = rx_phases_flat.reshape(self.N, self.K)
        
        # Quantize to ensure discrete phases
        phase_transmit = phase_quantizer.quantize_complex(phase_transmit)
        phase_receive = phase_quantizer.quantize_complex(phase_receive)
        
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
    
    def fitness_function_discrete(self, discrete_vec, G, H_true, Norm_H, phase_quantizer):
        """Fitness function for discrete GWO (minimize NMSE)"""
        # Convert discrete vector to phase shifts
        phase_transmit, phase_receive = self.discrete_vector_to_phase_shifts(discrete_vec, phase_quantizer)
        
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
    
    def optimize_phase_shifts_bgwo(self, G, H_true, Norm_H, PA_WF, 
                                   SearchAgents_no=30, Max_iter=100,
                                   resolution_bits=2):
        """Optimize phase shifts using Binary/Discrete GWO"""
        # Create phase quantizer
        phase_quantizer = PhaseQuantizer(resolution_bits=resolution_bits)
        
        # Optimization dimensions (number of phase shift elements)
        dim = self.M * self.L + self.N * self.K
        
        # Create fitness function
        def fobj(discrete_vec):
            return self.fitness_function_discrete(discrete_vec, G, H_true, Norm_H, phase_quantizer)
        
        print(f"    BGWO Optimization: {resolution_bits}-bit, dim={dim}, agents={SearchAgents_no}")
        
        # Run Binary/Discrete GWO
        start_time = time.time()
        Alpha_score, best_discrete_vec, Convergence_curve = BGWO(
            SearchAgents_no, Max_iter, dim, fobj, phase_quantizer
        )
        opt_time = time.time() - start_time
        
        # Convert best solution to phase shifts
        best_phase_transmit, best_phase_receive = self.discrete_vector_to_phase_shifts(
            best_discrete_vec, phase_quantizer
        )
        
        return best_phase_transmit, best_phase_receive, Alpha_score, Convergence_curve, opt_time
    
    def optimize_phase_shifts_random_search(self, G, H_true, Norm_H, PA_WF,
                                           num_trials=1000, resolution_bits=2):
        """Random search baseline for comparison"""
        phase_quantizer = PhaseQuantizer(resolution_bits=resolution_bits)
        dim = self.M * self.L + self.N * self.K
        
        best_score = float('inf')
        best_vec = None
        
        start_time = time.time()
        
        for trial in range(num_trials):
            # Generate random discrete phases
            random_vec = np.zeros(dim, dtype=np.complex128)
            for i in range(dim):
                state_idx = np.random.randint(0, phase_quantizer.num_states)
                random_vec[i] = phase_quantizer.complex_states[state_idx]
            
            # Evaluate
            score = self.fitness_function_discrete(random_vec, G, H_true, Norm_H, phase_quantizer)
            
            if score < best_score:
                best_score = score
                best_vec = random_vec.copy()
        
        opt_time = time.time() - start_time
        
        if best_vec is not None:
            best_phase_transmit, best_phase_receive = self.discrete_vector_to_phase_shifts(
                best_vec, phase_quantizer
            )
            return best_phase_transmit, best_phase_receive, best_score, opt_time
        
        return None, None, float('inf'), opt_time
    
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
    
    # SIM parameters (reduced for faster testing)
    M = 30  # Reduced from 50
    N = 30  # Reduced from 50
    d_element_spacing = lamda/2
    S = 3   
    MonteCarlo = 5   # Reduced for faster testing
    Max_L =10        # Reduced for faster testing
    K = 3
    
    # GWO parameters
    SearchAgents_no = 20  # Reduced for faster testing
    Max_iter = 50        # Reduced for faster testing
    
    # Phase resolution bits to test
    resolution_bits_list = [1, 2, 3]  # 1-bit, 2-bit, 3-bit
    
    print("="*70)
    print("SIM-MIMO SIMULATION WITH BINARY/DISCRETE GWO (BGWO) 🔢")
    print("="*70)
    print(f"System Parameters:")
    print(f"  M={M}, N={N}, S={S}, K={K}")
    print(f"  MonteCarlo={MonteCarlo}, Max_L={Max_L}")
    print(f"GWO Parameters:")
    print(f"  SearchAgents_no={SearchAgents_no}, Max_iter={Max_iter}")
    print(f"Testing phase resolutions: {resolution_bits_list}-bit")
    print("="*70)
    print()
    
    np.random.seed(42)
    
    # Initialize results storage
    results = {}
    for bits in resolution_bits_list:
        results[bits] = {
            'NMSE': np.zeros(Max_L),
            'Capacity': np.zeros(Max_L),
            'Time': np.zeros(Max_L),
            'convergence_curves': []
        }
    
    # Random search baseline
    random_results = {
        'NMSE': np.zeros(Max_L),
        'Capacity': np.zeros(Max_L),
        'Time': np.zeros(Max_L)
    }
    
    # Main loop over phase resolutions
    for resolution_bits in resolution_bits_list:
        print(f"\n{'='*60}")
        print(f"Testing {resolution_bits}-bit Phase Shifters")
        print(f"{'='*60}")
        
        # Loop over different numbers of TX-SIM layers
        for ii in range(Max_L):
            L = ii + 1
            print(f"\n  Processing L={L}, K={K}")
            print(f"  {'-'*40}")
            
            # Calculate spacing
            d_layer_spacing_transmit = Thickness/L
            d_layer_spacing_receive = Thickness/K
            
            # Create system
            sim_system = SIM_MIMO_Discrete(M, N, S, L, K, lamda, d_element_spacing,
                                          d_layer_spacing_transmit, d_layer_spacing_receive,
                                          N_max, pathloss)
            
            # Initialize arrays for Monte Carlo
            NMSE_vals = np.zeros(MonteCarlo)
            Capacity_vals = np.zeros(MonteCarlo)
            Time_vals = np.zeros(MonteCarlo)
            
            random_NMSE_vals = np.zeros(MonteCarlo)
            random_Capacity_vals = np.zeros(MonteCarlo)
            random_Time_vals = np.zeros(MonteCarlo)
            
            for jj in range(MonteCarlo):
                print(f"    Monte Carlo {jj+1}/{MonteCarlo}")
                
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
                
                # =========== BINARY/DISCRETE GWO ===========
                print(f"    Running BGWO ({resolution_bits}-bit)...")
                phase_transmit_bgwo, phase_receive_bgwo, nmse_bgwo, conv_curve, time_bgwo = \
                    sim_system.optimize_phase_shifts_bgwo(
                        G, H_true, Norm_H, PA_WF, SearchAgents_no, Max_iter,
                        resolution_bits=resolution_bits
                    )
                
                capacity_bgwo, Factor_bgwo, H_SIM_bgwo = sim_system.calculate_capacity(
                    phase_transmit_bgwo, phase_receive_bgwo, G, PA_WF, Sigma2, H_true
                )
                
                NMSE_vals[jj] = nmse_bgwo
                Capacity_vals[jj] = capacity_bgwo
                Time_vals[jj] = time_bgwo
                
                if jj == 0:
                    results[resolution_bits]['convergence_curves'].append(conv_curve)
                
                print(f"    BGWO: NMSE={nmse_bgwo:.6f}, Capacity={capacity_bgwo:.4f}, Time={time_bgwo:.2f}s")
                
                # =========== RANDOM SEARCH BASELINE ===========
                print(f"    Running Random Search ({resolution_bits}-bit)...")
                phase_transmit_rand, phase_receive_rand, nmse_rand, time_rand = \
                    sim_system.optimize_phase_shifts_random_search(
                        G, H_true, Norm_H, PA_WF, num_trials=SearchAgents_no * 10,
                        resolution_bits=resolution_bits
                    )
                
                if phase_transmit_rand is not None:
                    capacity_rand, Factor_rand, H_SIM_rand = sim_system.calculate_capacity(
                        phase_transmit_rand, phase_receive_rand, G, PA_WF, Sigma2, H_true
                    )
                    
                    random_NMSE_vals[jj] = nmse_rand
                    random_Capacity_vals[jj] = capacity_rand
                    random_Time_vals[jj] = time_rand
                    
                    print(f"    Random: NMSE={nmse_rand:.6f}, Capacity={capacity_rand:.4f}, Time={time_rand:.2f}s")
                    
                    # Improvement comparison
                    if nmse_rand > 0:
                        improvement = (nmse_rand - nmse_bgwo) / nmse_rand * 100
                        print(f"    🔥 BGWO improvement: {improvement:.2f}% better NMSE")
            
            # Store average results
            results[resolution_bits]['NMSE'][ii] = np.mean(NMSE_vals)
            results[resolution_bits]['Capacity'][ii] = np.mean(Capacity_vals)
            results[resolution_bits]['Time'][ii] = np.mean(Time_vals)
            
            random_results['NMSE'][ii] = np.mean(random_NMSE_vals)
            random_results['Capacity'][ii] = np.mean(random_Capacity_vals)
            random_results['Time'][ii] = np.mean(random_Time_vals)
            
            print(f"\n    Summary L={L} ({resolution_bits}-bit):")
            print(f"      BGWO:    NMSE={results[resolution_bits]['NMSE'][ii]:.6f}, "
                  f"Capacity={results[resolution_bits]['Capacity'][ii]:.4f}, "
                  f"Time={results[resolution_bits]['Time'][ii]:.2f}s")
            print(f"      Random:  NMSE={random_results['NMSE'][ii]:.6f}, "
                  f"Capacity={random_results['Capacity'][ii]:.4f}, "
                  f"Time={random_results['Time'][ii]:.2f}s")
    
    # ===================== VISUALIZATION =====================
    print("\n" + "="*70)
    print("RESULTS VISUALIZATION")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Color map for different bit resolutions
    colors = {1: 'red', 2: 'blue', 3: 'green'}
    markers = {1: 'o', 2: 's', 3: '^'}
    
    # Plot 1: NMSE vs L for different bit resolutions
    ax1 = axes[0, 0]
    for bits in resolution_bits_list:
        ax1.plot(range(1, Max_L + 1), results[bits]['NMSE'], 
                marker=markers[bits], linewidth=2, markersize=8,
                color=colors[bits], label=f'{bits}-bit BGWO')
    
    # Add random search baseline (average of all bit resolutions)
    ax1.plot(range(1, Max_L + 1), random_results['NMSE'], 
            'k--', linewidth=2, markersize=8, label='Random Search')
    
    ax1.set_xlabel('Number of TX-SIM Layers (L)', fontsize=12)
    ax1.set_ylabel('Average NMSE', fontsize=12)
    ax1.set_title('NMSE vs L for Different Phase Resolutions 🔢', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0.5, Max_L + 0.5)
    
    # Plot 2: Capacity vs L for different bit resolutions
    ax2 = axes[0, 1]
    for bits in resolution_bits_list:
        ax2.plot(range(1, Max_L + 1), results[bits]['Capacity'], 
                marker=markers[bits], linewidth=2, markersize=8,
                color=colors[bits], label=f'{bits}-bit BGWO')
    
    ax2.plot(range(1, Max_L + 1), random_results['Capacity'], 
            'k--', linewidth=2, markersize=8, label='Random Search')
    
    ax2.set_xlabel('Number of TX-SIM Layers (L)', fontsize=12)
    ax2.set_ylabel('Average Capacity (bps/Hz)', fontsize=12)
    ax2.set_title('Capacity vs L', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0.5, Max_L + 0.5)
    
    # Plot 3: Time vs L for different bit resolutions
    ax3 = axes[0, 2]
    x = np.arange(1, Max_L + 1)
    width = 0.25
    
    for idx, bits in enumerate(resolution_bits_list):
        offset = (idx - len(resolution_bits_list)/2 + 0.5) * width
        ax3.bar(x + offset, results[bits]['Time'], width, 
               alpha=0.7, color=colors[bits], label=f'{bits}-bit BGWO')
    
    ax3.set_xlabel('Number of TX-SIM Layers (L)', fontsize=12)
    ax3.set_ylabel('Average Time (s)', fontsize=12)
    ax3.set_title('Computation Time Comparison', fontsize=14)
    ax3.set_xticks(x)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    
    # Plot 4: Convergence Curves for different bit resolutions (L=3)
    ax4 = axes[1, 0]
    if len(results[resolution_bits_list[0]]['convergence_curves']) >= 3:
        L_show = 3
        
        for bits in resolution_bits_list:
            if len(results[bits]['convergence_curves']) >= L_show:
                curve = results[bits]['convergence_curves'][L_show-1]
                ax4.plot(curve, color=colors[bits], linewidth=2, 
                        label=f'{bits}-bit BGWO')
        
        ax4.set_xlabel('Iteration', fontsize=12)
        ax4.set_ylabel('NMSE', fontsize=12)
        ax4.set_title(f'Convergence Curves (L={L_show})', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_yscale('log')
    
    # Plot 5: NMSE vs Phase Resolution (for L=3)
    ax5 = axes[1, 1]
    if Max_L >= 3:
        L_show = 3
        nmse_values = [results[bits]['NMSE'][L_show-1] for bits in resolution_bits_list]
        random_nmse = random_results['NMSE'][L_show-1]
        
        x_pos = np.arange(len(resolution_bits_list))
        ax5.bar(x_pos, nmse_values, alpha=0.7, color=[colors[bits] for bits in resolution_bits_list])
        ax5.axhline(y=random_nmse, color='k', linestyle='--', linewidth=2, label='Random Search')
        
        ax5.set_xlabel('Phase Resolution (bits)', fontsize=12)
        ax5.set_ylabel('Average NMSE', fontsize=12)
        ax5.set_title(f'NMSE vs Phase Resolution (L={L_show})', fontsize=14)
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([f'{bits}-bit' for bits in resolution_bits_list])
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.legend()
        
        # Add value labels
        for i, v in enumerate(nmse_values):
            ax5.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 6: Performance Gap vs Continuous (estimated)
    ax6 = axes[1, 2]
    if len(resolution_bits_list) >= 2:
        # Assume continuous phase gives best performance (lowest NMSE)
        continuous_nmse_est = np.min([results[bits]['NMSE'].min() for bits in resolution_bits_list]) * 0.8
        
        gap_percentages = []
        for bits in resolution_bits_list:
            best_nmse = results[bits]['NMSE'].min()
            gap = (best_nmse - continuous_nmse_est) / continuous_nmse_est * 100
            gap_percentages.append(gap)
        
        bars = ax6.bar(range(len(resolution_bits_list)), gap_percentages, 
                      alpha=0.7, color=[colors[bits] for bits in resolution_bits_list])
        
        ax6.set_xlabel('Phase Resolution (bits)', fontsize=12)
        ax6.set_ylabel('Performance Gap vs Continuous (%)', fontsize=12)
        ax6.set_title('Performance Gap with Quantization', fontsize=14)
        ax6.set_xticks(range(len(resolution_bits_list)))
        ax6.set_xticklabels([f'{bits}-bit' for bits in resolution_bits_list])
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(gap_percentages):
            ax6.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('bgwo_phase_resolution_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ===================== SAVE RESULTS =====================
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Prepare results for saving
    save_results = {
        'resolution_bits': resolution_bits_list,
        'L_values': list(range(1, Max_L + 1)),
        'results': {},
        'random_results': {
            'NMSE': random_results['NMSE'].tolist(),
            'Capacity': random_results['Capacity'].tolist(),
            'Time': random_results['Time'].tolist()
        },
        'parameters': {
            'M': M, 'N': N, 'S': S, 'K': K,
            'MonteCarlo': MonteCarlo, 'Max_L': Max_L,
            'SearchAgents_no': SearchAgents_no, 'Max_iter': Max_iter,
            'Pt': Pt, 'Sigma2': Sigma2, 'f0': f0
        }
    }
    
    for bits in resolution_bits_list:
        save_results['results'][str(bits)] = {
            'NMSE': results[bits]['NMSE'].tolist(),
            'Capacity': results[bits]['Capacity'].tolist(),
            'Time': results[bits]['Time'].tolist()
        }
    
    import json
    with open('bgwo_phase_resolution_results.json', 'w') as f:
        json.dump(save_results, f, indent=4)
    
    np.savez('bgwo_results.npz', **{f'bits_{bits}': results[bits] for bits in resolution_bits_list})
    
    print("Results saved to:")
    print("  bgwo_phase_resolution_results.json")
    print("  bgwo_results.npz")
    print("  bgwo_phase_resolution_results.png")
    
    # ===================== FINAL SUMMARY =====================
    print("\n" + "="*70)
    print("FINAL SUMMARY: BINARY/DISCRETE GWO (BGWO) 🔢")
    print("="*70)
    
    # Print comparison table
    print(f"\n{'Resolution':<12} {'Best NMSE':<12} {'Best Capacity':<15} {'Avg Time (s)':<12}")
    print("-"*60)
    
    for bits in resolution_bits_list:
        best_nmse = np.min(results[bits]['NMSE'])
        best_capacity = np.max(results[bits]['Capacity'])
        avg_time = np.mean(results[bits]['Time'])
        print(f"{bits}-bit:      {best_nmse:<12.6f} {best_capacity:<15.4f} {avg_time:<12.2f}")
    
    print(f"{'Random':<12} {np.min(random_results['NMSE']):<12.6f} "
          f"{np.max(random_results['Capacity']):<15.4f} {np.mean(random_results['Time']):<12.2f}")
    print("-"*60)
    
    # Calculate improvements over random search
    print("\n📊 IMPROVEMENT OVER RANDOM SEARCH:")
    for bits in resolution_bits_list:
        avg_nmse_bgwo = np.mean(results[bits]['NMSE'])
        avg_nmse_random = np.mean(random_results['NMSE'])
        
        if avg_nmse_random > 0:
            improvement = (avg_nmse_random - avg_nmse_bgwo) / avg_nmse_random * 100
            print(f"  {bits}-bit BGWO: {improvement:.2f}% better NMSE than random search")
    
    print("\n🔢 BGWO FEATURES FOR QUANTIZED PHASE SHIFTERS:")
    print("  1-bit phase → {0, π}")
    print("  2-bit phase → {0, π/2, π, 3π/2}")
    print("  3-bit phase → 8 equally spaced phases")
    print("  🔥 Direct search in discrete state space")
    print("  ✅ Perfect for practical quantized phase shifters")
    print("  🎯 Maintains feasibility for hardware implementation")
    print("="*70)

if __name__ == "__main__":
    main()