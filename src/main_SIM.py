import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import sinc
import scipy.linalg as la

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
    
    # Sort channels in ascending order of |a|^2 (or descending order of 1/|a|^2)
    # The MATLAB code seems to assume a is already sorted appropriately
    # Based on the algorithm, a should be in descending order of |a|^2
    
    P_sum = 0
    ii = 0
    
    for ii in range(n - 1):
        # Calculate the sum power when all (1:ii+1) channels are filled
        # Note: MATLAB uses 1-based indexing, Python uses 0-based
        P_sum += (ii + 1) * (sigma2 / (a[ii + 1]**2) - sigma2 / (a[ii]**2))
        
        if P_sum >= P_total:
            break
    
    # Adjust ii for MATLAB's 1-based indexing logic
    ii += 1  # Now ii corresponds to MATLAB's ii
    
    if P_sum >= P_total:
        # Subtract the excess power for all (1:ii) channels
        PA_WF = sigma2 / (a[ii]**2) - sigma2 / (a[:ii]**2) - (P_sum - P_total) / ii
        
        # Match the dimension
        PA_WF_full = np.zeros(n)
        PA_WF_full[:ii] = PA_WF
        return PA_WF_full
    else:
        # Add the extra power for all (1:ii+1) channels
        PA_WF = sigma2 / (a[ii]**2) - sigma2 / (a[:ii+1]**2) + (P_total - P_sum) / (ii + 1)
        return PA_WF

# Parameters setup
Thickness = 0.05  # Thickness of TX-SIM and RX-SIM
Pt = 10**(20/10)  # Transmit power
Sigma2 = 10**(-110/10)  # Average noise power at the receiver
c = 3e8  # Speed of light
f0 = 28e9  # Radio frequency
lamda = c/f0  # Wavelength
N_max = 10  # Number of meta-atoms on each row

# Pathloss calculation
PL = -20*np.log10(4*np.pi/lamda) - 35*np.log10(250)
pathloss = 10**(PL/10)

M = 100  # Number of meta-atoms on each layer of TX-SIM
N = 100  # Number of meta-atoms on each layer of RX-SIM
d_element_spacing = lamda/2  # Element spacing
S = 4  # Number of data streams
MonteCarlo = 10  # Number of independent experiments
Max_L = 10  # The maximum number of metasurface layers in TX-SIM
K = 1  # CHANGED: RX-SIM has only 1 layer (changed from 10 to 1)

# Initialize arrays
NMSE_average = np.zeros(Max_L)
Capacity_average = np.zeros(Max_L)

# Main loop over different numbers of TX-SIM layers
for ii in range(Max_L):
    L = ii + 1  # The number of metasurface layers in TX-SIM
    start_time = time.time()
    
    # Calculate inter-layer spacing
    d_layer_spacing_transmit = Thickness/L
    d_layer_spacing_receive = Thickness/K  # Will be Thickness/1 = Thickness
    
    print(f"Simulating L={L}, K={K}...")
    
    # Initialize matrices for TX-SIM
    W_T = np.zeros((M, M), dtype=np.complex128)
    Corr_T = np.zeros((M, M), dtype=np.complex128)
    
    # Calculate inter-layer transmission coefficient matrix W_T and channel correlation matrix Corr_T
    for mm1 in range(M):
        m_z = (mm1 // N_max) + 1  # Eq. (3)
        m_x = (mm1 % N_max) + 1   # Eq. (3)
        
        for mm2 in range(M):
            n_z = (mm2 // N_max) + 1
            n_x = (mm2 % N_max) + 1
            
            d_temp = np.sqrt((m_x - n_x)**2 + (m_z - n_z)**2) * d_element_spacing  # Eq. (1)
            d_temp2 = np.sqrt(d_layer_spacing_transmit**2 + d_temp**2)  # Eq. (5)
            
            W_T[mm2, mm1] = (lamda/(4*np.pi*d_temp2) * 
                            np.exp(-1j*2*np.pi*d_temp2/lamda))  # old model
            
            Corr_T[mm2, mm1] = sinc(2*d_temp/lamda)  # Eq. (14)
    
    # Initialize matrices for RX-SIM
    U_R = np.zeros((N, N), dtype=np.complex128)
    Corr_R = np.zeros((N, N), dtype=np.complex128)
    
    # Calculate inter-layer transmission coefficient matrix U_R and channel correlation matrix Corr_R
    for nn1 in range(N):
        m_z = (nn1 // N_max) + 1  # Eq. (4)
        m_x = (nn1 % N_max) + 1   # Eq. (4)
        
        for nn2 in range(N):
            n_z = (nn2 // N_max) + 1
            n_x = (nn2 % N_max) + 1
            
            d_temp = np.sqrt((m_x - n_x)**2 + (m_z - n_z)**2) * d_element_spacing  # Eq. (2)
            d_temp2 = np.sqrt(d_layer_spacing_receive**2 + d_temp**2)  # Eq. (6)
            
            U_R[nn2, nn1] = (lamda/(4*np.pi*d_temp2) * 
                            np.exp(-1j*2*np.pi*d_temp2/lamda))  # old model
            
            Corr_R[nn2, nn1] = sinc(2*d_temp/lamda)  # Eq. (15)
    
    # The channel from transmitter to the first layer of TX-SIM
    W_T_1 = np.zeros((M, S), dtype=np.complex128)
    for mm in range(M):
        m_z = (mm // N_max) + 1
        m_x = (mm % N_max) + 1
        
        for nn in range(S):
            d_transmit = np.sqrt(
                d_layer_spacing_transmit**2 + 
                ((m_x - (1 + N_max)/2) * d_element_spacing)**2 +
                ((m_z - (1 + N_max)/2) * d_element_spacing - 
                 (nn - (1 + S)/2) * lamda/2)**2
            )  # Eq. (7)
            
            W_T_1[mm, nn] = (lamda/(4*np.pi*d_transmit) * 
                            np.exp(-1j*2*np.pi*d_transmit/lamda))
    
    # The channel from the last layer of RX-SIM to the receiver
    U_R_1 = np.zeros((S, N), dtype=np.complex128)
    for mm in range(N):
        m_z = (mm // N_max) + 1
        m_x = (mm % N_max) + 1
        
        for nn in range(S):
            d_receive = np.sqrt(
                d_layer_spacing_receive**2 +
                ((m_x - (1 + N_max)/2) * d_element_spacing)**2 +
                ((m_z - (1 + N_max)/2) * d_element_spacing - 
                 (nn - (1 + S)/2) * lamda/2)**2
            )  # Eq. (8)
            
            U_R_1[nn, mm] = (lamda/(4*np.pi*d_receive) * 
                            np.exp(-1j*2*np.pi*d_receive/lamda))
    
    # Set random seed for reproducibility
    np.random.seed(1)
    
    NMSE = np.zeros(MonteCarlo)
    Capacity = np.zeros(MonteCarlo)
    
    for jj in range(MonteCarlo):
        print(f"  Monte Carlo run {jj+1}/{MonteCarlo}")
        
        # Generate HMIMO channel
        G_independent = np.sqrt(1/2) * (np.random.randn(N, M) + 
                                       1j*np.random.randn(N, M))
        
        # Calculate correlation matrices square roots
        Corr_R_sqrt = la.sqrtm(Corr_R)
        Corr_T_sqrt = la.sqrtm(Corr_T)
        
        G = np.sqrt(pathloss) * Corr_R_sqrt @ G_independent @ Corr_T_sqrt  # HMIMO channel
        
        # SVD of HMIMO channel
        U, G_svd, Vh = la.svd(G, full_matrices=False)
        H_true = np.diag(G_svd[:S])  # Target channel
        H_true_vec = H_true.flatten('F')
        Norm_H = np.linalg.norm(H_true_vec)**2  # Norm of the target end-to-end channel
        
        h_diag = np.diag(H_true)
        
        # Power allocation using water-filling algorithm
        PA_WF = water_filling(Pt, Sigma2, h_diag)
        
        # Random initialization
        Num_initialization = max(L, K) * 10
        Error_old_set = np.zeros(Num_initialization)
        phase_transmit_set = np.zeros((M, L, Num_initialization), dtype=np.complex128)
        phase_receive_set = np.zeros((N, K, Num_initialization), dtype=np.complex128)
        
        for tt in range(Num_initialization):
            phase_transmit = np.random.randn(M, L) + 1j*np.random.randn(M, L)
            phase_transmit = phase_transmit / np.abs(phase_transmit)  # TX-SIM phase shifts
            
            phase_receive = np.random.randn(N, K) + 1j*np.random.randn(N, K)
            phase_receive = phase_receive / np.abs(phase_receive)  # RX-SIM phase shifts
            
            # Calculate TX-SIM response
            P = np.diag(phase_transmit[:, 0]) @ W_T_1
            for l in range(L - 1):
                P = np.diag(phase_transmit[:, l + 1]) @ W_T @ P
            
            # Calculate RX-SIM response (simplified since K=1)
            Q = U_R_1 @ np.diag(phase_receive[:, 0])
            # No loop needed for K=1
            
            H_SIM = Q @ G @ P  # Practical SIM-aided end-to-end channel
            H_SIM_vec = H_SIM.flatten('F')
            
            # Compensation factor
            Factor = np.linalg.pinv(H_SIM_vec.reshape(-1, 1).T @ H_SIM_vec.reshape(-1, 1)) @ \
                    H_SIM_vec.reshape(-1, 1).T @ H_true_vec.reshape(-1, 1)
            Factor = Factor[0, 0]
            
            Error_old_set[tt] = np.linalg.norm(Factor * H_SIM_vec - H_true_vec)**2 / Norm_H
            
            phase_transmit_set[:, :, tt] = phase_transmit
            phase_receive_set[:, :, tt] = phase_receive
        
        # Select the best initialization
        d_max = np.argmin(Error_old_set)
        Error_old = Error_old_set[d_max]
        phase_transmit = phase_transmit_set[:, :, d_max]
        phase_phase_transmit = np.angle(phase_transmit)
        phase_receive = phase_receive_set[:, :, d_max]
        phase_phase_receive = np.angle(phase_receive)
        
        # Calculate initial SIM responses
        P = np.diag(phase_transmit[:, 0]) @ W_T_1
        for l in range(L - 1):
            P = np.diag(phase_transmit[:, l + 1]) @ W_T @ P
        
        Q = U_R_1 @ np.diag(phase_receive[:, 0])
        # No loop needed for K=1
        
        H_SIM = Q @ G @ P
        H_SIM_vec = H_SIM.flatten('F')
        
        Factor = np.linalg.pinv(H_SIM_vec.reshape(-1, 1).T @ H_SIM_vec.reshape(-1, 1)) @ \
                H_SIM_vec.reshape(-1, 1).T @ H_true_vec.reshape(-1, 1)
        Factor = Factor[0, 0]
        
        # Gradient descent optimization
        step = 0.1  # learning rate
        Error_new = 10000  # A preset value as large as possible
        iteration = 0
        
        while abs(Error_new - Error_old) >= Error_old * 0.001:
            iteration += 1
            
            # Calculate partial derivatives for TX-SIM phase shifts (Eq. 23)
            Derivative_transmit_phase_shift = np.zeros((M, L))
            Temp1 = np.zeros(S)
            
            for ll in range(L):
                for mm in range(M):
                    X_left = W_T_1
                    for ll_left in range(ll):
                        X_left = W_T @ np.diag(phase_transmit[:, ll_left]) @ X_left
                    
                    X_right = Q @ G
                    for ll_right in range(L - ll - 1):
                        X_right = X_right @ np.diag(phase_transmit[:, L - 1 - ll_right]) @ W_T
                    
                    for ss1 in range(S):
                        temp1 = X_right[:, mm] * X_left[mm, ss1]
                        Temp1[ss1] = 2 * np.imag(
                            (Factor * phase_transmit[mm, ll] * temp1).conj().T @ 
                            (Factor * H_SIM[:, ss1] - H_true[:, ss1])
                        )
                    
                    Derivative_transmit_phase_shift[mm, ll] = np.sum(Temp1)
            
            # Calculate partial derivatives for RX-SIM phase shifts (Eq. 24)
            Derivative_receive_phase_shift = np.zeros((N, K))
            Temp2 = np.zeros(S)
            
            for kk in range(K):
                for nn in range(N):
                    Y_left = U_R_1
                    for kk_left in range(kk):
                        Y_left = Y_left @ np.diag(phase_receive[:, kk_left]) @ U_R
                    
                    Y_right = G @ P
                    for kk_right in range(K - kk - 1):
                        Y_right = U_R @ np.diag(phase_receive[:, K - 1 - kk_right]) @ Y_right
                    
                    for ss1 in range(S):
                        Y = Y_left[ss1, nn] * Y_right[nn, :]
                        Temp2[ss1] = 2 * np.imag(
                            (Factor * H_SIM[ss1, :] - H_true[ss1, :]) @ 
                            (Factor * phase_receive[nn, kk] * Y).conj().T
                        )
                    
                    Derivative_receive_phase_shift[nn, kk] = np.sum(Temp2)
            
            # Normalize derivatives and update phase shifts
            Derivative_transmit_phase_shift = (np.pi * Derivative_transmit_phase_shift / 
                                             np.max(np.abs(Derivative_transmit_phase_shift)))  # Eq. (27)
            phase_phase_transmit = phase_phase_transmit - step * Derivative_transmit_phase_shift
            phase_transmit = np.exp(1j * phase_phase_transmit)
            
            Derivative_receive_phase_shift = (np.pi * Derivative_receive_phase_shift / 
                                            np.max(np.abs(Derivative_receive_phase_shift)))  # Eq. (28)
            phase_phase_receive = phase_phase_receive - step * Derivative_receive_phase_shift
            phase_receive = np.exp(1j * phase_phase_receive)
            
            step = step * 0.5  # Update learning rate
            
            # Recalculate SIM responses
            P = np.diag(phase_transmit[:, 0]) @ W_T_1
            for l in range(L - 1):
                P = np.diag(phase_transmit[:, l + 1]) @ W_T @ P
            
            Q = U_R_1 @ np.diag(phase_receive[:, 0])
            # No loop needed for K=1
            
            H_SIM = Q @ G @ P
            H_SIM_vec = H_SIM.flatten('F')
            
            Factor = np.linalg.pinv(H_SIM_vec.reshape(-1, 1).T @ H_SIM_vec.reshape(-1, 1)) @ \
                    H_SIM_vec.reshape(-1, 1).T @ H_true_vec.reshape(-1, 1)
            Factor = Factor[0, 0]
            
            Error_old = Error_new
            Error_new = np.linalg.norm(Factor * H_SIM - H_true)**2 / Norm_H
            
            if iteration % 20 == 0:
                print(f"    Iteration {iteration}: NMSE = {Error_new:.6f}")
        
        NMSE[jj] = Error_new
        
        # Calculate capacity
        C_single_stream = np.zeros(S)
        for pp in range(S):
            interference = np.sum(np.abs(Factor * H_SIM[pp, :])**2 * PA_WF) - \
                          PA_WF[pp] * np.abs(Factor * H_SIM[pp, pp])**2
            C_single_stream[pp] = np.log2(1 + 
                PA_WF[pp] * np.abs(Factor * H_SIM[pp, pp])**2 / 
                (Sigma2 + interference)
            )
        
        Capacity[jj] = np.sum(C_single_stream)
        print(f"  Run {jj+1}: NMSE = {NMSE[jj]:.6f}, Capacity = {Capacity[jj]:.4f} bps/Hz")
    
    NMSE_average[ii] = np.mean(NMSE)
    Capacity_average[ii] = np.mean(Capacity)
    
    print(f"L={L}, K={K} completed: Average NMSE = {NMSE_average[ii]:.6f}, Average Capacity = {Capacity_average[ii]:.4f} bps/Hz")
    print(f"Time: {time.time() - start_time:.2f}s\n")

# Visualization
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(np.abs(Q @ G @ P), aspect='auto', cmap='hot')
plt.colorbar()
plt.title(f'Final |Q*G*P| (K={K})')
plt.xlabel('Transmit Streams')
plt.ylabel('Receive Streams')

plt.subplot(2, 2, 2)
plt.plot(range(1, Max_L + 1), NMSE_average, 'o-', linewidth=2, markersize=8)
plt.xlabel('Number of TX-SIM layers (L)')
plt.ylabel('Average NMSE')
plt.title(f'NMSE vs L (K={K})')
plt.grid(True)
plt.xlim(0.5, Max_L + 0.5)

plt.subplot(2, 2, 3)
plt.plot(range(1, Max_L + 1), Capacity_average, 's-', linewidth=2, markersize=8)
plt.xlabel('Number of TX-SIM layers (L)')
plt.ylabel('Average Capacity (bps/Hz)')
plt.title(f'Capacity vs L (K={K})')
plt.grid(True)
plt.xlim(0.5, Max_L + 0.5)

# Add comparison plot
plt.subplot(2, 2, 4)
x = range(1, Max_L + 1)
width = 0.35
plt.bar([i - width/2 for i in x], NMSE_average / np.max(NMSE_average), 
        width=width, label='Normalized NMSE', alpha=0.7)
plt.bar([i + width/2 for i in x], Capacity_average / np.max(Capacity_average), 
        width=width, label='Normalized Capacity', alpha=0.7)
plt.xlabel('Number of TX-SIM layers (L)')
plt.ylabel('Normalized Performance')
plt.title(f'Normalized Performance Comparison (K={K})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0.5, Max_L + 0.5)

plt.tight_layout()
plt.show()

# Save results
np.save(f'NMSE_K_{K}.npy', NMSE_average)
np.save(f'Capacity_K_{K}.npy', Capacity_average)

print("="*60)
print("SIMULATION COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Configuration: K = {K} (RX-SIM layers)")
print(f"TX-SIM layers tested: L = 1 to {Max_L}")
print(f"\nSummary Results:")
for i in range(Max_L):
    print(f"  L={i+1}: NMSE = {NMSE_average[i]:.6f}, Capacity = {Capacity_average[i]:.4f} bps/Hz")
print(f"\nResults saved to:")
print(f"  NMSE_K_{K}.npy")
print(f"  Capacity_K_{K}.npy")
print("="*60)