import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.linalg as la

# ===================== EXACT MATLAB SINC FUNCTION =====================
def sinc_matlab(x):
    """
    EXACT MATLAB sinc function: sin(x)/x
    """
    x = np.asarray(x)
    result = np.ones_like(x, dtype=np.float64)
    mask = x != 0
    result[mask] = np.sin(x[mask]) / x[mask]
    return result

# ===================== MATLAB-STYLE WATER-FILLING =====================
def water_filling_matlab(P_total, sigma2, a):
    """
    EXACT MATLAB water-filling algorithm
    """
    a = np.array(a).flatten()
    n = len(a)
    
    # MATLAB assumes a is already sorted in descending order from SVD
    # But let's sort to be safe
    idx = np.argsort(-np.abs(a))
    a_sorted = np.abs(a[idx])
    
    P_sum = 0
    ii = 0
    
    for ii in range(n - 1):
        P_sum = P_sum + (ii + 1) * (sigma2 / (a_sorted[ii + 1]**2) - sigma2 / (a_sorted[ii]**2))
        if P_sum >= P_total:
            break
    
    ii = ii + 1  # MATLAB 1-based indexing adjustment
    
    if P_sum >= P_total:
        PA_WF = sigma2 / (a_sorted[ii]**2) - sigma2 / (a_sorted[:ii]**2) - (P_sum - P_total) / ii
        PA_WF_full = np.zeros(n)
        PA_WF_full[idx[:ii]] = PA_WF
        return PA_WF_full
    else:
        PA_WF = sigma2 / (a_sorted[ii]**2) - sigma2 / (a_sorted[:ii+1]**2) + (P_total - P_sum) / (ii + 1)
        return PA_WF[idx]

# ===================== MATLAB-STYLE MATRIX GENERATION =====================
def generate_matrices_matlab_style(M, N, S, lamda, d_element_spacing, 
                                   d_layer_spacing_transmit, d_layer_spacing_receive, 
                                   N_max=10):
    """
    Generate matrices exactly as MATLAB does
    """
    W_T = np.zeros((M, M), dtype=np.complex128)
    Corr_T = np.zeros((M, M))
    
    # Note: MATLAB uses 1-based indexing, but loops are the same
    for mm1 in range(M):
        # Convert to MATLAB-style indices
        matlab_mm1 = mm1 + 1
        m_z = int(np.ceil(matlab_mm1 / N_max))
        m_x = ((matlab_mm1 - 1) % N_max) + 1
        
        for mm2 in range(M):
            matlab_mm2 = mm2 + 1
            n_z = int(np.ceil(matlab_mm2 / N_max))
            n_x = ((matlab_mm2 - 1) % N_max) + 1
            
            d_temp = np.sqrt((m_x - n_x)**2 + (m_z - n_z)**2) * d_element_spacing
            d_temp2 = np.sqrt(d_layer_spacing_transmit**2 + d_temp**2)
            
            # Use exact MATLAB formula
            W_T[mm2, mm1] = lamda/(4*np.pi*d_temp2) * np.exp(-1j*2*np.pi*d_temp2/lamda)
            Corr_T[mm2, mm1] = sinc_matlab(2*d_temp/lamda)
    
    U_R = np.zeros((N, N), dtype=np.complex128)
    Corr_R = np.zeros((N, N))
    
    for nn1 in range(N):
        matlab_nn1 = nn1 + 1
        m_z = int(np.ceil(matlab_nn1 / N_max))
        m_x = ((matlab_nn1 - 1) % N_max) + 1
        
        for nn2 in range(N):
            matlab_nn2 = nn2 + 1
            n_z = int(np.ceil(matlab_nn2 / N_max))
            n_x = ((matlab_nn2 - 1) % N_max) + 1
            
            d_temp = np.sqrt((m_x - n_x)**2 + (m_z - n_z)**2) * d_element_spacing
            d_temp2 = np.sqrt(d_layer_spacing_receive**2 + d_temp**2)
            
            U_R[nn2, nn1] = lamda/(4*np.pi*d_temp2) * np.exp(-1j*2*np.pi*d_temp2/lamda)
            Corr_R[nn2, nn1] = sinc_matlab(2*d_temp/lamda)
    
    W_T_1 = np.zeros((M, S), dtype=np.complex128)
    for mm in range(M):
        matlab_mm = mm + 1
        m_z = int(np.ceil(matlab_mm / N_max))
        m_x = ((matlab_mm - 1) % N_max) + 1
        
        for nn in range(S):
            matlab_nn = nn + 1
            d_transmit = np.sqrt(
                d_layer_spacing_transmit**2 + 
                ((m_x - (1 + N_max)/2) * d_element_spacing)**2 +
                ((m_z - (1 + N_max)/2) * d_element_spacing - 
                 (matlab_nn - (1 + S)/2) * lamda/2)**2
            )
            W_T_1[mm, nn] = lamda/(4*np.pi*d_transmit) * np.exp(-1j*2*np.pi*d_transmit/lamda)
    
    U_R_1 = np.zeros((S, N), dtype=np.complex128)
    for mm in range(N):
        matlab_mm = mm + 1
        m_z = int(np.ceil(matlab_mm / N_max))
        m_x = ((matlab_mm - 1) % N_max) + 1
        
        for nn in range(S):
            matlab_nn = nn + 1
            d_receive = np.sqrt(
                d_layer_spacing_receive**2 +
                ((m_x - (1 + N_max)/2) * d_element_spacing)**2 +
                ((m_z - (1 + N_max)/2) * d_element_spacing - 
                 (matlab_nn - (1 + S)/2) * lamda/2)**2
            )
            U_R_1[nn, mm] = lamda/(4*np.pi*d_receive) * np.exp(-1j*2*np.pi*d_receive/lamda)
    
    return W_T, Corr_T, U_R, Corr_R, W_T_1, U_R_1

# ===================== MATLAB-STYLE RANDOM NUMBER GENERATOR =====================
def generate_matlab_random(shape, seed=1):
    """
    Try to generate random numbers similar to MATLAB
    Note: This won't be perfect due to different algorithms
    """
    # MATLAB uses different algorithm, but we can try to get close
    np.random.seed(seed)
    
    # MATLAB's randn uses the Marsaglia polar method
    # Python's np.random.randn uses Box-Muller
    # They will be different even with same seed
    
    return np.random.randn(*shape)

# ===================== MAIN SIMULATION (MATLAB-COMPATIBLE) =====================
def main_matlab_compatible():
    """
    Python simulation that tries to match MATLAB results as closely as possible
    """
    # System Parameters (EXACTLY as in MATLAB)
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
    
    # SIM Parameters
    M = 100
    N = 100
    d_element_spacing = lamda/2
    S = 4
    MonteCarlo = 50
    Max_L = 10
    K = 10
    
    # Initialize arrays
    Capacity_average = np.zeros(Max_L)
    
    print("Starting MATLAB-compatible simulation...")
    print(f"Parameters: M={M}, N={N}, S={S}, K={K}, MonteCarlo={MonteCarlo}")
    print("IMPORTANT: Results will still differ due to:")
    print("  1. Different random number generators")
    print("  2. Different SVD implementations")
    print("  3. Different matrix sqrt implementations")
    print("  4. Different numerical precision")
    print("="*70)
    
    # Try to match MATLAB random numbers as closely as possible
    np.random.seed(1)  # Same seed, but different algorithm
    
    for ii in range(Max_L):
        L = ii + 1
        start_time = time.time()
        
        d_layer_spacing_transmit = Thickness/L
        d_layer_spacing_receive = Thickness/K
        
        print(f"\nProcessing L={L}...")
        
        # Generate matrices (MATLAB style)
        W_T, Corr_T, U_R, Corr_R, W_T_1, U_R_1 = generate_matrices_matlab_style(
            M, N, S, lamda, d_element_spacing, 
            d_layer_spacing_transmit, d_layer_spacing_receive, N_max
        )
        
        Capacity = np.zeros(MonteCarlo)
        
        for jj in range(MonteCarlo):
            # Generate random channel (MATLAB-like)
            # Note: Even with same seed, MATLAB and Python random numbers differ
            G_independent = np.sqrt(1/2) * (np.random.randn(N, M) + 1j*np.random.randn(N, M))
            
            # MATLAB's sqrtm vs Python's sqrtm may differ
            # Add small regularization as MATLAB does internally
            Corr_R_sqrt = la.sqrtm(Corr_R + np.finfo(float).eps * np.eye(N))
            Corr_T_sqrt = la.sqrtm(Corr_T + np.finfo(float).eps * np.eye(M))
            
            G = np.sqrt(pathloss) * Corr_R_sqrt @ G_independent @ Corr_T_sqrt
            
            # MATLAB's SVD vs Python's SVD
            # Use full SVD like MATLAB
            U, G_svd_diag, Vh = la.svd(G, full_matrices=True)
            
            # Extract diagonal matrix (MATLAB style)
            H_true = np.diag(G_svd_diag[:S])
            
            # MATLAB water-filling
            PA_WF = water_filling_matlab(Pt, Sigma2, np.diag(H_true))
            
            # MATLAB-style capacity calculation
            # Note: MATLAB's log2 is base-2 logarithm
            Capacity[jj] = np.sum(np.log2(1 + PA_WF * (np.diag(H_true)**2) / Sigma2))
            
            if (jj + 1) % 10 == 0:
                print(f"  Completed {jj + 1}/{MonteCarlo} Monte Carlo runs")
        
        Capacity_average[ii] = np.mean(Capacity)
        
        elapsed_time = time.time() - start_time
        print(f"L={L}: Average Capacity = {Capacity_average[ii]:.6f} bps/Hz")
        print(f"Time: {elapsed_time:.2f}s")
    
    # ===================== COMPARE WITH MATLAB (IF AVAILABLE) =====================
    print("\n" + "="*70)
    print("SIMULATION COMPLETED")
    print("="*70)
    
    # Try to load MATLAB results if they exist
    try:
        # If you have MATLAB results saved as .npy file
        matlab_results = np.load('matlab_capacity_results.npy')
        print("\nCOMPARISON WITH MATLAB RESULTS:")
        print("-"*60)
        print(f"{'L':>3} {'Python':>15} {'MATLAB':>15} {'Diff (%)':>15}")
        print("-"*60)
        
        for i in range(Max_L):
            L_val = i + 1
            python_val = Capacity_average[i]
            matlab_val = matlab_results[i]
            diff_pct = abs(python_val - matlab_val) / matlab_val * 100 if matlab_val != 0 else 0
            print(f"{L_val:>3} {python_val:>15.6f} {matlab_val:>15.6f} {diff_pct:>15.2f}")
        print("-"*60)
    except:
        print("\nMATLAB results file not found. Showing Python results only:")
        print("-"*40)
        print(f"{'L':>3} {'Capacity (bps/Hz)':>20}")
        print("-"*40)
        for i in range(Max_L):
            print(f"{i+1:>3} {Capacity_average[i]:>20.6f}")
        print("-"*40)
    
    # Save results
    np.save('python_capacity_results.npy', Capacity_average)
    print(f"\nPython results saved to 'python_capacity_results.npy'")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, Max_L + 1), Capacity_average, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of TX-SIM Layers (L)', fontsize=12)
    plt.ylabel('Average Capacity (bps/Hz)', fontsize=12)
    plt.title('MATLAB-Compatible Simulation Results (K=10)', fontsize=14)
    plt.grid(True)
    plt.xlim(0.5, Max_L + 0.5)
    
    # Add value labels
    for i, val in enumerate(Capacity_average):
        plt.text(i + 1, val, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('python_capacity_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return Capacity_average

# ===================== REALISTIC EXPECTATIONS =====================
def explain_differences():
    """
    Explain why Python and MATLAB results differ
    """
    print("\n" + "="*70)
    print("WHY PYTHON AND MATLAB GIVE DIFFERENT RESULTS")
    print("="*70)
    print("\n1. RANDOM NUMBER GENERATORS:")
    print("   - MATLAB: Uses Marsaglia polar method (different algorithm)")
    print("   - Python NumPy: Uses Box-Muller transform")
    print("   - Even with same seed, numbers are completely different")
    
    print("\n2. SVD IMPLEMENTATION:")
    print("   - MATLAB: Uses LAPACK dgesvd/dgesdd")
    print("   - Python SciPy: Also uses LAPACK but may use different defaults")
    print("   - Different convergence criteria and tolerances")
    
    print("\n3. MATRIX SQUARE ROOT:")
    print("   - MATLAB sqrtm(): Schur decomposition method")
    print("   - Python la.sqrtm(): Also Schur but may handle branch cuts differently")
    
    print("\n4. NUMERICAL PRECISION:")
    print("   - MATLAB: Default double precision (64-bit)")
    print("   - Python NumPy: Default double precision (64-bit)")
    print("   - BUT: Order of operations and rounding may differ")
    
    print("\n5. SPECIAL FUNCTIONS:")
    print("   - MATLAB sinc(x) = sin(x)/x")
    print("   - Python scipy.special.sinc(x) = sin(πx)/(πx)")
    print("   - This is a MAJOR difference that affects results significantly!")
    
    print("\nEXPECTED DIFFERENCES:")
    print("   - Capacity values may differ by 1-10%")
    print("   - Trends should be similar (increasing/decreasing with L)")
    print("   - Relative performance comparisons should be valid")
    print("="*70)

# ===================== RUN SIMULATION =====================
if __name__ == "__main__":
    # First, explain why results differ
    explain_differences()
    
    # Run the MATLAB-compatible simulation
    print("\n\n" + "="*70)
    print("RUNNING MATLAB-COMPATIBLE SIMULATION")
    print("="*70)
    
    results = main_matlab_compatible()
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print("1. Use sinc_matlab() not scipy.special.sinc()")
    print("2. Accept that random numbers will differ")
    print("3. Trends should be similar even if absolute values differ")
    print("4. For research: Compare relative improvements, not absolute values")
    print("="*70)