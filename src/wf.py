import numpy as np

def WF(P_total, sigma2, a):
    a = np.array(a).reshape(-1)
    P_sum = 0.0

    for ii in range(len(a) - 1):
        P_sum += (ii + 1) * (sigma2 / a[ii + 1]**2 - sigma2 / a[ii]**2)
        if P_sum >= P_total:
            break

    if P_sum >= P_total:
        PA_WF = sigma2 / a[ii + 1]**2 - sigma2 / a[:ii + 1]**2 - (P_sum - P_total) / (ii + 1)
        PA_WF = np.concatenate([PA_WF, np.zeros(len(a) - len(PA_WF))])
    else:
        PA_WF = sigma2 / a[ii + 1]**2 - sigma2 / a[:ii + 2]**2 + (P_total - P_sum) / (ii + 2)

    return PA_WF
