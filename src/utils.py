# utils.py
import numpy as np

def db2pow(db):
    return 10**(db/10)

def pow2db(p):
    return 10*np.log10(p)
