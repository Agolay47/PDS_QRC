# src/ar2.py

import numpy as np

def generate_linear_series(alpha: float,
                           beta: float,
                           A: float,
                           B: float,
                           T_max: int) -> np.ndarray:
    
    y = np.zeros(T_max + 1, dtype=float)
    y[0] = A
    y[1] = B
    for t in range(1, T_max):
        y[t + 1] = alpha * y[t] + beta * y[t - 1]
    return y
