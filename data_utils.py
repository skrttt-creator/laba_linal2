import numpy as np

def generate_data(n_samples=500, type='blobs', noise=0.0, seed=42):
    np.random.seed(seed)
    
    if type == 'blobs':
        n_class = n_samples // 2
        X = np.random.randn(n_samples, 2)
        X[:n_class] += 2.5 
        y = np.array([0]*n_class + [1]*n_class)
        
    elif type == 'xor':
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        
    elif type == 'circle':
        X = np.random.randn(n_samples, 2)
        dist = np.linalg.norm(X, axis=1)
        y = (dist > 1.0).astype(int) 
        
    if noise > 0:
        n_flip = int(noise * n_samples)
        flip_indices = np.random.choice(n_samples, n_flip, replace=False)
        y[flip_indices] = 1 - y[flip_indices] 

    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    return X[indices], y[indices]

def add_polynomial_features(X):
    x1 = X[:, 0:1]
    x2 = X[:, 1:2]
    return np.hstack([X, x1 * x2, x1**2, x2**2])
