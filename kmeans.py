import numpy as np
import random as rnd

def init_clusters(k=3, X=None):
    C = np.empty((k, X.shape[1]))
    for i in range(X.shape[1]):
        xmax = X[:,i].max()
        xmin = X[:,i].min()
        C[:,i] = np.random.random(size=k) * (xmax-xmin) + xmin
    return C

def cost(X, C):
    k = C.shape[0]
    d = np.zeros((X.shape[0], k))
    for i in range(k):
        d[:,i] = distance(X-C[i])
    return d

def distance(v):
    return np.einsum('ij,ij->i', v, v)

def kmeans_step(k=3, X=None):
    C = init_clusters(k=k, X=X)
    C_old = np.zeros(C.shape)
    C_diff = np.mean(distance(C-C_old))

    tolerance = 1.e-7

    while C_diff > tolerance:
        y = np.argmin(cost(X, C), axis=1)

        for i in range(k):
            if i not in y:
                y[rnd.randrange(len(y))] = i
            C[i] = X[y==i].mean(axis=0)

        C_diff = np.mean(distance(C-C_old))
        C_old = C.copy()

    J = np.mean(cost(X, C))

    return C, y, J

def kmeans(k=3, X=None, steps=1):
    C = [None] * steps
    y = [None] * steps
    J = [None] * steps

    for i in range(steps):
        C[i], y[i], J[i] = kmeans_step(k=k, X=X)
    i = np.argmin(J)

    return C[i], y[i]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight')
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=200, n_features=2, centers=3, random_state=42)

    C, yc = kmeans(k=3, X=X, steps=2)

    plt.figure()
    plt.scatter(X[:,0], X[:,1], s=20)
    plt.scatter(C[:,0], C[:,1], s=80, marker='*')
    plt.tight_layout()
    plt.savefig('kmeans.pdf')
