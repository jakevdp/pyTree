from time import time

import numpy as np

from cpp_balltree import BallTree as cppBallTree

from npy_balltree import BallTree as npyBallTree

rseed = np.random.randint(1E4)
print 'rseed =',  rseed
np.random.seed(rseed)

N = 1000
D = 5
k = 10
leaf_size = 20

X = np.random.random((N,D))

#import pylab
#pylab.scatter(X[:,0], X[:,1])
#for i in range(X.shape[0]):
#    pylab.text(X[i,0], X[i,1], str(i))
#pylab.xlim(0,1)
#pylab.ylim(0,1)
#pylab.axis('scaled')

def compute_neighbors(X, BT):
    t0 = time()
    ball_tree = BT(X, leaf_size)
    t1 = time()
    dist, ind = ball_tree.query(X, k, return_distance = True)
    t2 = time()

    i_sort = np.argsort(dist, 1)
    for i in range(dist.shape[0]):
        dist[i] = dist[i, i_sort[i]]
        ind[i] = ind[i, i_sort[i]]

    print "  build: %.1e sec" % (t1 - t0)
    print "  query: %.1e sec" % (t2 - t1)

    return dist, ind

print "cython/npy code:"
dist, ind = compute_neighbors(X, npyBallTree)

print "c++ code:"
sk_dist, sk_ind = compute_neighbors(X, cppBallTree)

print "Matches:"
print ' indices:  ', np.allclose(dist, sk_dist)
print ' distances:', np.allclose(ind, sk_ind)

i_diff = np.unique(np.where(abs(dist - sk_dist) > 1E-10)[0])
for i in i_diff:
    print i, ind[i], sk_ind[i]
    print '  ', dist[i], sk_dist[i]

try:
    pylab.show()
except:
    pass
