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

#N = 5
#D = 2
#k = 3
#leaf_size = 1

X = np.random.random((N,D))
                       
def compute_neighbors(X, BT):
    t0 = time()
    ball_tree = BT(X, leaf_size)
    t1 = time()
    dist, ind = ball_tree.query(X, k, return_distance = True)
    t2 = time()
    count = ball_tree.query_radius(X, 0.2, count_only = True)
    t3 = time()

    i_sort = np.argsort(dist, 1)
    for i in range(dist.shape[0]):
        dist[i] = dist[i, i_sort[i]]
        ind[i] = ind[i, i_sort[i]]

    print "  build: %.1e sec" % (t1 - t0)
    print "  query: %.1e sec" % (t2 - t1)
    print "  q_rad: %.1e sec" % (t3 - t2)

    return dist, ind, count

print "cython/npy code:"
dist, ind, count = compute_neighbors(X, npyBallTree)

print "c++ code:"
sk_dist, sk_ind, sk_count = compute_neighbors(X, cppBallTree)

print "Matches:"
print ' indices:  ', np.allclose(dist, sk_dist)
print ' distances:', np.allclose(ind, sk_ind)
print ' counts:', np.allclose(count, sk_count)

i_diff = np.unique(np.where(abs(dist - sk_dist) > 1E-10)[0])

if i_diff > 0:
    print 70*'!'
    print '    ', i_diff, 'differences detected!!'
    print '     note: random seed =', rseed
