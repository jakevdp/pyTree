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
    r_ind = ball_tree.query_radius(X, 0.2, count_only = False)
    t4 = time()
    rd_ind, rd_dist = ball_tree.query_radius(X, 0.2, return_distance = True)
    t5 = time()

    for i in range(len(r_ind)):
        r_ind[i].sort()
        
        i_sort = np.argsort(rd_ind[i])
        rd_ind[i] = rd_ind[i][i_sort]
        rd_dist[i] = rd_dist[i][i_sort]

    print "  build:          %.1e sec" % (t1 - t0)
    print "  query:          %.1e sec" % (t2 - t1)
    print "  q_rad count:    %.1e sec" % (t3 - t2)
    print "  q_rad idx:      %.1e sec" % (t4 - t3)
    print "  q_rad idx/dist: %.1e sec" % (t5 - t4)

    return dist, ind, count, r_ind, rd_ind, rd_dist

print "cython/npy code:"
(dist, ind, count, 
 rind, rdind, rddist) = compute_neighbors(X, npyBallTree)

print "c++ code:"
(sk_dist, sk_ind, sk_count, 
 sk_rind, sk_rdind, sk_rddist) = compute_neighbors(X, cppBallTree)

print "Matches:"
print ' indices:  ', np.allclose(dist, sk_dist)
print ' distances:', np.allclose(ind, sk_ind)
print ' rad. counts:', np.allclose(count, sk_count)
print ' rad. indices:', np.all([np.allclose(rind[i], sk_rind[i])
                                for i in range(len(rind))])
print ' rad. distances:', np.all([np.allclose(rddist[i], sk_rddist[i])
                                 for i in range(len(rddist))])

