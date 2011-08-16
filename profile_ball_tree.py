import pstats, cProfile

import numpy as np

from npy_balltree import BallTree

def calc_neighbors(N=1000, D=2, k=10, leaf_size=20, filename='profile.out'):
    X = np.random.random((N,D))
    ball_tree = BallTree(X, leaf_size)
    dist, ind = ball_tree.query(X, k, return_distance=True)

cProfile.runctx("calc_neighbors()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
