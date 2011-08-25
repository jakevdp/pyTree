from npy_balltree import BallTree
import numpy as np
import cPickle as pickle

X = np.random.random((10,3))
ball_tree = BallTree(X, 2)

ind, dist = ball_tree.query(X[0], 3)
print ind
print dist

s = pickle.dumps(ball_tree)
ball_tree2 = pickle.loads(s)

ind2, dist2 = ball_tree2.query(X[0], 3)
print ind2
print dist2
