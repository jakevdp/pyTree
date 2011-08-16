import numpy as np
import pylab as pl

def calc_tree_size(N, leaf_size=1):
    """
    compute the tree size for a given N.
    """
    if N <= leaf_size:
        return 1

    else:
        N1 = int(N/2)
        N2 = N - N1
        return 1 + calc_tree_size(N1, leaf_size) + calc_tree_size(N2, leaf_size)

def calc_max_index(N, leaf_size=1, i=0):
    if N <= leaf_size:
        return i

    else:
        N2 = int(N/2)
        N1 = N - N2

        i1 = 2 * i + 1
        i2 = 2 * i + 2

        return max(calc_max_index(N1, leaf_size, i1),
                   calc_max_index(N2, leaf_size, i2))
        

if __name__ == '__main__':
    leaf_size = 20

    N = np.arange(1, 1050)
    N_nodes = np.array([calc_tree_size(n, leaf_size) for n in N])

    array_size = 1 + np.array([calc_max_index(n, leaf_size) for n in N])

    upper_bound = 2 ** (1 + np.ceil(np.log2((N + leaf_size - 1)/ leaf_size))) - 1

    pl.subplot(211)
    pl.title("leaf_size = %i" % leaf_size)
    pl.plot(N, N_nodes, '-k', label='number of nodes')
    pl.plot(N, array_size, '-r', label='required array size')
    pl.plot(N, upper_bound, '-g', label='analytic upper bound')
    pl.legend(loc=0)
    pl.ylabel('number of nodes')

    pl.subplot(212)
    pl.plot(N, upper_bound - array_size)
    pl.ylabel('wasted nodes')
    pl.xlabel('n_samples')

    pl.show()
    
        
