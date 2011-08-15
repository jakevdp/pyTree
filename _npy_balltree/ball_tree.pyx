# cython: profile=True

import numpy as np
cimport numpy as np

cimport cython

#define data type
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

#define integer/index type
ITYPE = np.uint32
ctypedef np.uint32_t ITYPE_t

#define boolean type
BOOL = np.int32
ctypedef np.int32_t BOOL_t

#max function
cdef inline DTYPE_t dmax(DTYPE_t x, DTYPE_t y):
    if x >= y:
        return x
    else:
        return y

cdef inline DTYPE_t dmin(DTYPE_t x, DTYPE_t y):
    if x <= y:
        return x
    else:
        return y

cdef DTYPE_t infinity = np.inf

#distance function
#@cython.boundscheck(False)
cdef DTYPE_t dist(np.ndarray[DTYPE_t, ndim=1, mode='c'] x1,
                  np.ndarray[DTYPE_t, ndim=1, mode='c'] x2,
                  DTYPE_t p):
    cdef ITYPE_t i
    cdef DTYPE_t r, d
    cdef ITYPE_t L = x1.shape[0]
    r = 0
    if p == 2:
        for i from 0 <= i < L:
            d = x1[i] - x2[i]
            r += d * d
        r = r ** 0.5
    elif p == infinity:
        for i from 0 <= i < L:
            r = dmax(r, abs(x1[i] - x2[i]))
    elif p == 1:
        for i from 0 <= i < L:
            r += abs(x1[i] - x2[i])
    else:
        for i from 0 <= i < L:
            d = abs(x1[i] - x2[i])
            r += d ** p
        r = r ** (1. / p)
    return r

cdef DTYPE_t dist_p_ptr(DTYPE_t *x1, DTYPE_t *x2, ITYPE_t n, DTYPE_t p):
    cdef ITYPE_t i
    cdef DTYPE_t r, d
    r = 0
    if p == 2:
        for i from 0 <= i < n:
            d = x1[i] - x2[i]
            r += d * d
    elif p == infinity:
        for i from 0 <= i < n:
            r = dmax(r, abs(x1[i] - x2[i]))
    elif p == 1:
        for i from 0 <= i < n:
            r += abs(x1[i] - x2[i])
    else:
        for i from 0 <= i < n:
            d = abs(x1[i] - x2[i])
            r += d ** p
    return r

cdef DTYPE_t dist_ptr(DTYPE_t *x1, DTYPE_t *x2, ITYPE_t n, DTYPE_t p):
    cdef ITYPE_t i
    cdef DTYPE_t r, d
    r = 0
    if p == 2:
        for i from 0 <= i < n:
            d = x1[i] - x2[i]
            r += d * d
        r = r ** 0.5
    elif p == infinity:
        for i from 0 <= i < n:
            r = dmax(r, abs(x1[i] - x2[i]))
    elif p == 1:
        for i from 0 <= i < n:
            r += abs(x1[i] - x2[i])
    else:
        for i from 0 <= i < n:
            d = abs(x1[i] - x2[i])
            r += d ** p
        r = r ** (1. / p)
    return r
    

# dist_p returns d^p
# in order to recover the true distance, call dist_from_dist_p()
cdef DTYPE_t dist_p(np.ndarray[DTYPE_t, ndim=1, mode='c'] x1,
                    np.ndarray[DTYPE_t, ndim=1, mode='c'] x2,
                    DTYPE_t p):
    cdef ITYPE_t i
    cdef DTYPE_t r, d
    cdef ITYPE_t L = x1.shape[0]
    r = 0
    if p == 2:
        for i from 0 <= i < L:
            d = x1[i] - x2[i]
            r += d * d
    elif p == infinity:
        for i from 0 <= i < L:
            r = dmax(r, abs(x1[i] - x2[i]))
    elif p == 1:
        for i from 0 <= i < L:
            r += abs(x1[i] - x2[i])
    else:
        for i from 0 <= i < L:
            d = abs(x1[i] - x2[i])
            r += d ** p
    return r


cdef DTYPE_t dist_from_dist_p(DTYPE_t r, DTYPE_t p):
    if p == 2:
        return r ** 0.5
    elif p == infinity:
        return r
    elif p == 1:
        return r
    else:
        return r ** (1. / p)


cdef class BallTree:
    """
    BallTree class

    implementation of BallTree using numpy arrays for picklability

    Implementation Notes
    --------------------
    
    Instead of dynamically allocating Node objects, this implementation
    instead uses a pre-allocated numpy array with implicit linking.
    Four arrays are used:
        data : float, shape = (n_samples, n_features)
            The input data array from which the tree is built

        node_float_arr : float, shape = (n_nodes, n_features + 1)
            For node index i, the array has the following:
                centroid = node_float_arr[i, :n_features]
                radius = node_float_arr[i, n_features]

        node_int_arr : int, shape = (n_nodes, 3)
            For node index i, the array has the following:
                idx_start = node_int_arr[i, 0]
                idx_end = node_int_arr[i, 1]
                is_leaf = node_int_arr[i, 2]

        indices : int, shape = (n_samples,)
            This array stores a list of indices to which the node arrays point
            for a node with idx_start and idx_end, the indices of the points
            in the node are found in indices[idx_start:idx_end]

    The nodes are implicitly linked, in a way that is similar to that of the
    heapsort algorithm.  Node i (zero-indexed) has children at indices
    (2 * i + 1) and (2 * i + 2).  Node i has parent floor((i-1)/2).  Node 0
    has no parent.

    Because all necessary information is stored in these arrays, the object
    can be pickled and unpickled with no loss of data (unlike the basic
    implementation using pointers and dynamic allocation of nodes).
    """
    cdef np.ndarray data
    cdef np.ndarray idx_array
    cdef np.ndarray node_float_arr
    cdef np.ndarray node_int_arr
    cdef ITYPE_t p
    
    def __init__(self, X, leaf_size=20, p=2):
        self.data = np.asarray(X, dtype=DTYPE)
        assert self.data.ndim == 2
        
        cdef ITYPE_t n_samples = self.data.shape[0]
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef ITYPE_t n_nodes = (2 ** (1 + np.ceil(np.log2((n_samples + 2)
                                                          / leaf_size))) - 1)

        self.idx_array = np.arange(n_samples, dtype=ITYPE)
    
        self.node_float_arr = np.empty((n_nodes, n_features + 1),
                                       dtype=DTYPE, order='C')
        self.node_int_arr = np.empty((n_nodes, 3),
                                     dtype=ITYPE, order='C')
        self.p = p

        Node_build(leaf_size, p, n_samples, n_features, n_nodes,
                   <DTYPE_t*> self.data.data,
                   <ITYPE_t*> self.idx_array.data,
                   <DTYPE_t*> self.node_float_arr.data,
                   <ITYPE_t*> self.node_int_arr.data)


    def query(self, X, k, return_distance=True):
        X = np.asarray(X, dtype=DTYPE, order='C')
        X = np.atleast_2d(X)
        assert X.shape[-1] == self.data.shape[1]
        assert k <= self.data.shape[0]

        # almost-flatten x for iteration
        orig_shape = X.shape
        X = X.reshape((-1, X.shape[-1]))

        cdef ITYPE_t i
        cdef ITYPE_t N
        cdef np.ndarray distances = np.empty((X.shape[0], k), dtype=DTYPE)
        cdef np.ndarray idx_array = np.empty((X.shape[0], k), dtype=ITYPE)

        distances[:] = np.inf

        for i from 0 <= i < X.shape[0]:
            Node_query(0, X[i], self.p, k, distances[i], idx_array[i],
                       self.data, self.idx_array,
                       self.node_float_arr, self.node_int_arr)
            
        # deflatten results
        if return_distance:
            return (distances.reshape((orig_shape[:-1]) + (k,)),
                    idx_array.reshape((orig_shape[:-1]) + (k,)))
        else:
            return idx_array.reshape((orig_shape[:-1]) + (k,))
        

######################################################################
# Node_build
#  This function builds the ball tree.
#  Four data buffers should be passed to the function:
#
#    data: size = n_samples * n_features
#         this array holds the input data
#
#    idx_array: size = n_samples
#        on input, this should be [0, 1, 2, ... n_samples - 1]
#        on output, the indices will be re-ordered so that the indices
#                   within each node are contiguous.
#
#    node_float_arr: size = n_nodes * (n_features + 1)
#        for each node, this contains the centroid location and the radius
#
#    node_int_arr: size = n_nodes * 3
#        for each node, this contains (idx_start, idx_end, is_leaf)
#        idx_start, idx_end : these point to the portion of idx_array
#                             which point to the points in this node
#        is_leaf : used as a boolean: tells whether this node is a leaf or not.
#
cdef void Node_build(ITYPE_t leaf_size, ITYPE_t p,
                     ITYPE_t n_samples, ITYPE_t n_features,
                     ITYPE_t n_nodes,
                     DTYPE_t* data,
                     ITYPE_t* idx_array,
                     DTYPE_t* node_float_arr,
                     ITYPE_t* node_int_arr):
    cdef ITYPE_t idx_start = 0
    cdef ITYPE_t idx_end = n_samples
    cdef ITYPE_t n_points = n_samples
    cdef DTYPE_t radius
    cdef ITYPE_t i, i_node, i_parent

    cdef DTYPE_t* centroid = node_float_arr
    cdef ITYPE_t* node_info = node_int_arr
    cdef ITYPE_t* parent_info
    cdef DTYPE_t* point

    if n_points == 0:
        raise ValueError, "zero-sized node"

    #------------------------------------------------------------
    # take care of the head node
    node_int_arr[0] = idx_start
    node_int_arr[1] = idx_end

    # determine Node centroid
    compute_centroid(centroid, data, idx_array,
                     n_features, n_samples)

    # determine Node radius
    radius = 0
    for i from idx_start <= i < idx_end:
        radius = dmax(radius, 
                      dist_p_ptr(centroid, data + n_features * idx_array[i],
                                 n_features, p))
    centroid[n_features] = dist_from_dist_p(radius, p)

    # check if this is a leaf
    if n_points <= leaf_size:
        node_info[2] = 1

    else:
        # not a leaf
        node_info[2] = 0
        
        # find dimension with largest spread
        i_max = find_split_dim(data, idx_array + idx_start,
                               n_features, n_points)
        
        # sort idx_array along this dimension
        partition_indices(data,
                          idx_array + idx_start,
                          i_max,
                          n_points / 2,
                          n_features,
                          n_points)

    #------------------------------------------------------------
    # cycle through all child nodes
    for i_node from 1 <= i_node < n_nodes:
        i_parent = (i_node - 1) / 2
        parent_info = node_int_arr + 3 * i_parent
        
        node_info = node_int_arr + 3 * i_node
        node_info[2] = 1

        # if parent is a leaf then we stop here
        if parent_info[2]:
            continue
    
        centroid = node_float_arr + i_node * (n_features + 1)
        
        # find indices for this node
        idx_start = parent_info[0]
        idx_end = parent_info[1]
        
        if i_node % 2 == 1:
            idx_start = (idx_start + idx_end) / 2
        else:
            idx_end = (idx_start + idx_end) / 2

        node_info[0] = idx_start
        node_info[1] = idx_end

        n_points = idx_end - idx_start

        if n_points == 0:
            raise ValueError, "zero-sized node"

        elif n_points == 1:
            #copy this point to centroid
            copy_array(centroid, 
                       data + idx_array[idx_start] * n_features,
                       n_features)

            #store radius in array
            centroid[n_features] = 0

            #is a leaf
            node_info[2] = 1

        else:
            # determine Node centroid
            compute_centroid(centroid, data, idx_array + idx_start,
                             n_features, n_points)

            # determine Node radius
            radius = 0
            for i from idx_start <= i < idx_end:
                radius = dmax(radius, 
                              dist_p_ptr(centroid,
                                       data + n_features * idx_array[i],
                                       n_features, p))
            centroid[n_features] = dist_from_dist_p(radius, p)

            if n_points <= leaf_size:
                node_info[2] = 1

            else:
                # not a leaf
                node_info[2] = 0
                
                # find dimension with largest spread
                i_max = find_split_dim(data, idx_array + idx_start,
                                       n_features, n_points)
                
                # sort indices along this dimension
                partition_indices(data,
                                  idx_array + idx_start,
                                  i_max,
                                  n_points / 2,
                                  n_features,
                                  n_points)


cdef inline void copy_array(DTYPE_t* x, DTYPE_t* y, ITYPE_t n):
    # copy array y into array x
    cdef ITYPE_t i
    for i from 0 <= i < n:
        x[i] = y[i]

cdef void compute_centroid(DTYPE_t* centroid,
                           DTYPE_t* data,
                           ITYPE_t* node_indices,
                           ITYPE_t n_features,
                           ITYPE_t n_points):
    # centroid points to an array of length n_features
    # data points to an array of length n_samples * n_features
    # node_indices = idx_array + idx_start
    cdef DTYPE_t *this_pt
    cdef ITYPE_t i, j
    
    for j from 0 <= j < n_features:
        centroid[j] = 0

    for i from 0 <= i < n_points:
        this_pt = data + n_features * node_indices[i]
        for j from 0 <= j < n_features:
            centroid[j] += this_pt[j]

    for j from 0 <= j < n_features:
        centroid[j] /= n_points


cdef ITYPE_t find_split_dim(DTYPE_t* data,
                            ITYPE_t* node_indices,
                            ITYPE_t n_features,
                            ITYPE_t n_points):
    #i_max = np.argmax(np.max(data, 0) - np.min(data, 0))
    cdef DTYPE_t min_val, max_val, val, spread, max_spread
    cdef ITYPE_t i, j, j_max

    j_max = 0
    max_spread = 0

    for j from 0 <= j < n_features:
        max_val = data[node_indices[0] * n_features + j]
        min_val = max_val
        for i from 1 <= i < n_points:
            val = data[node_indices[i] * n_features + j]
            max_val = dmax(max_val, val)
            min_val = dmin(min_val, val)
        spread = max_val - min_val
        if spread > max_spread:
            max_spread = spread
            j_max = j
    return j_max


cdef inline void swap(ITYPE_t* arr, ITYPE_t i1, ITYPE_t i2):
    cdef ITYPE_t tmp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = tmp

cdef void partition_indices(DTYPE_t* data,
                            ITYPE_t* node_indices,
                            ITYPE_t split_dim,
                            ITYPE_t split_index,
                            ITYPE_t n_features,
                            ITYPE_t n_points):
    cdef ITYPE_t left, right, midindex, i
    cdef DTYPE_t d1, d2
    left = 0
    right = n_points - 1
    
    while True:
        midindex = left
        for i from left <= i < right:
            d1 = data[node_indices[i] * n_features + split_dim] 
            d2 = data[node_indices[right] * n_features + split_dim]
            if d1 < d2:
                swap(node_indices, i, midindex)
                midindex += 1
        swap(node_indices, midindex, right)
        if midindex == split_index:
            break
        elif midindex < split_index:
            left = midindex + 1
        else:
            right = midindex - 1


######################################################################


#@cython.boundscheck(False)
cdef DTYPE_t calc_dist_LB(ITYPE_t i_node,
                          np.ndarray[DTYPE_t, ndim=1, mode='c'] pt,
                          np.ndarray[DTYPE_t, ndim=2, mode='c'] node_float_arr,
                          ITYPE_t p):
    cdef ITYPE_t n = pt.size
    return dmax(0, (dist(pt, node_float_arr[i_node, :n], p)
                    - node_float_arr[i_node, n]))


#@cython.boundscheck(False)
cdef void Node_query(ITYPE_t i_node,
                     np.ndarray[DTYPE_t, ndim=1, mode='c'] pt,
                     ITYPE_t p, ITYPE_t k,
                     np.ndarray[DTYPE_t, ndim=1] near_set_dist,
                     np.ndarray[ITYPE_t, ndim=1] near_set_indx,
                     np.ndarray[DTYPE_t, ndim=2, mode='c'] data,
                     np.ndarray[ITYPE_t, ndim=1, mode='c'] idx_array,
                     np.ndarray[DTYPE_t, ndim=2, mode='c'] node_float_arr,
                     np.ndarray[ITYPE_t, ndim=2, mode='c'] node_int_arr,
                     DTYPE_t dist_LB = -1):
    cdef DTYPE_t dist_pt, dist_LB_1, dist_LB_2
    cdef ITYPE_t i, i_pos
    cdef ITYPE_t idx_start = node_int_arr[i_node, 0]
    cdef ITYPE_t idx_end = node_int_arr[i_node, 1]
    cdef ITYPE_t n_points = idx_end - idx_start
    cdef ITYPE_t n_features = data.shape[1]
    
    if dist_LB < 0:
        dist_LB = calc_dist_LB(i_node, pt, node_float_arr, p)

    #------------------------------------------------------------
    # Case 1: query point is outside node radius
    if dist_LB >= max_heap_largest(near_set_dist):
        return

    #------------------------------------------------------------
    # Case 2: this is a leaf node.  Update set of nearby points
    elif node_int_arr[i_node, 2]:
        for i from idx_start <= i < idx_end:
            if n_points == 1:
                dist_pt = dist_LB
            else:
                dist_pt = dist(pt, data[idx_array[i]], p)

            insert_max_heap(dist_pt, idx_array[i],
                                near_set_dist, near_set_indx)

    #------------------------------------------------------------
    # Case 3: Node is not a leaf.  Recursively query subnodes
    else:
        dist_LB_1 = calc_dist_LB(2 * i_node + 1,
                                 pt, node_float_arr, p)
        dist_LB_2 = calc_dist_LB(2 * i_node + 2,
                                 pt, node_float_arr, p)
        if dist_LB_1 <= dist_LB_2:
            Node_query(2 * i_node + 1, pt, p, k, near_set_dist, near_set_indx,
                       data, idx_array, node_float_arr, node_int_arr, dist_LB_1)
            Node_query(2 * i_node + 2, pt, p, k, near_set_dist, near_set_indx,
                       data, idx_array, node_float_arr, node_int_arr, dist_LB_2)
        else:
            Node_query(2 * i_node + 2, pt, p, k, near_set_dist, near_set_indx,
                       data, idx_array, node_float_arr, node_int_arr, dist_LB_2)
            Node_query(2 * i_node + 1, pt, p, k, near_set_dist, near_set_indx,
                       data, idx_array, node_float_arr, node_int_arr, dist_LB_1)
        
                     

#----------------------------------------------------------------------
# max_heap functions
#
# This is a basic implementation of a fixed-size binary max-heap.
# It is used to keep track of the k nearest neighbors in a query.
#
# The root node is at heap[0].  The two child nodes of node i are at
# (2 * i + 1) and (2 * i + 2).
# The parent node of node i is node floor((i-1)/2).  Node 0 has no parent.
#
# An empty heap should be full of infinities
#
cdef inline DTYPE_t max_heap_largest(
    np.ndarray[DTYPE_t, ndim=1, mode='c'] heap):
    return heap[0]

cdef void insert_max_heap(DTYPE_t val, ITYPE_t i_val,
                          np.ndarray[DTYPE_t, ndim=1, mode='c'] heap,
                          np.ndarray[ITYPE_t, ndim=1, mode='c'] idx_array):
    # note that an empty heap is full of infinities
    # check whether we need val in the heap
    cdef ITYPE_t i, ic1, ic2, i_tmp
    cdef DTYPE_t d_tmp

    cdef ITYPE_t heap_size = heap.size

    if val > heap[0]:
        return
    
    # insert val at position zero
    heap[0] = val
    idx_array[0] = i_val

    i = 0
    while 1:
        ic1 = 2 * i + 1
        ic2 = 2 * i + 2

        if ic1 >= heap_size:
            break
        elif ic2 >= heap_size:
            if heap[ic1] > heap[i]:
                i_swap = ic1
            else:
                break
        elif heap[ic1] >= heap[ic2]:
            if heap[i] < heap[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if heap[i] < heap[ic2]:
                i_swap = ic2
            else:
                break

        d_tmp = heap[i]
        heap[i] = heap[i_swap]
        heap[i_swap] = d_tmp
        
        i_tmp = idx_array[i]
        idx_array[i] = idx_array[i_swap]
        idx_array[i_swap] = i_tmp
    
        i = i_swap
