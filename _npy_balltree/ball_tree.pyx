# cython: profile=True

# Author: Jake Vanderplas <vanderplas@astro.washington.edu>
# License: BSD

"""
Ball Tree
=========

Description
-----------

Examples
--------

Implementation Notes
--------------------

A ball tree is a data object which speeds up nearest neighbor
searches in high dimensions (see scikit-learn neighbors module
documentation for an overview of neighbor trees). There are many 
types of ball trees.  This package provides a basic implementation 
in cython.  

A ball tree can be thought of as a collection of nodes.  Each node
stores a centroid, a radius, and the pointers to two child nodes.

* centroid : the centroid of a node is the mean of all the locations 
    of points within the node
* radius : the radius of a node is the distance from the centroid
    to the furthest point in the node.
* subnodes : each node has a maximum of 2 child nodes.  The data within
    the parent node is divided between the two child nodes.

In a typical tree implementation, nodes may be classes or structures which
are dynamically allocated as needed.  This offers flexibility in the number
of nodes, and leads to very straightforward and readable code.  It also means
that the tree can be dynamically augmented or pruned with new data, in an
in-line fashion.  This approach generally leads to recursive code: upon
construction, the head node constructs its child nodes, the child nodes
construct their child nodes, and so-on.

The current package uses a different approach: all node data is stored in
a set of numpy arrays which are pre-allocated.  The main advantage of this
approach is that the whole object can be quickly and easily saved to disk
and reconstructed from disk.  This also allows for an iterative interface
which gives more control over the heap, and leads to speed.  There are a
few disadvantages, however: once the tree is built, augmenting or pruning it
is not as straightforward.  Also, the size of the tree must be known from the
start, so there is not as much flexibility in building it.

Because understanding a ball tree is simpler with recursive code, here is some
pseudo-code to show the structure of the main functionality

    # Ball Tree pseudo code

    class Node:
        #class data:
        centroid
        radius
        child1, child2

        #class methods:
        def construct(data):
            centroid = compute_centroid(data)
            radius = compute_radius(centroid, data)

            # Divide the data into two approximately equal sets.
            # This is often done by splitting along a single dimension.  
            data1, data2 = divide(data)
            
            if number_of_points(data1) > 0:
                child1.construct(data1)

            if number_of_points(data2) > 0:
                child2.construct(data2)

        def query(pt, neighbors_heap):
            # compute the minimum distance from pt to any point in this node
            d = distance(point, centroid)
            if d < radius:
                min_distance = 0
            else:
                min_distance = d - radius
            
            if min_distance > max_distance_in(neighbors_heap):
                # all these points are too far away.  cut off the search here
                return
            elif node_size > 1:
                child1.query(pt, neighbors_heap)
                child2.query(pt, neighbors_heap)


    object BallTree:
        #class data:
        data
        root_node
        def construct(data, num_leaves):
            root_node.construct(data)

        def query(point, num_neighbors):
            neighbors_heap = empty_heap_of_size(num_neighbors)
            root_node.query(point, neighbors_heap)
                
This certainly is not a complete description, but should give the basic idea
of the form of the algorithm.  The implementation below is much faster than
anything mimicking the pseudo-code above, but for that reason is much more 
opaque.  Here's the basic idea.

Given input data of size `(n_samples, n_features)`, BallTree computes the
expected number of nodes `n_nodes`, an allocates the following arrays:

* `data` : a float array of shape `(n_samples, n_features)
    This is simply the input data.  If the input matrix is well-formed
    (contiguous, c-ordered, correct data type) then no copy is needed
* `idx_array` : an integer array of size `n_samples`
    This can be thought of as an array of pointers to the data in `data`.
    Rather than shuffling around the data itself, we shuffle around pointers
    to the rows in data.
* `node_float_arr` : a float array of shape (n_nodes, n_features + 1)
    This stores the floating-point information associated with each node.
    For a node of index `i_node`, the node centroid is stored at
    `node_float_arr[i_node, :n_features]` and the node radius is stored at
    `node_float_arr[n_features]`.
* `node_int_arr` : an integer array of shape (n_nodes, 3)
    This stores the integer information associated with each node.  For
    a node of index `i_node`, the following variables are stored:
    - `idx_start = node_int_arr[i_node, 0]`
    - `idx_end = node_int_arr[i_node, 1]`
    - `is_leaf = node_int_arr[i_node, 2]`
    `idx_start` and `idx_end` point to locations in `idx_array` that reference
    the data in this node.  That is, the points within the current node are
    given by `data[idx_array[idx_start:idx_end]]`.
    `is_leaf` is a boolean value which tells whether this node is a leaf: that
    is, whether or not it has children.

You may notice that there are no pointers from parent nodes to child nodes and
vice-versa.  This is implemented implicitly:  For a node with index `i`, the
two children are found at indices `2 * i + 1` and `2 * i + 2`, while the
parent is found at index `floor((i - 1) / 2)`.  The root node, of course,
has no parent.

With this data structure in place, we can implement the functionality of the
BallTree pseudo-code spelled-out above, throwing in a few clever tricks to
make it efficient. Most of the data passing done in this code uses raw data 
pointers.  Using numpy arrays would be preferable for safety, but the 
overhead of array slicing and sub-array construction leads to execution 
time which is several orders of magnitude slower than the current 
implementation.
"""

import numpy as np
cimport numpy as np

cimport cython
cimport stdlib

#define data type
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

#define integer/index type
ITYPE = np.uint32
ctypedef np.uint32_t ITYPE_t

#define boolean type
BOOL = np.int32
ctypedef np.int32_t BOOL_t

cdef DTYPE_t infinity = np.inf

@cython.profile(False)
cdef inline DTYPE_t dmax(DTYPE_t x, DTYPE_t y):
    if x >= y:
        return x
    else:
        return y

@cython.profile(False)
cdef inline DTYPE_t dmin(DTYPE_t x, DTYPE_t y):
    if x <= y:
        return x
    else:
        return y

@cython.profile(False)
cdef inline DTYPE_t dabs(DTYPE_t x):
    if x >= 0:
        return x
    else:
        return -x

@cython.cdivision(True)
cdef DTYPE_t dist(DTYPE_t *x1, DTYPE_t *x2, ITYPE_t n, DTYPE_t p):
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
            r = dmax(r, dabs(x1[i] - x2[i]))
    elif p == 1:
        for i from 0 <= i < n:
            r += dabs(x1[i] - x2[i])
    else:
        for i from 0 <= i < n:
            d = dabs(x1[i] - x2[i])
            r += d ** p
        r = r ** (1. / p)
    return r

# dist_p returns d^p
# in order to recover the true distance, call dist_from_dist_p()
@cython.cdivision(True)
cdef DTYPE_t dist_p(DTYPE_t *x1, DTYPE_t *x2, ITYPE_t n, DTYPE_t p):
    cdef ITYPE_t i
    cdef DTYPE_t r, d
    r = 0
    if p == 2:
        for i from 0 <= i < n:
            d = x1[i] - x2[i]
            r += d * d
    elif p == infinity:
        for i from 0 <= i < n:
            r = dmax(r, dabs(x1[i] - x2[i]))
    elif p == 1:
        for i from 0 <= i < n:
            r += dabs(x1[i] - x2[i])
    else:
        for i from 0 <= i < n:
            d = dabs(x1[i] - x2[i])
            r += d ** p
    return r

@cython.cdivision(True)
cdef DTYPE_t dist_from_dist_p(DTYPE_t r, DTYPE_t p):
    if p == 2:
        return r ** 0.5
    elif p == infinity:
        return r
    elif p == 1:
        return r
    else:
        return r ** (1. / p)


######################################################################
# stack.  This is used to keep track of the stack in Node_query
#
cdef struct stack_item:
    DTYPE_t dist_LB
    ITYPE_t i_node


cdef struct stack:
    int n
    stack_item* heap
    int size


@cython.profile(False)
cdef inline void stack_create(stack* self, int size):
    self.size = size
    self.heap = <stack_item*> stdlib.malloc(sizeof(stack_item) * size)
    self.n = 0


@cython.profile(False)
cdef inline void stack_destroy(stack* self):
    stdlib.free(self.heap)


@cython.profile(False)
cdef inline void stack_resize(stack* self, int new_size):
    if new_size < self.n:
        raise ValueError("new_size smaller than current")
    self.size = new_size
    self.heap = <stack_item*>stdlib.realloc(<void*> self.heap,
                                            new_size * sizeof(stack_item))

@cython.profile(False)
cdef inline void stack_push(stack* self, stack_item item):
    if self.n >= self.size:
        stack_resize(self, 2 * self.size + 1)

    self.heap[self.n] = item
    self.n += 1

@cython.profile(False)
cdef inline stack_item stack_pop(stack* self):
    if self.n == 0:
        raise ValueError("popping empty stack")
    
    self.n -= 1
    return self.heap[self.n]
    
######################################################################
# estimate_num_nodes
#  This is an estimate of the number of nodes needed given the number
#  of input data samples and the size of a leaf node.  Though the exact
#  value could be calculated explicitly, this gives an empirically
#  determined upper-bound, as long as any node with an even number of
#  points puts exactly half in each child node, and any node with an odd
#  number of points puts (N+1)/2 in the first child (with index 2*i+1) and
#  (N-1)/2 in the second child (with index 2*i+2)
#
#  For leaf_size ~ 20, the estimate leads to very little wasted space.
#  For leaf_size near 1, and for a near integer log2(n_samples / leaf_size),
#  the wasted space can be more significant.
cdef inline ITYPE_t estimate_num_nodes(ITYPE_t n_samples,
                                       ITYPE_t leaf_size):
    return 2 ** (1 + np.ceil(np.log2((n_samples + leaf_size - 1)
                                     / leaf_size))) - 1


######################################################################
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

        assert p >= 1
        
        cdef ITYPE_t n_samples = self.data.shape[0]
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef ITYPE_t n_nodes = estimate_num_nodes(n_samples, leaf_size)

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

        X = np.asarray(X, dtype=DTYPE, order='C')

        cdef ITYPE_t i
        cdef ITYPE_t N
        cdef np.ndarray distances = np.empty((X.shape[0], k), dtype=DTYPE)
        cdef np.ndarray idx_array = np.empty((X.shape[0], k), dtype=ITYPE)
        cdef np.ndarray Xi

        distances[:] = np.inf

        cdef DTYPE_t* dist_ptr = <DTYPE_t*> distances.data
        cdef ITYPE_t* idx_ptr = <ITYPE_t*> idx_array.data
        cdef ITYPE_t n_samples_X = X.shape[0]
        cdef ITYPE_t n_samples = self.data.shape[0]
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef ITYPE_t n_neighbors = k

        #FIXME: better estimate of stack size
        cdef stack node_stack
        stack_create(&node_stack, n_samples)

        for i from 0 <= i < n_samples_X:
            Xi = X[i]
            Node_query(<DTYPE_t*>Xi.data, self.p, k, n_features,
                       dist_ptr, idx_ptr,
                       <DTYPE_t*>self.data.data,
                       <ITYPE_t*>self.idx_array.data,
                       <DTYPE_t*>self.node_float_arr.data,
                       <ITYPE_t*>self.node_int_arr.data,
                        &node_stack)
            dist_ptr += n_neighbors
            idx_ptr += n_neighbors
        
        stack_destroy(&node_stack)
            
        # deflatten results
        if return_distance:
            return (distances.reshape((orig_shape[:-1]) + (k,)),
                    idx_array.reshape((orig_shape[:-1]) + (k,)))
        else:
            return idx_array.reshape((orig_shape[:-1]) + (k,))
        
@cython.cdivision(True)
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
                      dist_p(centroid, data + n_features * idx_array[i],
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
                              dist_p(centroid,
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

@cython.profile(False)
cdef inline void copy_array(DTYPE_t* x, DTYPE_t* y, ITYPE_t n):
    # copy array y into array x
    cdef ITYPE_t i
    for i from 0 <= i < n:
        x[i] = y[i]

@cython.cdivision(True)
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


@cython.profile(False)
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

@cython.profile(False)
cdef inline DTYPE_t calc_dist_LB(DTYPE_t* pt,
                                 DTYPE_t* centroid,
                                 ITYPE_t n_features,
                                 DTYPE_t p):
    return dmax(0, (dist(pt, centroid, n_features, p)
                    - centroid[n_features]))


cdef void Node_query(DTYPE_t* pt,
                     ITYPE_t p, ITYPE_t k,
                     ITYPE_t n_features,
                     DTYPE_t* near_set_dist,
                     ITYPE_t* near_set_indx,
                     DTYPE_t* data,
                     ITYPE_t* idx_array,
                     DTYPE_t* node_float_arr,
                     ITYPE_t* node_int_arr,
                     stack* node_stack):
    cdef DTYPE_t dist_pt, dist_LB, dist_LB_1, dist_LB_2
    cdef ITYPE_t i, i1, i2, i_node, idx_start, idx_end

    cdef ITYPE_t* node_info = node_int_arr

    cdef stack_item item

    item.i_node = 0
    item.dist_LB = calc_dist_LB(pt, node_float_arr,
                                n_features, p)
    stack_push(node_stack, item)

    while(node_stack.n > 0):        
        item = stack_pop(node_stack)
        i_node = item.i_node
        dist_LB = item.dist_LB

        node_info = node_int_arr + i_node * 3

        idx_start = node_info[0]
        idx_end = node_info[1]

        #------------------------------------------------------------
        # Case 1: query point is outside node radius
        #if dist_LB >= max_heap_largest(near_set_dist):
        #    continue
        if dist_LB >= pqueue_largest(near_set_dist, k):
            continue

        #------------------------------------------------------------
        # Case 2: this is a leaf node.  Update set of nearby points
        elif node_info[2]:
            for i from idx_start <= i < idx_end:
                dist_pt = dist(pt,
                               data + n_features * idx_array[i],
                               n_features, p)

                #if dist_pt < max_heap_largest(near_set_dist):
                #    max_heap_insert(dist_pt, idx_array[i],
                #                    near_set_dist, near_set_indx, k)
                if dist_pt < pqueue_largest(near_set_dist, k):
                    pqueue_insert(dist_pt, idx_array[i],
                                  near_set_dist, near_set_indx, k)

        #------------------------------------------------------------
        # Case 3: Node is not a leaf.  Recursively query subnodes
        #         starting with the one whose centroid is closest
        else:
            i1 = 2 * i_node + 1
            i2 = i1 + 1
            dist_LB_1 = calc_dist_LB(pt, (node_float_arr
                                          + i1 * (n_features + 1)),
                                     n_features, p)
            dist_LB_2 = calc_dist_LB(pt, (node_float_arr
                                          + i2 * (n_features + 1)),
                                     n_features, p)


            # append children to stack: last-in-first-out
            if dist_LB_2 <= dist_LB_1:
                item.i_node = i1
                item.dist_LB = dist_LB_1
                stack_push(node_stack, item)
                
                item.i_node = i2
                item.dist_LB = dist_LB_2
                stack_push(node_stack, item)

            else:
                item.i_node = i2
                item.dist_LB = dist_LB_2
                stack_push(node_stack, item)
                
                item.i_node = i1
                item.dist_LB = dist_LB_1
                stack_push(node_stack, item)


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
@cython.profile(False)
cdef inline DTYPE_t max_heap_largest(DTYPE_t* heap):
    return heap[0]


cdef void max_heap_insert(DTYPE_t val, ITYPE_t i_val,
                          DTYPE_t* heap,
                          ITYPE_t* idx_array,
                          ITYPE_t heap_size):
    cdef ITYPE_t i, ic1, ic2, i_tmp
    cdef DTYPE_t d_tmp

    # check if val should be in heap
    if val > heap[0]:
        return
    
    # insert val at position zero
    heap[0] = val
    idx_array[0] = i_val

    i = 0
    while 1:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= heap_size:
            break
        elif ic2 >= heap_size:
            if heap[ic1] > val:
                i_swap = ic1
            else:
                break
        elif heap[ic1] >= heap[ic2]:
            if val < heap[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if val < heap[ic2]:
                i_swap = ic2
            else:
                break

        heap[i] = heap[i_swap]
        idx_array[i] = idx_array[i_swap]
    
        i = i_swap
    
    heap[i] = val
    idx_array[i] = i_val

#------------------------------------------------------------
# alternative: priority queue

@cython.profile(False)
cdef inline DTYPE_t pqueue_largest(DTYPE_t* queue, ITYPE_t queue_size):
    return queue[queue_size - 1]

cdef inline void pqueue_insert(DTYPE_t val, ITYPE_t i_val,
                               DTYPE_t* queue, ITYPE_t* idx_array,
                               ITYPE_t queue_size):
    cdef ITYPE_t i_lower = 0
    cdef ITYPE_t i_upper = queue_size - 1
    cdef ITYPE_t i_mid
    cdef int i #unsigned leads to problems in the for-loop
    
    if val >= queue[i_upper]:
        return
    elif val <= queue[i_lower]:
        i_mid = i_lower
    else:
        while True:
            if (i_upper - i_lower) < 2:
                i_mid = i_lower + 1
                break
            else:
                i_mid = (i_lower + i_upper) / 2

            if i_mid == i_lower:
                i_mid += 1
                break

            if val >= queue[i_mid]:
                i_lower = i_mid
            else:
                i_upper = i_mid

    for i from queue_size - 1 > i >= i_mid:
        if i > queue_size:
            break
        queue[i + 1] = queue[i]
        idx_array[i + 1] = idx_array[i]

    queue[i_mid] = val
    idx_array[i_mid] = i_val
            
