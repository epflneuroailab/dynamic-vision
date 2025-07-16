import numpy as np
from joblib import Parallel, delayed
from collections import OrderedDict
from brainscore_vision.model_helpers.activations.pca import flatten
from sklearn.random_projection import _check_density, _check_input_size, check_random_state, sample_without_replacement
import scipy.sparse as sp
import numpy as np
from numba import njit, prange
from scipy.sparse import csc_matrix

from .groups import IMAGE_MODELS

@njit(parallel=True)
def dense_sparse_mult_csc(B, A_data, A_indices, A_indptr):
    rows_B, cols_B = B.shape
    rows_A = len(A_indices)
    cols_A = len(A_indptr) - 1
    
    # Initialize the result matrix with zeros
    result = np.zeros((rows_B, cols_A))
    
    # Iterate over the columns of the sparse matrix A
    for j in prange(cols_A):
        start_idx = A_indptr[j]
        end_idx = A_indptr[j + 1]
        
        for i in range(start_idx, end_idx):
            row_A = A_indices[i]
            value_A = A_data[i]
            
            # Multiply the corresponding row in B with the non-zero value in A
            for k in range(rows_B):
                result[k, j] += B[k, row_A] * value_A
    
    return result

def dense_sparse_mult(B, A):
    """
    Wrapper function to multiply a dense matrix B with a sparse csc_matrix A.

    Args:
        B (np.ndarray): The dense matrix.
        A (csc_matrix): The sparse matrix in CSC format.

    Returns:
        np.ndarray: The result of B * A.
    """
    if not isinstance(A, csc_matrix):
        raise TypeError("Matrix A must be a scipy.sparse.csc_matrix")
    if not isinstance(B, np.ndarray):
        raise TypeError("Matrix B must be a numpy.ndarray")

    A = A.astype(np.float32)
    B = B.astype(np.float32)

    # Extract the sparse matrix components
    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr
    
    # Perform the multiplication using the optimized Numba function
    return dense_sparse_mult_csc(B, A_data, A_indices, A_indptr).astype(np.float16)


CACHE = {}

mapper = Parallel(temp_folder='/tmp/joblib_memmapping_folder', n_jobs=-1, verbose=False, backend="multiprocessing")

def _compute(batch, matrix):
    return batch @ matrix.T

_compute_delayed = delayed(_compute)

def sparse_random_matrix(n_components, n_features, density='auto',
                          random_state=42):
    """Generalized Achlioptas random sparse matrix for random projection

    Setting density to 1 / 3 will yield the original matrix by Dimitris
    Achlioptas while setting a lower value will yield the generalization
    by Ping Li et al.

    If we note :math:`s = 1 / density`, the components of the random matrix are
    drawn from:

      - -sqrt(s) / sqrt(n_components)   with probability 1 / 2s
      -  0                              with probability 1 - 1 / s
      - +sqrt(s) / sqrt(n_components)   with probability 1 / 2s

    Read more in the :ref:`User Guide <sparse_random_matrix>`.

    Parameters
    ----------
    n_components : int,
        Dimensionality of the target projection space.

    n_features : int,
        Dimensionality of the original source space.

    density : float in range ]0, 1] or 'auto', optional (default='auto')
        Ratio of non-zero component in the random projection matrix.

        If density = 'auto', the value is set to the minimum density
        as recommended by Ping Li et al.: 1 / sqrt(n_features).

        Use density = 1 / 3.0 if you want to reproduce the results from
        Achlioptas, 2001.

    random_state : int, RandomState instance or None, optional (default=None)
        Controls the pseudo random number generator used to generate the matrix
        at fit time.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    components : array or CSR matrix with shape [n_components, n_features]
        The generated Gaussian random matrix.

    See Also
    --------
    SparseRandomProjection

    References
    ----------

    .. [1] Ping Li, T. Hastie and K. W. Church, 2006,
           "Very Sparse Random Projections".
           https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf

    .. [2] D. Achlioptas, 2001, "Database-friendly random projections",
           http://www.cs.ucsc.edu/~optas/papers/jl.pdf

    """

    if random_state in CACHE:
        return CACHE[random_state]

    _check_input_size(n_components, n_features)
    density = _check_density(density, n_features)
    rng = check_random_state(random_state)

    if density == 1:
        # skip index generation if totally dense
        components = rng.binomial(1, 0.5, (n_components, n_features)) * 2 - 1
        matrix = 1 / np.sqrt(n_components) * components

    else:
        # Generate location of non zero elements
        indices = []
        offset = 0
        indptr = [offset]
        for _ in range(n_components):
            # find the indices of the non-zero components for row i
            n_nonzero_i = rng.binomial(n_features, density)
            indices_i = sample_without_replacement(n_features, n_nonzero_i,
                                                   random_state=rng)
            indices.append(indices_i)
            offset += n_nonzero_i
            indptr.append(offset)

        indices = np.concatenate(indices)

        # Among non zero components the probability of the sign is 50%/50%
        data = rng.binomial(1, 0.5, size=np.size(indices)) * 2 - 1

        # build the CSR structure by concatenating the rows
        components = sp.csr_matrix((data, indices, indptr),
                                   shape=(n_components, n_features))

        matrix = np.sqrt(1 / density) / np.sqrt(n_components) * components

    CACHE[random_state] = matrix
    return matrix

def get_projection(source, target_size, seed, parallel=True, batch_size=4):
    # generate a random projection, but fix it for a given source_size and target_size
    source_size = source.shape[1]
    if target_size >= source_size:
        return source

    matrix = sparse_random_matrix(target_size, source_size, random_state=seed)
    
    # import time
    # a = time.time()
    # print(f"Source size: {source_size}")
    if True:
        ret = dense_sparse_mult(source, matrix.T)
    elif source_size < 50_000:
        ret = _compute(source, matrix)
    else:
        batches = [source[i:i+batch_size] for i in range(0, source.shape[0], batch_size)]
        ret = mapper(_compute_delayed(batch, matrix) for batch in batches)
        ret = np.concatenate(ret, axis=0)
    # b = time.time()
    # print(f"Time: {b-a}")
    return ret

class RandomProjection:
    def __init__(self, activations_extractor, n_components):
        self.handle = True
        self.n_components = n_components
        self._extractor = activations_extractor
        self._inferencer = activations_extractor.inferencer

    def __call__(self, activations, layer_name, stimulus):
        if self.handle:
            model_id = self._extractor._identifier
            dtype = activations.dtype
            layer_spec = self._inferencer.layer_activation_format[layer_name]

            if "T" in layer_spec:
                T_index = layer_spec.index("T")
                # move T to front, consistent with the temporal inferencer
                activations = np.moveaxis(activations, T_index, 0)  
            else:
                activations = np.expand_dims(activations, 0)

            def apply_random_proj(activations):
                activations = flatten(activations)
                source_size = activations.shape[1]
                target_size = self.n_components
                if isinstance(self.n_components, float):
                    target_size = int(source_size * self.n_components)
                seed = hash(self._extractor.identifier+"."+layer_name) % 2**32
                projection = get_projection(activations, target_size, seed)
                return projection

            activations = apply_random_proj(activations)
            
            # reduce all the other channels to size 1
            if "T" in layer_spec:
                T = activations.shape[0]
                num_channels = len(layer_spec) - 2  # already T, C
                C_index = layer_spec.index("C")
                channel_indicies = list(range(len(layer_spec)))
                channel_indicies.remove(T_index)
                channel_indicies.remove(C_index)
                n_components = activations.shape[1]
                activations = activations.reshape(T, n_components, *([1] * num_channels))
                activations = np.transpose(activations, [T_index, C_index] + channel_indicies)

            else:
                activations = activations[0]
                num_channels = len(layer_spec) - 1  # already C
                C_index = layer_spec.index("C")
                channel_indicies = list(range(len(layer_spec)))
                channel_indicies.remove(C_index)
                n_components = activations.shape[0]
                activations = activations.reshape(n_components, *([1] * num_channels))
                activations = np.transpose(activations, [C_index] + channel_indicies)
            
            activations = activations.astype(dtype)

        return activations
    

def _register_downsampling_hook(model, downsample_to=5000):
    print(f"Downsample model activations to {downsample_to} using sparse random projection.")
    extractor = model.activations_model._extractor
    layer_random_proj = RandomProjection(extractor, n_components=downsample_to)
    extractor.inferencer._executor.register_after_hook(layer_random_proj)
    return model