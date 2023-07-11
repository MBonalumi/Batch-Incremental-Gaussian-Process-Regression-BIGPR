import numpy as np

def matrix_inverse_remove_i(A, i):
    '''
    This function calculates the inverse of a matrix after removing the i-th row and column.

    Given A, which is the inversion of a matrix A, we want to calculate the inversion of A after removing the i-th row and column.
    This can be done with a complexity of O(n^2) instead of O(n^3) by using the following formula:

    1. transpose the i-th row and column to the last row and column (n-th)
    2. consider the matrix to be defined by blocks:
        A   =   [ a, b ]
                [ c, d ]
        where b (n-1 x 1), c (1 x n-1), d (1 x 1) compose the row and column to be removed, and a (n-1 x n-1) is the remaining part of the matrix.
    3. the inversion of the matrix A after removing i-th column and row is given by:
        A\i_inv =  a - b * d^-1 * c

    Parameters
    ----------
    A: numpy.ndarray
        The inverted matrix of the full original matrix A.
    i : int
        The index of the row and column to be removed.

    Returns
    -------
    numpy.ndarray
        The inverted matrix of the original matrix A after removing the i-th row and column.

        
    Please refer to this link for the complete explanation and proof:
    https://stats.stackexchange.com/questions/450146/updating-the-inverse-covariance-matrix-after-deleting-the-i-th-column-and-row-of
    '''

    n = A.shape[0]
    assert A.shape[1] == n, "Ainv must be a square matrix"

    # return a - np.outer(b, c)/d

    # transpose the i-th row and column with the last row and column (n-th)
    A[[i, n-1], :] = A[[n-1, i], :]
    A[:, [i, n-1]] = A[:, [n-1, i]]

    # consider the matrix to be defined by blocks:
    # Ainv =  [ a, b ]
    #         [ c, d ]
    # where b (n-1 x 1), c (1 x n-1), d (1 x 1) compose the row and column to be removed, and a (n-1 x n-1) is the remaining part of the matrix.
    a = A[:n-1, :n-1]
    b = A[:n-1, n-1]
    c = A[n-1, :n-1]
    d = A[n-1, n-1]

    # the inversion of the matrix A after removing i-th column and row is given by:
    # A\i_inv =  a - b * d^-1 * c
    Ainv_new = a - np.outer(b, c)/d

    # now we need to transpose back the i-th row and column to the correct position
    indices = np.arange(n-1)    # n-1 because we removed one row/col
    to_roll = indices[i:]
    to_roll = np.roll(to_roll, -1)
    indices2 = indices.copy()
    indices2[i:] = to_roll
    Ainv_new[:, indices] = Ainv_new[:, indices2]
    Ainv_new[indices, :] = Ainv_new[indices2, :]

    #and we are done

    return Ainv_new


def matrix_inverse_remove_indices(A, i_s):
    '''
    This function calculates the inverse of a matrix after removing the all rows and columns of indices i_s.

    Given A, which is the inversion of a matrix A, we want to calculate the inversion of A after removing rows and columns of indices i_s.
    This can be done with a complexity of O(n^2) instead of O(n^3) by using the following formula:

    1.  order i_s in descending order
    2.  call x_s all As indices
    3.  np.roll of len(A) - i_s[-1]  
        (or i_s[-h] - i_s[-h-1] after the first step)
    4.  consider the matrix to be defined by blocks:
            A   =   [ a, b ]
                    [ c, d ]
    5.  A = a
        stack bd with old ones (if any)
        stack c  with old ones (if any) 
    6. now repeat step 2 for each i in i_s
    7. the inversion of the matrix A after removing i-th column and row is given by:
        A\i_inv =  a - b * d^-1 * c

    Parameters
    ----------
    A :     numpy.ndarray
            The inverted matrix of the full original matrix A.
    i_s :   numpy ndarray
            The indices of the rows and columns to be removed.

    Returns
    -------
    numpy.ndarray
        The inverted matrix of the original matrix A after removing the i-th row and column.

        
    Please refer to this link for the complete explanation and proof:
    https://stats.stackexchange.com/questions/450146/updating-the-inverse-covariance-matrix-after-deleting-the-i-th-column-and-row-of
    '''
    # check A square matrix
    n = A.shape[0]
    assert A.shape[1] == n, "Ainv must be a square matrix"

    # order i_s in descending order
    assert len(i_s) > 0, "i_s must be a non-empty list"
    i_s = np.sort(np.array(i_s))[::-1]

    # call x_s all As indices
    x_s = np.arange(A.shape[0])
    adj_v = np.zeros(len(x_s), np.int32)

    swap = -1
    for i in i_s:
        x_s[[i, swap]] = x_s[[swap, i]]
        swap -= 1

    A[:, :] = A[x_s, :]
    A[:, :] = A[:, x_s]

    amt = i_s.shape[0]
    a = A[:-amt, :-amt]
    b = A[:-amt, -amt:]
    c = A[-amt:, :-amt]
    d = A[-amt:, -amt:]

    dinv = np.linalg.inv(d)

    result = a - np.matmul(b, np.matmul(dinv, c))

    for i in i_s:
        adj_v[x_s > i] += 1

    x_s-=adj_v
    x_s = x_s[:-amt]

    result[x_s,:] = result[np.arange(result.shape[0]), :]
    result[:,x_s] = result[:, np.arange(result.shape[0])]

    return result