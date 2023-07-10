import numpy as np

def matrix_block_inversion(Ainv,B,C,D):
    '''
    This function performs the inverted of a matrix X, defined by blocks as:
    
            X = [A B]
                [C D]
    
    The algorithm performed is Block Matrix Inversion.
    (refer to https://arxiv.org/pdf/2305.11103.pdf for more information)

    This algorithm requires A and its schur complement S(A) = D-CA'B to be invertible.
    If not, one could use D and its schur complement S(D) = A-BD'C instead, being sure those are invertible.
    In this case call matrix_block_inversion_D instead.

    If A and its Schur complement S are invertible, then the inverse of X is:

            X'= [  A' + A'BS'CA'   -A'BS'  ]
                [     -S'CA'         S'    ]
    
    The advantage is that only the inverse of A and S are required.
    This is convenient when A' is already known.

    For example when expanding the kernel matrix in Gaussian Processes techniques.
    (see https://github.com/MBonalumi/Batch-Incremental-Gaussian-Process-Regression-BIGPR)
    '''

    assert Ainv.shape[0] == Ainv.shape[1]
    assert D.shape[0] == D.shape[1]
    assert B.shape[0] == Ainv.shape[0] and B.shape[1] == D.shape[0]
    assert C.shape[0] == D.shape[0] and C.shape[1] == Ainv.shape[0]


    S = D - C @ Ainv @ B
    Sinv = np.linalg.inv(S)

    Xinv00 = Ainv + Ainv @ B @ Sinv @ C @ Ainv
    Xinv01 = -Ainv @ B @ Sinv
    Xinv10 = -Sinv @ C @ Ainv
    Xinv11 = Sinv

    Xinv = np.block([   [Xinv00, Xinv01],
                        [Xinv10, Xinv11]   ])

    return Xinv


def matrix_block_inversion_D(A,B,C,Dinv):
    '''
    This function performs the inverted of a matrix X, defined by blocks as:
    
            X = [A B]
                [C D]
    
    The algorithm performed is Block Matrix Inversion.
    (refer to https://arxiv.org/pdf/2305.11103.pdf for more information)

    This algorithm requires D and its schur complement S(D) = A-BD'C to be invertible.
    If not, one could use A and its schur complement S(A) = D-CA'B instead, being sure those are invertible.
    In this case call matrix_block_inversion instead.

    If D and its Schur complement S are invertible, then the inverse of X is:

            X'= [      S'        -S'BD'     ]
                [   -D'CS'     D'+D'CS'BD'  ]
    
    The advantage is that only the inverse of D and S are required.
    This is convenient when D' is already known.
    '''

    assert A.shape[0] == A.shape[1]
    assert Dinv.shape[0] == Dinv.shape[1]
    assert B.shape[0] == A.shape[0] and B.shape[1] == Dinv.shape[0]
    assert C.shape[0] == Dinv.shape[0] and C.shape[1] == A.shape[0]


    S = A - B @ Dinv @ C
    Sinv = np.linalg.inv(S)

    Xinv00 = Sinv
    Xinv01 = -Sinv @ B @ Dinv
    Xinv10 = -Dinv @ C @ Sinv
    Xinv11 = Dinv + Dinv @ C @ Sinv @ B @ Dinv

    Xinv = np.block([   [Xinv00, Xinv01],
                        [Xinv10, Xinv11]   ])

    return Xinv
