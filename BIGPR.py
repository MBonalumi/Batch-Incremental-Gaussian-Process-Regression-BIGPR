
import numpy as np
import csv
from collections import deque
import random
import time
from matrix_block_inversion import matrix_block_inversion
from matrix_inverse_remove import matrix_inverse_remove_i, matrix_inverse_remove_indices

class HyperParam(object):
    def __init__(self, theta_f=1, len=1, theta_n=0.1):
        self.theta_f = theta_f       # for squared exponential kernel
        self.len = len           # for squared exponential kernel
        self.theta_n = theta_n     # for squared exponential kernel

class BIGPR(object):
    '''
    This project implements the Incremental Gaussian Process Regression (IGPR) algorithm adding support for batch learning.
    Training a model via learning one sample at a time is also supported
    All the code involved has been mostly rewritten to improve performance and optimization,
    exploiting the vectorial capabilities of numpy and the use of matrix inversion lemmas.

    Learning batches and learning single samples might not provide the same k_matrix
    '''
    def __init__(self, init_x, init_y):
        self.hyperparam = HyperParam(1, 1, 0.1)
        self.max_k_matrix_size = 400
        self.lamda = 1
        self.count = 0
        self.kernel_x = deque()
        self.kernel_y = deque()
        self.kernel_x.append(init_x)
        self.kernel_y.append(init_y)
        self.k_matrix = np.ones((1, 1)) + self.hyperparam.theta_n * self.hyperparam.theta_n
        self.inv_k_matrix = np.ones((1, 1)) / (self.hyperparam.theta_n * self.hyperparam.theta_n)
        self.is_av = False
        temp = np.sum(self.k_matrix, axis=0)
        self.delta = deque()
        for i in range(temp.shape[0]):
            self.delta.append(temp[i])
        
        # self.informativity = deque().append(0.)        # i.e. max covariance wrt other samples
        self.info_mat = deque().append(0.)             # i.e. covariance wrt other samples, ordered, excluding self

        self.samples_substituted_count = 0
        self.samples_substituted = []


    def is_available(self):
        if not self.is_av:
            self.is_av = len(self.kernel_x) >= 2
        return self.is_av


    def load_csv(self, file_name):
        with open(file_name, "r") as f:
            reader = csv.reader(f)
            columns = [row for row in reader]
        columns = np.array(columns)
        m_x, n_x = columns.shape
        data_set = np.zeros((m_x,n_x))
        for i in range(m_x):
            for j in range(n_x):
                data_set[i][j] = float(columns[i][j])
        return data_set


    def learn(self, new_x, new_y):
        self.delta = deque(np.array(self.delta) * self.lamda)

        if not self.is_available():
            self.kernel_x.append(new_x)
            self.kernel_y.append(new_y)
            self.calculate_SE_kernel()
            self.inv_k_matrix = np.linalg.inv(self.k_matrix)

        elif len(self.kernel_x) < self.max_k_matrix_size:
            self.aug_update_SE_kernel(new_x, new_y)

        else:
            # call the same as the batch method because more optimized and gives same result
            self.aug_update_SE_kernel(new_x, new_y)
            self.remove_kernel_samples(1)
            # OLD WAY -> slightly less efficient
            # self.sub_kernel_sample(new_x, new_y)
            

    def learn_batch(self, new_xs, new_ys):
        self.delta = deque(np.array(self.delta) * self.lamda)
        #TODO: find a way to screen new_xs instead of adding blankly, to avoid unnecessary computation

        if not self.is_available():
            self.kernel_x.extend(new_xs)
            self.kernel_y.extend(new_ys)
            self.calculate_SE_kernel()
            self.inv_k_matrix = np.linalg.inv(self.k_matrix)
            #check if max size overshoot 
            if len(self.kernel_x) > self.max_k_matrix_size:
                self.remove_kernel_samples( len(self.kernel_x) - self.max_k_matrix_size )

        # contained in last elif
        # elif len(self.kernel_x) == self.max_k_matrix_size:
        #     # if matrix already maxxed, update for amt=new_xs.shape[0]
        #     self.batch_aug_update_SE_kernel(new_xs, new_ys)
        #     self.remove_kernel_samples(new_xs.shape[0])

        elif len(self.kernel_x) + len(new_xs) < self.max_k_matrix_size:
            # if matrix + new_xs not maxxed yet, insert new samples
            self.batch_aug_update_SE_kernel(new_xs, new_ys)

        elif len(self.kernel_x) + len(new_xs) >= self.max_k_matrix_size:
            new_xs, new_ys, _, _ = self.screen_new_samples(new_xs, new_ys)
            if len(new_xs) == 0:
                return

            # otherwise, if matrix not maxxed, but will be after adding new_xs
            # add new_xs but only remove a smaller amt
            # using this formula:
            amt_toremove = len(new_xs) + len(self.kernel_x) - self.max_k_matrix_size

            #now we add all and remove to come back to the max size
            self.batch_aug_update_SE_kernel(new_xs, new_ys)
            self.remove_kernel_samples(amt_toremove)

        else:
            print("ERROR: shouldn't be here")
            exit(1)


    def calculate_SE_kernel(self, kernel_x=None, return_values=False):
        if kernel_x is None:
            kernel_x = self.kernel_x
        
        n = len(kernel_x)

        #compute kernel matrix
        k_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                k_matrix[i][j] = np.sum(np.square(kernel_x[i] - kernel_x[j]))
                k_matrix[i][j] = k_matrix[i][j] / (-2)
                k_matrix[i][j] = k_matrix[i][j] / self.hyperparam.len
                k_matrix[i][j] = k_matrix[i][j] / self.hyperparam.len
                k_matrix[i][j] = np.exp(k_matrix[i][j])
                k_matrix[i][j] = k_matrix[i][j] * self.hyperparam.theta_f
                k_matrix[i][j] = k_matrix[i][j] * self.hyperparam.theta_f
        k_matrix = k_matrix + self.hyperparam.theta_n * self.hyperparam.theta_n * np.eye(n)
        infomat = self.compute_info_mat(k_matrix)

        #compute delta
        d = np.sum(k_matrix, axis=0)
        delta = deque(d)

        #compute informativity
        # info = np.array([self.informativity_metric(np.delete(kernel_x, i), kernel_x[i]) for i in range(n)])
        # assert len(info) == n
        # informativity = deque(info)

        if return_values:
            # return k_matrix, delta, info
            return k_matrix, delta, infomat
        else:
            self.k_matrix = k_matrix
            self.info_mat = infomat
            self.delta = delta
            # self.informativity = informativity


    def compute_info_mat(self, kmat):
        infomat = deque()
        for i in range(kmat.shape[0]):
            temp = kmat[i,:].copy()
            temp[i] = 0
            infomat.append(np.argsort(temp)[::-1])
        
        return infomat


    def predict(self, coming_x):
        if self.is_available():
            k_x = np.array(self.kernel_x)
            cross_kernel_k = np.sum(np.square(k_x - coming_x), axis=1)
            cross_kernel_k /= -2 * self.hyperparam.len * self.hyperparam.len
            cross_kernel_k = np.exp(cross_kernel_k)
            cross_kernel_k *= self.hyperparam.theta_f * self.hyperparam.theta_f
            cross_kernel_k = cross_kernel_k.reshape((1, -1))

            kernel_y_mat = np.array(self.kernel_y)
            prediction = cross_kernel_k.dot(self.inv_k_matrix.dot(kernel_y_mat))
        else:
            print("Not available")
            prediction = self.kernel_y[0]
        return prediction


    def aug_update_SE_kernel(self, new_x, new_y):
        n = len(self.kernel_x)
        self.kernel_x.append(new_x)
        self.kernel_y.append(new_y)
        k_x = np.array(self.kernel_x)

        new_row = np.sum(np.square(k_x - new_x) , axis=1) 
        new_row /= (-2 * self.hyperparam.len * self.hyperparam.len)
        new_row = np.exp(new_row)
        new_row *= self.hyperparam.theta_f * self.hyperparam.theta_f
        new_row = new_row.reshape(1, -1)

        self.k_matrix = np.vstack((self.k_matrix, new_row[:,:-1]))
        self.k_matrix = np.hstack((self.k_matrix, new_row.T))

        self.k_matrix[n, n] += self.hyperparam.theta_n * self.hyperparam.theta_n

        # NB using block inversion is worse, 2.5x time

        # compute inv matrix 
        b = self.k_matrix[0:n, n].reshape((n, 1))
        d = self.k_matrix[n, n]
        e = self.inv_k_matrix.dot(b)
        g = 1 / (d - (b.T).dot(e))
        haha_11 = self.inv_k_matrix + g[0][0]*e.dot(e.T)
        haha_12 = -g[0][0]*e
        haha_21 = -g[0][0]*(e.T)
        haha_22 = g
        temp_1 = np.hstack((haha_11, haha_12))
        temp_2 = np.hstack((haha_21, haha_22))
        self.inv_k_matrix = np.vstack((temp_1, temp_2))

        
        # udpate delta        
        d += self.k_matrix[:-1, n]
        d = np.append(d, 0)
        d[n] += self.k_matrix[:, n].sum()
        self.delta = deque(d)

        # # update informativity
        # info = np.array(self.informativity)
        # new_x_info = self.get_sample_informativity(k_x[:-1], new_x)
        # info2 = self.calculate_array_sample_distance(k_x[:-1], new_x)
        # info = np.maximum(info, info2)
        # info = np.append(info, new_x_info)
        # self.informativity = deque(info)

        # update info mat
        # i can just add the row for the new value, no need to recompute the whole matrix
        # it will be like a diagonal matrix,  but since it's simmetric, it's fine
        # when removing, add both elements to the removal candidates
        # PROBLEM: might not translate to numpy

        # WHEN ADDING IN THE NOT-BATCHED, I NEED TO PUT TO 0 THE SELF-COVARIANCE TERM!
        row = new_row[0].copy()
        row[n] = 0
        new_info_row = np.argsort(row)[::-1]
        self.info_mat.append(new_info_row)


    def screen_new_samples(self, new_xs, new_ys):
        xs = new_xs.copy()
        ys = new_ys.copy()
        k_x = np.array(self.kernel_x)

        # calculate new submatrix (i.e. new_xs kernel)
        new_k_matrix, _, new_infomat = self.calculate_SE_kernel(kernel_x=xs, return_values=True)

        new_infomat_maxidx = np.array([new_infomat[i][0] for i in range(len(new_infomat))])
        new_infomat_maxval = new_k_matrix[np.arange(len(new_infomat)), new_infomat_maxidx]

        killed_idxs = []

        while xs.shape[0] > 1 and new_infomat_maxval.max() >= 0.95: #dont even bother adding if informativity is too high (0.95 but it could be even 0.8)
            kill = np.sort( new_infomat_maxidx[np.where(new_infomat_maxval == new_infomat_maxval.max())] )[::-1][0]
            killed_idxs.append(kill)
            new_k_matrix = np.delete(new_k_matrix, kill, axis=0)
            new_k_matrix = np.delete(new_k_matrix, kill, axis=1)
            new_infomat = self.compute_info_mat(new_k_matrix)
            xs = np.delete(xs, kill, axis=0)
            ys = np.delete(ys, kill, axis=0)

            new_infomat_maxidx = np.array([new_infomat[i][0] for i in range(len(new_infomat))])
            new_infomat_maxval = new_k_matrix[np.arange(len(new_infomat)), new_infomat_maxidx]

        # print worst self.info_mat values, in amount as xs.shape[0]
        self_infomat_maxidx = np.array([self.info_mat[i][0] for i in range(len(self.info_mat))])
        self_infomat_maxval = self.k_matrix[np.arange(len(self.info_mat)), self_infomat_maxidx]
        self_killable_val = np.sort(self_infomat_maxval)[:xs.shape[0]]
        # self_killable_idx = self_infomat_maxidx[np.argsort(self_infomat_maxval)[:xs.shape[0]]]

        # calculate rectangular matrix -> informativity
        xs_info = []
        for i,x in enumerate(xs):
            row = np.sum(np.square(k_x - x), axis=1) / (-2 * self.hyperparam.len * self.hyperparam.len)
            row = np.exp(row) * self.hyperparam.theta_f * self.hyperparam.theta_f
            worst_i = np.argsort(row)[::-1][0]
            xs_info.append(row[worst_i])

        dont_add = []

        for i in np.argsort(xs_info):
            if xs_info[i] > self_killable_val[-1]:
                dont_add.append(i)
            else:
                self_killable_val[-1] = xs_info[i]
                self_killable_val = np.sort(self_killable_val)

        xs = np.delete(xs, dont_add, axis=0)
        ys = np.delete(ys, dont_add, axis=0)

        return xs, ys, killed_idxs, dont_add


    def batch_aug_update_SE_kernel(self, new_xs, new_ys):
        k_x = np.array(self.kernel_x)
        self.kernel_x.extend(new_xs)
        self.kernel_y.extend(new_ys)

        # calculate new submatrix (i.e. new_xs kernel)
        new_k_matrix, _, _ = self.calculate_SE_kernel(kernel_x=new_xs, return_values=True)
        
        # calculate rectangular matrix
        new_rows = np.array([np.sum(np.square(k_x - new_x), axis=1) for new_x in new_xs])
        new_rows /= (-2 * self.hyperparam.len * self.hyperparam.len)
        new_rows = np.exp(new_rows)
        new_rows *= self.hyperparam.theta_f * self.hyperparam.theta_f

        self.inv_k_matrix = matrix_block_inversion( Ainv=self.inv_k_matrix, B=new_rows.T, C=new_rows, D=new_k_matrix )

        # compose the new overall matrix
        self.k_matrix = np.vstack(( self.k_matrix, new_rows))
        self.k_matrix = np.hstack(( self.k_matrix, np.vstack((new_rows.T, new_k_matrix))    ))

        # assert np.allclose(self.inv_k_matrix, np.linalg.inv(self.k_matrix)), "Inverse matrix is not correct"

        # update delta
        d = np.array(self.delta)
        d += new_rows.sum(axis=0)
        d_new = np.vstack((new_rows.T, new_k_matrix)).sum(axis=0)
        d = np.append(d, d_new)
        self.delta = deque(d)

        # #update informativity
        # info = np.array(self.informativity)
        # new_xs_info = np.array([self.get_sample_informativity(k_x, x) for x in new_xs])
        # info2 = np.array([self.calculate_array_sample_distances(k_x, x) for x in new_xs]).sum(axis=0)
        # info = np.maximum(info, info2)
        # info = np.append(info, new_xs_info)
        # self.informativity = deque(info)

        # update info mat
        rows = self.k_matrix[-len(new_xs):, :].copy()
        new_info_rows=[]
        for i in range(len(new_xs)):
            rows[i, -len(new_xs)+i] = 0.
            new_info_rows.append(np.argsort(rows[i])[::-1])
        self.info_mat.extend(np.array(new_info_rows))


    def sub_kernel_sample(self, new_x, new_y):
        # new_delta = self.count_delta(new_x) # not using it
        # max_value, max_index = self.get_max(self.delta)
        new_info = self.get_sample_informativity(self.kernel_x, new_x)
        info = np.array([self.info_mat[i][0] for i in range(len(self.info_mat))])
        info_values = self.k_matrix[np.arange(len(info)), info]
        max_value, max_index = np.max(info_values), np.argmax(info_values)
        pass
        if new_info < max_value:
            self.samples_substituted_count += 1
            self.samples_substituted.append(max_index)

            # self.schur_update_SE_kernel(new_x, new_y)
            self.SM_update_SE_kernel(new_x, new_y, max_index)
            self.count = self.count + 1
            if self.count > 0:#int(self.max_k_matrix_size/3):
                self.count = 0
                self.calculate_SE_kernel()
                #TODO: use batch method to avoid recomputing the whole matrix + inverse
                self.inv_k_matrix = np.linalg.inv(self.k_matrix)


    def schur_update_SE_kernel(self, new_x, new_y):
        n = len(self.kernel_x)

        self.kernel_x.append(new_x)
        self.kernel_y.append(new_y)
        self.kernel_x.popleft()
        self.kernel_y.popleft()

        K2 = self.k_matrix[1:n, 1:n]

        K2[:, -1] = np.sum(np.square(self.kernel_x - new_x), axis=1)
        K2[:, -1] /= (-2 * self.hyperparam.len * self.hyperparam.len)
        K2[:, -1] = np.exp(K2[:, -1])
        K2[:, -1] *= self.hyperparam.theta_f * self.hyperparam.theta_f

        K2[n-1, n-1] += self.hyperparam.theta_n * self.hyperparam.theta_n
        K2[n-1, 0:n-1] = (K2[0:n-1, n-1]).T        

        # print('k_matrix', self.k_matrix)
        # print('new k_matrix', K2)
        # print('inv_k_matrix', self.inv_k_matrix)
        e = self.inv_k_matrix[0][0]
        # print('e', e)
        f = self.inv_k_matrix[1:n, 0].reshape((n-1, 1))
        # print('f', f)
        g = K2[n-1, n-1]
        # print('g', g)
        h = K2[0:n-1, n-1].reshape((n-1, 1))
        # print('h', h)
        H = self.inv_k_matrix[1:n, 1:n]
        # print('H', H)
        B = H - (f.dot(f.T)) / e
        # print('B', B)
        s = 1 / (g - (h.T).dot(B.dot(h)))
        # print('s', s)
        haha_11 = B + (B.dot(h)).dot((B.dot(h)).T) * s
        haha_12 = -B.dot(h) * s
        haha_21 = -(B.dot(h)).T * s
        haha_22 = s
        temp_1 = np.hstack((haha_11, haha_12))
        temp_2 = np.hstack((haha_21, haha_22))
        self.inv_k_matrix = np.vstack((temp_1, temp_2))

        # update delta
        self.delta.popleft()
        self.delta.append(0)

        self.delta -= self.k_matrix[0, :-1]
        self.delta += K2[n-1, :n-1]
        self.delta[n-1] += K2[0:n-1, n-1].sum()

        # # update informativity
        # # must be recomputed from scratch -> O(n^2) (it's okay for matrices)
        # k_x = np.array(self.kernel_x)   #already comprehends the new_x
        # info = np.array([self.get_sample_informativity(np.delete(k_x, i), k_x[i]) for i in range(n)])
        # self.informativity = deque(info)

        # update info mat -> upon removal, compute from scratch
        self.info_mat = self.compute_info_mat(self.k_matrix)

        # update k_matrix
        self.k_matrix = K2


    def SM_update_SE_kernel(self, new_x, new_y, index):
        n = len(self.kernel_x)
        self.kernel_x[index] = new_x
        self.kernel_y[index] = new_y
        new_k_matrix = self.k_matrix.copy()

        new_k_matrix[:, index] = np.sum(np.square(self.kernel_x - self.kernel_x[index]), axis=1)
        new_k_matrix[:, index] /= (-2 * self.hyperparam.len * self.hyperparam.len)
        new_k_matrix[:, index] = np.exp(new_k_matrix[:, index])
        new_k_matrix[:, index] *= self.hyperparam.theta_f * self.hyperparam.theta_f

        new_k_matrix[index, index] += self.hyperparam.theta_n * self.hyperparam.theta_n
        new_k_matrix[index, :] = (new_k_matrix[:, index]).T

        r = new_k_matrix[:, index].reshape((n, 1)) - self.k_matrix[:, index].reshape((n, 1))
        A = self.inv_k_matrix - (self.inv_k_matrix.dot(r.dot(self.inv_k_matrix[index, :].reshape((1, n)))))/(1 + r.transpose().dot(self.inv_k_matrix[:, index].reshape((n, 1)))[0, 0])
        self.inv_k_matrix = A - ((A[:, index].reshape((n, 1))).dot(r.transpose().dot(A)))/(1 + (r.transpose().dot(A[:, index].reshape((n, 1))))[0, 0])

        # update delta
        d = np.array(self.delta)
        d -= self.k_matrix[index, :]
        d += new_k_matrix[index, :]
        d[index] = np.sum(new_k_matrix[:, index])
        self.delta = deque(d)

        # # update informativity
        # # must be recomputed from scratch -> O(n^2) (it's okay for matrices)
        # k_x = np.array(self.kernel_x)
        # info = np.array([self.get_sample_informativity(np.delete(k_x, i), k_x[i]) for i in range(n)])
        # self.informativity = deque(info)

        # update info mat -> upon removal, compute from scratch
        self.info_mat = self.compute_info_mat(new_k_matrix)

        self.k_matrix = new_k_matrix


    def remove_kernel_samples(self, amount, infomat=None, kmat=None, delta=None):
        '''
        removes an amount of samples from the kernel,
        so to make the kernel size back to its maximum size when exceeding

        after some tries, the most effective way to do this is to:
        - choose all the samples with max correlation with another sample
        - among those, choose the ones with a highest delta (sum of all correlations)
        - if more than one, remove the highest index (keep oldest, discard newest, to avoid exploring forever)

        for i in I:
            remove:
                - self.kernel_x[i]
                - self.kernel_y[i]
                - self.k_matrix[i, :]
                - self.k_matrix[:, i]
                - self.inv_k_matrix[i, :]   -> i cannot just do this
                - self.inv_k_matrix[:, i]
                - self.delta[i]
        '''
        if amount<=0:
            return
        
        FLAG_work_on_self = False
        if (infomat, kmat, delta) == (None, None, None):
            FLAG_work_on_self = True
            infomat = np.array(self.info_mat, dtype=object)
            kmat = np.array(self.k_matrix)
            delta = np.array(self.delta)
        elif infomat is None or kmat is None or delta is None:
            raise ValueError('infomat, kmat and delta must be all None or all not None')

        kill_list = []

        for i in range(amount):
            info_max_idxs = np.array([inforow[0] for inforow in infomat])        #infomat already sorted
            info_max_vals = kmat[np.arange(len(info_max_idxs)), info_max_idxs]
            
            # filter by max correlation
            max_correlation_idxs = np.argwhere(info_max_vals == np.amax(info_max_vals)).flatten()
            max_correlation_idxs = np.unique((info_max_idxs[max_correlation_idxs], max_correlation_idxs))    # info[argmaxs] are the simmetric indices (not always present in argmaxs since infomat is "diagonal", deque of arrays)

            # filter by worse delta
            max_delta_idxs = np.argwhere(delta[max_correlation_idxs] == np.amax(delta[max_correlation_idxs])).flatten()

            # filter by oldest (highest index)
            kill = np.sort(max_correlation_idxs[max_delta_idxs].flatten())[-1]      # if more than one, remove newest
            kill_list.append(kill)

            # UPDATE infomat, delta, kmat
            #remove `kill` from  infomat , recompute  info  at the beginning of the loop
            infomat[kill] = np.zeros(len(infomat[kill]), dtype=np.int32)
            for i in range(len(infomat)): 
                infomat[i]=np.delete(infomat[i], np.where(np.isin(infomat[i], kill)))
                # infomat[i][infomat[i] > kill]-=1        # adjust indices numbers
            delta -= kmat[kill, :]
            # delta = np.delete(delta, kill)
            # kmat = np.delete(kmat, kill, axis=0)
            # kmat = np.delete(kmat, kill, axis=1)


        #kill stuff
        kill_list = np.sort(kill_list)[::-1]

        kmat = np.delete(kmat, kill_list, axis=0)
        kmat = np.delete(kmat, kill_list, axis=1)
        infomat = np.delete(infomat, kill_list, axis=0)
        delta = np.delete(delta, kill_list)

        if FLAG_work_on_self:
                #TODO: perform matrix_inverse_remove with many indices at once
            self.inv_k_matrix = matrix_inverse_remove_indices(self.inv_k_matrix, kill_list)
            
            for kill in kill_list:
                del self.kernel_x[kill]
                del self.kernel_y[kill]
                for i in range(len(infomat)): 
                    # if i delete idx 1000, idx 1001 will become 1000, etc...
                    infomat[i][infomat[i] > kill]-=1        # adjust indices numbers

            self.samples_substituted_count += len(kill_list)
            self.samples_substituted.append(kill_list)

            # update info mat -> upon removal, compute from scratch
            self.k_matrix = kmat
            # self.info_mat = self.compute_info_mat(kmat)
            # infomat doesn't need to be recomputed
            self.info_mat = deque(infomat)
            self.delta = deque(delta)

            # assert np.allclose( np.abs(np.rint(np.matmul(self.k_matrix, self.inv_k_matrix))), np.eye(len(self.k_matrix)) )

            assert len(self.kernel_x) == len(self.kernel_y) == self.k_matrix.shape[0] == self.k_matrix.shape[1] == self.inv_k_matrix.shape[0] == self.inv_k_matrix.shape[1] == len(self.delta) == len(self.info_mat) == self.max_k_matrix_size,\
                "not all kernel structures have the same size after removal"
        else:
            return np.array(kill_list), infomat, kmat, delta

    def count_delta(self, new_x):
        '''
        this functions provides a metric to choose the informativity of a sample
        but imagine a sample S is the same to an already existing sample,
        that is, though, very informative, hence low correlation with all the other samples

        S would be added for sure, since its delta is very low, but it is not informative at all,
        since it is a duplicate

        i am rewriting to use as a metric the single highest correlation with the other samples
        '''
        n = len(self.kernel_x)

        d = np.sum(np.square(self.kernel_x - new_x), axis=1)
        d /= (-2 * self.hyperparam.len * self.hyperparam.len)
        d = np.exp(d)
        d *= self.hyperparam.theta_f * self.hyperparam.theta_f
        d = np.sum(d)

        return d
    

    def get_sample_informativity(self, kernel_x, x):
        '''
        This functions provides a metric to choose the informativity of a sample:
        returns the highest correlation between the new sample and the already existing samples.

        If x is present in kernel_x, be sure to pass kernel_x removing it first,
        otherwise this function will return 1.0
        '''
        d = np.sum(np.square(kernel_x - x), axis=1)    #still summing over the features dimension
        d /= (-2 * self.hyperparam.len * self.hyperparam.len)
        d = np.exp(d)
        d *= self.hyperparam.theta_f * self.hyperparam.theta_f
        return np.max(d)
    

    def calculate_array_sample_distances(self, kernel_x, new_x):
        '''
        This function takes in input a list of kernel samples,
        calulates the distance with new_x for each sample,
        and returns the np array of the distances.
        '''
        d = np.sum(np.square(kernel_x - new_x), axis=1)
        return d

