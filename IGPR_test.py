from IGPR import BIGPR
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

def load_csv(file_name):
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        columns = [row for row in reader]

    columns = np.array(columns)
    m_x, n_x = columns.shape
    data_set = np.zeros((m_x, n_x))
    for i in range(m_x):
        for j in range(n_x):
            data_set[i][j] = float(columns[i][j])
    return data_set

training_set = load_csv('training_set.csv')
training_target = load_csv('training_target.csv')
test_set = load_csv('test_set.csv')
test_target = load_csv('test_target.csv')

igpr = BIGPR(training_set[0, :], training_target[0, :])
igpr_batch = BIGPR(training_set[0, :], training_target[0, :])

# ========================
#       TUNE THESE
# ========================
batch_dim = 30
data_len = 1000
data_batches = int( np.ceil(data_len/batch_dim) )
max_kmat_size = 900

igpr.max_k_matrix_size  = max_kmat_size
igpr_batch.max_k_matrix_size = max_kmat_size


# training

print("start batch")
start = time.time()
for i in range(1, data_len, batch_dim):
    e = i+batch_dim if i+batch_dim < data_len else data_len
    igpr_batch.learn_batch(training_set[i:e, :], training_target[i:e, :])
end = time.time()
print("Done! time: ", end-start)

print("start mono")
start = time.time()
for i in range(1, data_len):
    igpr.learn(training_set[i, :], training_target[i, :])
end = time.time()
print("Done! time: ", end-start)


# assert np.allclose(igpr_batch.k_matrix, igpr.k_matrix), "k_matrix not equal"
# assert np.allclose(igpr_batch.kernel_x, igpr.kernel_x), "kernle_x not equal"
# assert np.allclose(igpr_batch.kernel_y, igpr.kernel_y), "kernle_y not equal"
# print("igpr and igpr_batch are equal")


# testing

pred = igpr.predict(training_set[0, :])
pred_batch = igpr_batch.predict(training_set[0, :])
for i in range(1, data_len):
    pred = np.vstack((pred, igpr.predict(training_set[i, :])))
    pred_batch = np.vstack((pred_batch, igpr_batch.predict(training_set[i, :])))

# assert np.allclose(pred, pred_batch), "prediction not equal"

# losses

loss = np.mean(np.square(pred - training_target[0:data_len, :]))
loss_batch = np.mean(np.square(pred_batch - training_target[0:data_len, :]))
print("loss: ", loss)
print("loss_batch: ", loss_batch)


# plotting

fig = plt.figure(figsize=(10, 6))
gs = GridSpec(3, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[0, 1])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[2, 1])
ax7 = fig.add_subplot(gs[0, 2])
ax8 = fig.add_subplot(gs[1, 2])
ax9 = fig.add_subplot(gs[2, 2])

ax1.set_title('True')
ax4.set_title('IGPR')
ax7.set_title('BIGPR')

ax1.plot(training_target[0:data_len, 0], label='target', color='red')
ax2.plot(training_target[0:data_len, 1], color='red')
ax3.plot(training_target[0:data_len, 2], color='red')
ax4.plot(pred[0:data_len, 0], label='pred', color='green')
ax5.plot(pred[0:data_len, 1], color='green')
ax6.plot(pred[0:data_len, 2], color='green')
ax7.plot(pred_batch[0:data_len, 0], label='pred_batch')
ax8.plot(pred_batch[0:data_len, 1])
ax9.plot(pred_batch[0:data_len, 2])


plt.show()


fig = plt.figure(figsize=(10, 6))
gs = GridSpec(3, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[0, 1])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[2, 1])
ax7 = fig.add_subplot(gs[0, 2])
ax8 = fig.add_subplot(gs[1, 2])
ax9 = fig.add_subplot(gs[2, 2])

ax1.set_title('True')
ax4.set_title('IGPR')
ax7.set_title('BIGPR')

ax1.plot(training_target[0:data_len, 3], label='target', color='red')
ax2.plot(training_target[0:data_len, 4], color='red')
ax3.plot(training_target[0:data_len, 5], color='red')
ax4.plot(pred[0:data_len, 3], label='pred', color='green')
ax5.plot(pred[0:data_len, 4], color='green')
ax6.plot(pred[0:data_len, 5], color='green')
ax7.plot(pred_batch[0:data_len, 3], label='pred_batch')
ax8.plot(pred_batch[0:data_len, 4])
ax9.plot(pred_batch[0:data_len, 5])

plt.show()


fig = plt.figure(figsize=(20, 6))
gs = GridSpec(3, 1, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])

ax1.set_title('y=3')
ax2.set_title('y=4')
ax3.set_title('y=5')

ax1.plot(training_target[0:data_len, 3], color='red', label='target')
ax1.plot(pred[0:data_len, 3], label='pred', color='green', linestyle='--')
ax1.plot(pred_batch[0:data_len, 3], label='pred_batch', linestyle='dashdot')
ax2.plot(training_target[0:data_len, 4], color='red')
ax2.plot(pred[0:data_len, 4], color='green')
ax2.plot(pred_batch[0:data_len, 4])
ax3.plot(training_target[0:data_len, 5], color='red')
ax3.plot(pred[0:data_len, 5], color='green')
ax3.plot(pred_batch[0:data_len, 5])

legend = ax1.legend(loc='upper right', shadow=True, fontsize='x-large')
# legend = ax2.legend(loc='upper right', shadow=True, fontsize='x-large')
# legend = ax3.legend(loc='upper right', shadow=True, fontsize='x-large')

plt.show()


