from train_nets import train_distinguisher
from cipher.des_vectorised import DESV
import numpy as np
from os import urandom

des = DESV(n_rounds=3)
# characteristic = [0x00808200, 0x60000000]
# characteristic = [0x193ff222, 0x199b1e31]
characteristic = [0x19600000, 0x00000000]
# characteristic = [0x405c0000, 0x04000000]
# characteristic = [0x49101902, 0x33111122]
train_distinguisher(
    des, characteristic, n_train_samples=4000000, n_val_samples=100000, lr_high=0.0035, lr_low=0.00022, reg_param=0.000000849, n_neurons=80, n_filters=16
)
