from train_nets import train_distinguisher
from cipher.des import DES
import numpy as np
from os import urandom

des = DES(n_rounds=3)
# characteristic = [0x00808200, 0x60000000]
# characteristic = [0x193ff222, 0x199b1e31]
characteristic = [0x19600000, 0x00000000]
train_distinguisher(
    des, characteristic, lr_high=0.0035, lr_low=0.00022, reg_param=0.000000849, n_neurons=80, n_filters=16
)

