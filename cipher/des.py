# original source code https://github.com/0x10001/des
# modified to accept numpy array in input
# the code is a mess
import copy
import struct
import numpy as np

from cipher.abstract_cipher import AbstractCipher

INITIAL_PERMUTATION = (
    57, 49, 41, 33, 25, 17, 9,  1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7,
    56, 48, 40, 32, 24, 16, 8,  0,
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
)

INVERSE_PERMUTATION = (
    39, 7,  47, 15, 55, 23, 63, 31,
    38, 6,  46, 14, 54, 22, 62, 30,
    37, 5,  45, 13, 53, 21, 61, 29,
    36, 4,  44, 12, 52, 20, 60, 28,
    35, 3,  43, 11, 51, 19, 59, 27,
    34, 2,  42, 10, 50, 18, 58, 26,
    33, 1,  41, 9,  49, 17, 57, 25,
    32, 0,  40, 8,  48, 16, 56, 24,
)

EXPANSION = (
    31, 0,  1,  2,  3,  4,
    3,  4,  5,  6,  7,  8,
    7,  8,  9,  10, 11, 12,
    11, 12, 13, 14, 15, 16,
    15, 16, 17, 18, 19, 20,
    19, 20, 21, 22, 23, 24,
    23, 24, 25, 26, 27, 28,
    27, 28, 29, 30, 31, 0,
)

PERMUTATION = (
    15, 6,  19, 20, 28, 11, 27, 16,
    0,  14, 22, 25, 4,  17, 30, 9,
    1,  7,  23, 13, 31, 26, 2,  8,
    18, 12, 29, 5,  21, 10, 3,  24,
)

PERMUTED_CHOICE1 = (
    56, 48, 40, 32, 24, 16, 8,
    0,  57, 49, 41, 33, 25, 17,
    9,  1,  58, 50, 42, 34, 26,
    18, 10, 2,  59, 51, 43, 35,
    62, 54, 46, 38, 30, 22, 14,
    6,  61, 53, 45, 37, 29, 21,
    13, 5,  60, 52, 44, 36, 28,
    20, 12, 4,  27, 19, 11, 3,
)

PERMUTED_CHOICE2 = (
    13, 16, 10, 23, 0,  4,
    2,  27, 14, 5,  20, 9,
    22, 18, 11, 3,  25, 7,
    15, 6,  26, 19, 12, 1,
    40, 51, 30, 36, 46, 54,
    29, 39, 50, 44, 32, 47,
    43, 48, 38, 55, 33, 52,
    45, 41, 49, 35, 28, 31,
)

SUBSTITUTION_BOX = (
    (
        14, 4,  13, 1,  2,  15, 11, 8,  3,  10, 6,  12, 5,  9,  0,  7,
        0,  15, 7,  4,  14, 2,  13, 1,  10, 6,  12, 11, 9,  5,  3,  8,
        4,  1,  14, 8,  13, 6,  2,  11, 15, 12, 9,  7,  3,  10, 5,  0,
        15, 12, 8,  2,  4,  9,  1,  7,  5,  11, 3,  14, 10, 0,  6,  13,
    ),
    (
        15, 1,  8,  14, 6,  11, 3,  4,  9,  7,  2,  13, 12, 0,  5,  10,
        3,  13, 4,  7,  15, 2,  8,  14, 12, 0,  1,  10, 6,  9,  11, 5,
        0,  14, 7,  11, 10, 4,  13, 1,  5,  8,  12, 6,  9,  3,  2,  15,
        13, 8,  10, 1,  3,  15, 4,  2,  11, 6,  7,  12, 0,  5,  14, 9,
    ),
    (
        10, 0,  9,  14, 6,  3,  15, 5,  1,  13, 12, 7,  11, 4,  2,  8,
        13, 7,  0,  9,  3,  4,  6,  10, 2,  8,  5,  14, 12, 11, 15, 1,
        13, 6,  4,  9,  8,  15, 3,  0,  11, 1,  2,  12, 5,  10, 14, 7,
        1,  10, 13, 0,  6,  9,  8,  7,  4,  15, 14, 3,  11, 5,  2,  12,
    ),
    (
        7,  13, 14, 3,  0,  6,  9,  10, 1,  2,  8,  5,  11, 12, 4,  15,
        13, 8,  11, 5,  6,  15, 0,  3,  4,  7,  2,  12, 1,  10, 14, 9,
        10, 6,  9,  0,  12, 11, 7,  13, 15, 1,  3,  14, 5,  2,  8,  4,
        3,  15, 0,  6,  10, 1,  13, 8,  9,  4,  5,  11, 12, 7,  2,  14,
    ),
    (
        2,  12, 4,  1,  7,  10, 11, 6,  8,  5,  3,  15, 13, 0,  14, 9,
        14, 11, 2,  12, 4,  7,  13, 1,  5,  0,  15, 10, 3,  9,  8,  6,
        4,  2,  1,  11, 10, 13, 7,  8,  15, 9,  12, 5,  6,  3,  0,  14,
        11, 8,  12, 7,  1,  14, 2,  13, 6,  15, 0,  9,  10, 4,  5,  3,
    ),
    (
        12, 1,  10, 15, 9,  2,  6,  8,  0,  13, 3,  4,  14, 7,  5,  11,
        10, 15, 4,  2,  7,  12, 9,  5,  6,  1,  13, 14, 0,  11, 3,  8,
        9,  14, 15, 5,  2,  8,  12, 3,  7,  0,  4,  10, 1,  13, 11, 6,
        4,  3,  2,  12, 9,  5,  15, 10, 11, 14, 1,  7,  6,  0,  8,  13,
    ),
    (
        4,  11,  2, 14, 15, 0,  8,  13, 3,  12, 9,  7,  5,  10, 6,  1,
        13, 0,  11, 7,  4,  9,  1,  10, 14, 3,  5,  12, 2,  15, 8,  6,
        1,  4,  11, 13, 12, 3,  7,  14, 10, 15, 6,  8,  0,  5,  9,  2,
        6,  11, 13, 8,  1,  4,  10, 7,  9,  5,  0,  15, 14, 2,  3,  12,
    ),
    (
        13, 2,  8,  4,  6,  15, 11, 1,  10, 9,  3,  14, 5,  0,  12, 7,
        1,  15, 13, 8,  10, 3,  7,  4,  12, 5,  6,  11, 0,  14, 9,  2,
        7,  11, 4,  1,  9,  12, 14, 2,  0,  6,  10, 13, 15, 3,  5,  8,
        2,  1,  14, 7,  4,  10, 8,  13, 15, 12, 9,  0,  3,  5,  6,  11,
    ),
)

ROTATES = (
    1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1,
)

class DES(AbstractCipher):

    def __init__(self, n_rounds=16, word_size=32, m=1, use_key_schedule=True):
        super(DES, self).__init__(
            n_rounds, word_size, n_words=2, n_main_key_words=2, n_round_key_words=2, use_key_schedule=use_key_schedule, main_key_word_size=32, round_key_word_size=32
        )
        self.current_round = 0
        self.decryption_round = 0

    def rotate_left(self, i28, k):
        return i28 << k & 0x0fffffff | i28 >> 28 - k

    def permute(self, data, bits, mapper):
        ret = 0
        for i, v in enumerate(mapper):
            if data & 1 << bits - 1 - v:
                ret |= 1 << len(mapper) - 1 - i
        return ret

    def f(self, block, key):
        block = copy.deepcopy(block)
        # print(f"BEFORE PERMUTE: {block}")
        block = self.permute(block, 32, EXPANSION)
        # print(f"AFTER PERMUTE: {block}")
        block ^= key
        ret = 0
        for i, box in enumerate(SUBSTITUTION_BOX):
            i6 = block >> 42 - i * 6 & 0x3f
            ret = ret << 4 | box[i6 & 0x20 | (i6 & 0x01) << 4 | (i6 & 0x1e) >> 1]
        res = self.permute(ret, 32, PERMUTATION)
        return res


    def key_schedule(self, key):
        master_key = key
        # print(f"{master_key=}, {len(master_key[0])}")
        ll = []
        key = np.array([[master_key[0][0]], [master_key[1][0]]]).byteswap().tobytes()
        key = struct.unpack(">Q", key)[0]
        # key, = struct.unpack(">Q", key)
        next_key = self.permute(key, 64, PERMUTED_CHOICE1)
        next_key = next_key >> 28, next_key & 0x0fffffff
        ks = [(0, 0) for i in range(self.n_rounds)]
        for i, bits in enumerate(ROTATES):
            next_key = self.rotate_left(next_key[0], bits), self.rotate_left(next_key[1], bits)
            to_use = next_key[0] << 28 | next_key[1]
            x = self.permute(next_key[0] << 28 | next_key[1], 56, PERMUTED_CHOICE2)
            ks[i] = [x >> 32, x & 0xffffffff]
            if i == self.n_rounds - 1:
                break
        # return np.array(ks)[:, np.newaxis]
        # kk = ks[:]
        ll.extend([ks])
        aa = np.array(ll)
        # print(aa[:, np.newaxis])
        return aa[0][:, np.newaxis]
    # return np.array(ll).reshape(self.n_rounds, 1, len(master_key[0]))



    def get_key(self, k):
        return (k[0][0] << 32) + k[0][1]

    def encrypt(self, p, keys):
        block = p
        block = block.byteswap().tobytes()
        block = struct.unpack(">Q", block)[0]
        block = self.permute(block, 64, INITIAL_PERMUTATION)
        block = block >> 32, block & 0xffffffff
        block = np.array([[block[0]], [block[1]]], dtype=np.uint32)
        for i in range(len(keys)):
            block = self.encrypt_one_round(block, keys[i])
        block = block.byteswap().tobytes()
        block = struct.unpack(">Q", block)[0]
        block = block >> 32, block & 0xffffffff
        block = block[1] << 32 | block[0]
        block = self.permute(block, 64, INVERSE_PERMUTATION)
        return np.array([[block >> 32], [block & 0xffffffff]], dtype=np.uint32)

    def encrypt_one_round(self, p, k, rc=None):
        block = p
        k = self.get_key(k)
        block = block.byteswap().tobytes()
        block = struct.unpack(">Q", block)[0]
        block = block >> 32, block & 0xffffffff

        block = block[1], block[0] ^ self.f(block[1], k)
        return np.array([[block[0]], [block[1]]], dtype=np.uint32)
    
    def decrypt(self, c, keys):
        block = c
        block = block.byteswap().tobytes()
        block = struct.unpack(">Q", block)[0]
        block = self.permute(block, 64, INITIAL_PERMUTATION)
        block = block >> 32, block & 0xffffffff
        block = np.array([[block[0]], [block[1]]], dtype=np.uint32)
        for i in range(len(keys) - 1, -1, -1):
            block = self.encrypt_one_round(block, keys[i])
        block = block.byteswap().tobytes()
        block = struct.unpack(">Q", block)[0]
        block = block >> 32, block & 0xffffffff
        block = self.permute(block[1] << 32 | block[0], 64, INVERSE_PERMUTATION)
        return np.array([[block >> 32], [block & 0xffffffff]], dtype=np.uint32)

    def decrypt_one_round(self, c, k, rc=None):
        block = c
        k = self.get_key(k)
        block = block[1][0], block[0][0]

        block = block[1], block[0] ^ self.f(block[1], k)
        ret = np.array([[block[1]], [block[0]]], dtype=np.uint32)
        return ret
    
    def calc_back(self, c, p=None, variant=0):
        return c
    
    @staticmethod
    def get_test_vectors():
        des = DES()
        key = np.array([[0xAABB0918], [0x2736CCDD]], dtype=np.uint32)
        pt = np.array([[0x123456ab], [0xCD132536]], dtype=np.uint32)
        ks = des.key_schedule(key)
        ct = np.array([[3233261776], [0x5f3a829c]], dtype=np.uint32)
        return [(des, pt, ks, ct)]

