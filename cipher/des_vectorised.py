# original source code https://github.com/0x10001/des
# Modified to make it working with numpy array and this is the vectorized version

import numpy as np

from numba.np.ufunc import parallel
from numba import njit
from cipher.abstract_cipher import AbstractCipher

def print_hex(a:np.ndarray):
    print(list(map(hex, a)))

INITIAL_PERMUTATION = np.array([
    57, 49, 41, 33, 25, 17, 9,  1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7,
    56, 48, 40, 32, 24, 16, 8,  0,
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    ])

INVERSE_PERMUTATION = np.array([
    39, 7,  47, 15, 55, 23, 63, 31,
    38, 6,  46, 14, 54, 22, 62, 30,
    37, 5,  45, 13, 53, 21, 61, 29,
    36, 4,  44, 12, 52, 20, 60, 28,
    35, 3,  43, 11, 51, 19, 59, 27,
    34, 2,  42, 10, 50, 18, 58, 26,
    33, 1,  41, 9,  49, 17, 57, 25,
    32, 0,  40, 8,  48, 16, 56, 24,
])

EXPANSION = np.array([
    31, 0,  1,  2,  3,  4,
    3,  4,  5,  6,  7,  8,
    7,  8,  9,  10, 11, 12,
    11, 12, 13, 14, 15, 16,
    15, 16, 17, 18, 19, 20,
    19, 20, 21, 22, 23, 24,
    23, 24, 25, 26, 27, 28,
    27, 28, 29, 30, 31, 0,
])

PERMUTATION = np.array([
    15, 6,  19, 20, 28, 11, 27, 16,
    0,  14, 22, 25, 4,  17, 30, 9,
    1,  7,  23, 13, 31, 26, 2,  8,
    18, 12, 29, 5,  21, 10, 3,  24,
])

PERMUTED_CHOICE1 = np.array([
    56, 48, 40, 32, 24, 16, 8,
    0,  57, 49, 41, 33, 25, 17,
    9,  1,  58, 50, 42, 34, 26,
    18, 10, 2,  59, 51, 43, 35,
    62, 54, 46, 38, 30, 22, 14,
    6,  61, 53, 45, 37, 29, 21,
    13, 5,  60, 52, 44, 36, 28,
    20, 12, 4,  27, 19, 11, 3,
    ])

PERMUTED_CHOICE2 = np.array([
    13, 16, 10, 23, 0,  4,
    2,  27, 14, 5,  20, 9,
    22, 18, 11, 3,  25, 7,
    15, 6,  26, 19, 12, 1,
    40, 51, 30, 36, 46, 54,
    29, 39, 50, 44, 32, 47,
    43, 48, 38, 55, 33, 52,
    45, 41, 49, 35, 28, 31,
])

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

# Njit doesn't like methods, it prefer functions
def _permute_vectorised(element, bits, mapper):
    ret = 0
    lm = len(mapper)
    for i, v in enumerate(mapper):
        if element & 1 << bits - 1 - v:
            ret |= 1 << lm - 1 - i
    return ret

def _inner_f_vectorised(element):
    ret = 0
    for i, box in enumerate(SUBSTITUTION_BOX):
        i6 = element >> 42 - i * 6 & 0x3f
        ret = ret << 4 | box[i6 & 0x20 | (i6 & 0x01) << 4 | (i6 & 0x1e) >> 1]
    return ret

vp = np.vectorize(njit(_permute_vectorised), excluded=['bits', 'mapper'], otypes=[np.uint64])
ifv = np.vectorize(njit(_inner_f_vectorised), otypes=[np.uint64])

class DESV(AbstractCipher):

    def __init__(self, n_rounds=16, word_size=32, m=1, use_key_schedule=True):
        super(DESV, self).__init__(
            n_rounds, word_size, n_words=2, n_main_key_words=2, n_round_key_words=2, use_key_schedule=use_key_schedule, main_key_word_size=32, round_key_word_size=32
        )
        self.current_round = 0
        self.decryption_round = 0
        self.vp = vp
        self.ifv = ifv

    def rotate_left(self, i28, k):
        return i28 << k & 0x0fffffff | i28 >> 28 - k

    def permute_vector(self, data:np.ndarray, bits, mapper):
        data = np.transpose(data)
        if bits == 32:
            _type = np.uint32
        else:
            _type = np.uint64
        if _type == np.uint64 and data.ndim >= 2:
            data = np.array(np.left_shift(np.uint64(data[:,0]), 32) + data[:,1], dtype=np.uint64)
        mapper = mapper
        ret = self.vp(data.astype(int), bits=bits, mapper=mapper)
        if np.array_equal(mapper, INITIAL_PERMUTATION) or np.array_equal(mapper, INVERSE_PERMUTATION):
            ret = np.array([np.right_shift(ret, 32), ret & 0xffffffff], dtype=np.uint32)
        return ret

    def f_vector(self, block, key):
        key = np.array(np.left_shift(np.uint64(key[:,0]), 32) + key[:,1], dtype=np.uint64)
        # key = np.frombuffer(key.byteswap().tobytes(), dtype=np.uint64).byteswap()
        # key = np.full(block.shape, k, dtype=np.uint64)
        # print(f"BEFORE PERMUTE: {block}")
        # print(f"{block=}, {key=}", end="\n\n")
        block = np.uint64(self.permute_vector(block, 32, EXPANSION))
        # print(f"AFTER PERMUTE: {block}")
        block ^= key
        # ifv = np.vectorize(self._inner_f_vectorised, otypes=[np.uint64])
        ret = self.ifv(block)
        return self.permute_vector(np.uint32(ret), 32, PERMUTATION)

    def encrypt_one_round_vector(self, p:np.ndarray, k:np.ndarray):
        block = p
        block = block[1], block[0] ^ self.f_vector(block[1], k)
        return np.array(block, dtype=np.uint32)

    def key_schedule(self, key):
        ss = len(key[0])
        key = key.transpose()
        key = np.array(np.left_shift(np.uint64(key[:,0]), 32) + key[:,1], dtype=np.uint64)
        # key = np.frombuffer(key.byteswap().tobytes(), dtype=np.uint64).byteswap()
        next_key = self.permute_vector(key, 64, PERMUTED_CHOICE1)
        next_key = next_key >> 28, next_key & 0x0fffffff
        next_key = np.array(next_key, dtype=np.uint32)
        ks = np.zeros((ss, self.n_rounds, 2), dtype=np.uint32)
        for i, bits in enumerate(ROTATES):
            next_key = np.array((self.rotate_left(next_key[0], bits), self.rotate_left(next_key[1], bits)),dtype=np.uint32)
            nkk = next_key.transpose()
            to_use = np.array(np.left_shift(np.uint64(nkk[:, 0]), 28) | nkk[:, 1])
            nk = self.permute_vector(to_use, 56, PERMUTED_CHOICE2)
            res = np.vstack([
                np.right_shift(nk, 32),
                np.bitwise_and(nk, 0xFFFFFFFF)
            ], dtype=np.uint32).T
            # print(f"{res=}, {i=}")
            ks[:,i] = res
            # print(ks)
            if i == self.n_rounds - 1:
                break
        return ks

    def encrypt(self, p, keys):
        block = p
        block = self.permute_vector(block, 64, INITIAL_PERMUTATION)
        for i in range(len(keys[0])):
            block = self.encrypt_one_round_vector(block, keys[:,i])
        block = block[1], block[0]
        block = self.permute_vector(block, 64, INVERSE_PERMUTATION)
        return block

    def decrypt(self, p, keys):
        block = p
        block = self.permute_vector(block, 64, INITIAL_PERMUTATION)
        for i in range(len(keys[0]) - 1, -1, -1):
            block = self.encrypt_one_round_vector(block, keys[:,i])
        block = block[1], block[0]
        block = self.permute_vector(block, 64, INVERSE_PERMUTATION)
        return block


    def encrypt_one_round(self, p, k, rc=None):
        return self.encrypt_one_round_vector(p, k)
    
    def decrypt_one_round(self, c, k, rc=None):
        return self.encrypt_one_round_vector(c, k)

    def calc_back(self, c, p=None, variant=0):
        return c
    
    @staticmethod
    def get_test_vectors():
        # pip install pycryptodome
        # from Crypto.Cipher import DES
        # des = DES.new(bytes.fromhex("aaaaaaaabbbbbbbb"), DES.MODE_ECB)
        # des.encrypt(bytes.fromhex("2222222266666666")).hex()
        # bc229a8e79cd67e6
        des = DESV()
        key = np.array([[0xAABB0918, 0xaaaaaaaa],
                        [0x2736CCDD, 0xbbbbbbbb]], dtype=np.uint32)
        pt = np.array([[0x123456ab, 0x22222222],
                       [0xCD132536, 0x66666666]], dtype=np.uint32)
        ks = des.key_schedule(key)
        ct = np.array([[3233261776, 0xbc229a8e],
                       [0x5f3a829c, 0x79cd67e6]], dtype=np.uint32)
        return [(des, pt, ks, ct)]

if __name__ == "__main__":
    des = DESV()
    a = np.array([[0x123456ab, 0xeeeeeeee, 0x22222222],
                  [0xcd132536, 0x20202020, 0x11111111]], dtype=np.uint32)
    key = np.array([[0xAABB0918], [0x2736CCDD]], dtype=np.uint32)
    k = np.array([
        [[6476, 3497189004], [0x1010, 0x2020]],
        [[0x10101010, 0x2020], [0x203030, 0x404243]],
        [[0x77777777, 0x777], [0x2222, 0x87482]],
    ], dtype=np.uint32)
    print(a, end="\n\n")
    res = des.encrypt_one_round_vector(a, k[:,0])

    print(res, end="\n\n")
    assert np.array_equal(res[:,0], np.array([3440583990, 48382583]))
    res = des.encrypt_one_round_vector(res, k[:,0])
    print(res)
    assert np.array_equal(res[:, 0], np.array([48382583,4052224482]))

    ks = np.array([
        [[6476, 3497189004], [17768, 1478147278]],
        [[0x10101010, 0x2020], [0x203030, 0x404243]],
        [[0x77777777, 0x777], [0x2222, 0x87482]],
    ], dtype=np.uint32)
    des = DESV(n_rounds=2)
    a = np.array([[0x123456ab, 0xeeeeeeee, 0x22222222],
                  [0xcd132536, 0x20202020, 0x11111111]], dtype=np.uint32)
    assert np.array_equal(des.encrypt(a, ks)[:, 0], np.array([ 148440032, 3072977163]))

    print("\n\n\n")
    d = DESV(n_rounds=16)
    key = np.array([[0xAABB0918, 0x30400440, 0x12901292],
                    [0x2736CCDD, 0x102a3011, 0x12abccc3]], dtype=np.uint32)
    ks = d.key_schedule(key)
    ct = des.encrypt(a, ks)
    print(ct)
    pt = des.decrypt(ct, ks)
    print(pt)
    assert np.array_equal(pt, a)

    print("\n\ndifferential test\n")
    d = DESV(n_rounds=1)
    key = np.array([[0xaaaaaa],
                    [0xbbbbbb]], dtype=np.uint32)
    pt0 = np.array([[0xccccccc],
                  [0xddddddd]], dtype=np.uint32)
    characteristic = np.array([[0xeeeeeeee],
                              [0x0]], dtype=np.uint32)
    pt1 = np.bitwise_xor(pt0, characteristic)
    print(f"{pt0.ravel()=}")
    print(f"{pt1.ravel()=}")
    ks = d.key_schedule(key)
    ct0 = d.encrypt_one_round(pt0, ks[0])
    ct1 = d.encrypt_one_round(pt1, ks[0])
    print(f"{ct0.ravel()=}")
    print(f"{ct1.ravel()=}")
    print(f"{characteristic.ravel()=}")
    print(f"{np.bitwise_xor(ct0, ct1).ravel()=}")


