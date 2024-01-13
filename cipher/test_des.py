import numpy as np
from des import DES as __DES
from speck import Speck

d = __DES(n_rounds=2)
# key = [[6476, 3497189004]]
# pt = np.array([[0x123456ab], [0xcd132536]], dtype=np.uint32)
# print(pt, end="\n\n")
# pt = d.encrypt_one_round(pt, key)
# print(pt, end="\n\n")
# pt = d.encrypt_one_round(pt, key)
# print(pt)
# 
# d = __DES(n_rounds=2)
# key = np.array([[0xAABB0918], [0x2736CCDD]], dtype=np.uint32)
# ks = d.key_schedule(key)
# pt = np.array([[0x123456ab], [0xcd132536]], dtype=np.uint32)
# print(d.encrypt(pt, ks))


d = __DES(n_rounds=16)
key = np.array([[0xAABB0918], [0x2736CCDD]], dtype=np.uint32)
ks = d.key_schedule(key)
pt = np.array([[0x123456ab], [0xcd132536]], dtype=np.uint32)
print(pt)
res = d.encrypt(pt, ks)
print(res)
print(d.decrypt(res, ks))
exit(0)




for round in range(2, 3):
  d = __DES(n_rounds=round)
  key = np.array([[0xAABB0918], [0x2736CCDD]], dtype=np.uint32)
  pt = np.array([[0x123456ab], [0xCD132536]], dtype=np.uint32)
  ks = d.key_schedule(key)
  print(f"{ks=}, {ks.dtype}")
  print("\nENCRYPTION\n\n\n")
  print(d.encrypt(pt, ks))
  exit(0)
  # print(f"{ks=}, {ks.shape=}")
  # print(pt)

  original_pt = pt.copy()

  enc = d.encrypt(pt, ks)
  exit(0)
  assert np.array_equal(d.decrypt(enc, ks), original_pt)

d = __DES(n_rounds=16)
key = np.array([[0xAABB0918], [0x2736CCDD]], dtype=np.uint32)
pt = np.array([[0x123456ab], [0xCD132536]], dtype=np.uint32)
ks = d.key_schedule(key)
# print(f"{ks=}, {ks.shape=}")
# print(pt)

original_pt = pt.copy()

enc = d.encrypt(pt, ks)
print(f"{enc=}")

exit(0)
d = __DES(n_rounds=8)
key = np.array([[0x41414141], [0x42424242]], dtype=np.uint32)
pt = np.array([[0xaaaa], [0xaaaa]], dtype=np.uint32)
ks = d.key_schedule(key)
# print(f"{ks=}, {ks.shape=}")
# print(pt)

original_pt = pt.copy()

enc = d.encrypt(pt, ks)
print(enc)
assert np.array_equal(d.decrypt(enc, ks), original_pt)

# for i in range(16):
#     pt = d.encrypt_one_round(pt, ks[i])
#     # print(hex(pt[0][0]), hex(pt[1][0]), hex(d.get_key(ks[i])))
#     # print(list(map(hex, pt)), hex(d.get_key(ks[i])), end="\n\n")
# # print("\n\n[*] decryption")
# for i in range(15, -1, -1):
#     pt = d.decrypt_one_round(pt, ks[i])
#   # print(hex(pt[0][0]), hex(pt[1][0]), hex(d.get_key(ks[i])))
# 
# print(pt)
# assert np.array_equal(pt, original_pt)
# 
# ks = d.draw_keys(3)
# print(f"{ks=}, {ks.shape=}")
# speck = Speck()
# ks = speck.draw_keys(4)
# print(f"{ks=}, {ks.shape=}")

# 
# pt = d.encrypt_one_round(pt, ks[1])
# print(list(map(hex, pt)), hex(ks[1][0]), end="\n\n")
# pt = d.decrypt_one_round(pt, ks[1])
# print(list(map(hex, pt)), hex(ks[1][0]), end="\n\n")
# 
# pt = d.decrypt_one_round(pt, ks[0])
# print(list(map(hex, pt)), hex(ks[0][0]), end="\n\n")
#     # d.encrypt_one_round(pt, ks[1]))
