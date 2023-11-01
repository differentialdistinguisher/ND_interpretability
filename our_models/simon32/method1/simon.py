# SIMON-32 Implementation

import numpy as np
from os import urandom
#import sys

def WORD_SIZE():
    return(16);

MASK_VAL = 2 ** WORD_SIZE() - 1;

def shuffle_together(l):
    state = np.random.get_state();
    for x in l:
        np.random.set_state(state);
        np.random.shuffle(x);

def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)));

def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL));

def enc_one_round(p, k):
    x , y  = p[0] , p[1];
    tmp_1 = rol(x,1) & rol(x,8); 
    tmp_2 = rol(x,2); 
    x = y ^ tmp_1 ^ tmp_2 ^ k;  
    return(x,p[0]);

def dec_one_round(c,k):
    x , y  = c[0] , c[1];
    tmp_1 = rol(y,1) & rol(y,8);
    tmp_2 = rol(y,2);
    y = x ^ tmp_1 ^ tmp_2 ^ k;
    return(c[1],y)

def expand_key(k, t):
    m = 4;
    z = "11111010001001010110000111001101111101000100101011000011100110";
    ks = [0 for i in range(t)];
    ks[0:m] = list(reversed(k[:len(k)]));
    for i in range(m, t):
      tmp = ror(ks[i-1],3);
      if (m==4):
          tmp = tmp ^ ks[i-3];
      tmp_1 = ror(tmp,1);
      tmp = tmp ^ tmp_1;
      tmp_2 = int(z[(i-m)%62]);
      ks[i] = ks[i-m] ^ tmp ^ tmp_2 ^ 0xfffc
    return(ks);

def encrypt(p, ks,r_start=1):
    x, y = p[0], p[1];
    for k in ks[r_start-1:]:
        x,y = enc_one_round((x,y), k);
    # return(np.array((x, y)));
    return (x,y)

def encrypt_phase(p, ks,r_start=5,r_end=8):
    x, y = p[0], p[1]
    for k in ks[r_start-1:r_end]:
        x,y = enc_one_round((x,y), k)
        # print(x,y,k)
    return(x, y)


def decrypt(c, ks):
    x, y = c[0], c[1];
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k);
    return(x,y);

def check_testvector():
  key = (0x1918,0x1110,0x0908,0x0100)
  pt = (0x6565, 0x6877)
  ks = expand_key(key, 32)
  ct = encrypt(pt, ks)
  # print(encrypt(pt, ks))
  #if (ct == (0xa868, 0x42f2)):
  if (ct == (0xc69b, 0xe9bb)):
    print("Testvector verified.")
    return(True);
  else:
    print("Testvector not verified.")
    print(ct)
    return(False);
check_testvector()
#sys.exit()

# convert_to_binary takes as input an array of ciphertext pairs
# where the first row of the array contains the lefthand side of the ciphertexts,
# the second row contains the righthand side of the ciphertexts,
# the third row contains the lefthand side of the second ciphertexts,
# and so on
# it returns an array of bit vectors containing the same data
def convert_to_binary(arr):
    X = np.zeros((4 * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
    for i in range(4 * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return(X)

def convert_to_binary_new(arr,WORD_SIZE=16,NO_OF_WORDS=2):
  X = np.zeros((NO_OF_WORDS * WORD_SIZE,len(arr[0])),dtype=np.uint8);
  for i in range(NO_OF_WORDS * WORD_SIZE):
    index = i // WORD_SIZE;
    offset = WORD_SIZE - (i % WORD_SIZE) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);

def make_target_diff_samples(n=10**7, nr=7, diff_type=1, diff=(0, 0x0020), return_keys=0,r_start=1):
    p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    if diff_type == 1:
        p1l, p1r = p0l ^ diff[0], p0r ^ diff[1]
    else:
        p1l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        p1r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    ks = expand_key(keys, (r_start-1)+nr)
    c0l, c0r = encrypt((p0l, p0r), ks,r_start=r_start)
    c1l, c1r = encrypt((p1l, p1r), ks,r_start=r_start)
    X = convert_to_binary([c0l, c0r, c1l, c1r])
    # X = convert_to_binary_new(np.array(np.array((c0l, c0r)) ^ np.array((c1l, c1r))),16,2);

    if return_keys == 0:
        return X
    else:
        return X, ks


def make_dataset_with_group_size(diff1,diff2,n, nr, diff=(0x0, 0x040), group_size=2,r_start=1,types = "normal"):
    num = n // 2
    assert num % group_size == 0
    # print(diff1,diff2,diff)
    
    if types.startswith("twodiff"):
        # print("twodiff")
        X_p = make_target_diff_samples(n=num, nr=nr, diff_type=1, diff=diff1, return_keys=0,r_start=r_start)
        X_n = make_target_diff_samples(n=num, nr=nr, diff_type=1, diff=diff2,return_keys=0,r_start=r_start)
    else:
        # print(types)
        X_p = make_target_diff_samples(n=num, nr=nr, diff_type=1, diff=diff, return_keys=0,r_start=r_start)
        X_n = make_target_diff_samples(n=num, nr=nr, diff_type=0,return_keys=0,r_start=r_start)

    Y_p = [1 for i in range(num // group_size)]
    Y_n = [0 for i in range(num // group_size)]
    X = np.concatenate((X_p, X_n), axis=0).reshape(n // group_size, -1)
    Y = np.concatenate((Y_p, Y_n))
    return X, Y




