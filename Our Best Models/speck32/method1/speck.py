import numpy as np
from os import urandom
import random


def WORD_SIZE():
    return(16)


def ALPHA():
    return(7)


def BETA():
    return(2)


MASK_VAL = 2 ** WORD_SIZE() - 1


def shuffle_together(l):
    state = np.random.get_state()
    for x in l:
        np.random.set_state(state)
        np.random.shuffle(x)


def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))


def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL))


def enc_one_round(p, k):
    c0, c1 = p[0], p[1]
    c0 = ror(c0, ALPHA())
    c0 = (c0 + c1) & MASK_VAL
    c0 = c0 ^ k
    c1 = rol(c1, BETA())
    c1 = c1 ^ c0
    return(c0,c1)


def dec_one_round(c,k):
    c0, c1 = c[0], c[1]
    c1 = c1 ^ c0
    c1 = ror(c1, BETA())
    c0 = c0 ^ k
    c0 = (c0 - c1) & MASK_VAL
    c0 = rol(c0, ALPHA())
    return(c0, c1)


def expand_key(k, t):
    ks = [0 for i in range(t)]
    ks[0] = k[len(k)-1]
    l = list(reversed(k[:len(k)-1]))
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i)
    return(ks)


def encrypt(p, ks,r_start=1):
    x, y = p[0], p[1]
    for k in ks[r_start-1:]:
        x,y = enc_one_round((x,y), k)
    return(x, y)


def decrypt(c, ks):
    x, y = c[0], c[1]
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k)
    return(x,y)


def check_testvector():
    key = (0x1918,0x1110,0x0908,0x0100)
    pt = (0x6574, 0x694c)
    ks = expand_key(key, 22)
    ct = encrypt(pt, ks)
    if (ct == (0xa868, 0x42f2)):
        print("Testvector verified.")
        return(True)
    else:
        print("Testvector not verified.")
        return(False)


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

def make_target_diff_samples(n=10**7, nr=7, diff_type=1, diff=(0x40, 0), return_keys=0,r_start=1,types = "normal",randoml=MASK_VAL,randomr=MASK_VAL):
    p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    if diff_type != 0:
        # print("diff_type",diff_type)
        p1l, p1r = p0l ^ diff[0], p0r ^ diff[1]
    else:
        # print("diff_type",diff_type)
        p1l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        p1r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    ks = expand_key(keys, (r_start-1)+nr)
    c0l, c0r = encrypt((p0l, p0r), ks,r_start)
    c1l, c1r = encrypt((p1l, p1r), ks,r_start)

    X = convert_to_binary([c0l, c0r, c1l, c1r])
    
    # if types.startswith("cpcd"):
    #     X1 = convert_to_binary_new(np.array(np.array((c0l, c0r)) ^ np.array((c1l, c1r))),16,2);
        
    if return_keys == 0:
        return X
    else:
        return X, ks


def make_dataset_with_group_size(diff1,diff2,n, nr, diff=(0x40, 0), group_size=1, r_start=1,types = "normal",randoml=MASK_VAL,randomr=MASK_VAL):
    num = n // 2
    assert num % group_size == 0
    # randoml=MASK_VAL,randomr=MASK_VAL
    if types.startswith("twodiff"):
        X_p = make_target_diff_samples(n=num, nr=nr, diff_type=1, diff=diff1, return_keys=0,r_start=r_start,types=types,randoml=randoml,randomr=randomr)
        X_n = make_target_diff_samples(n=num, nr=nr, diff_type=1, diff=diff2, return_keys=0,r_start=r_start,types=types,randoml=randoml,randomr=randomr)        
    elif types.endswith("_normal"):
        X_p = make_target_diff_samples(n=num, nr=nr, diff_type=2, diff=diff, return_keys=0,r_start=r_start,types=types,randoml=randoml,randomr=randomr)
        X_n = make_target_diff_samples(n=num, nr=nr, diff_type=1, diff=diff, return_keys=0,r_start=r_start,types=types,randoml=randoml,randomr=randomr)
    else:
        X_p = make_target_diff_samples(n=num, nr=nr, diff_type=1, diff=diff, return_keys=0,r_start=r_start,types=types,randoml=randoml,randomr=randomr)
        X_n = make_target_diff_samples(n=num, nr=nr, diff_type=0, return_keys=0,r_start=r_start,types=types,randoml=randoml,randomr=randomr)
    Y_p = [1 for i in range(num // group_size)]
    Y_n = [0 for i in range(num // group_size)]
    X = np.concatenate((X_p, X_n), axis=0).reshape(n // group_size, -1)
    Y = np.concatenate((Y_p, Y_n))
    return X, Y

def make_dataset_with_group_size_no_random(n, nr, diff=(0x80, 0), group_size=2, r_start=1):
    num = n 
    assert num % group_size == 0
    X = make_target_diff_samples(n=num, nr=nr, diff_type=1, diff=diff, return_keys=0,r_start=r_start)
    Y = [1 for i in range(num // group_size)]
    X = np.array(X).reshape(n // group_size, -1)
    Y = np.array(Y)
    return X, Y


def real_differences_data(n, nr, diff, group_size=2, r_start=1):

    # num = n // 2
    num = n 
    assert num % group_size == 0
    X_p = make_target_diff_samples(n=num, nr=nr, diff_type=1, diff=diff, return_keys=0,r_start=r_start)
    X_n = make_target_diff_samples(n=num, nr=nr, diff_type=0, return_keys=0,r_start=r_start)
    # Y_p = [1 for i in range(num // group_size)]
    # Y_n = [0 for i in range(num // group_size)]
    X_p = np.array(X_p).reshape(n // group_size, -1)
    X_n = np.array(X_n).reshape(n // group_size, -1)

    return X_p, X_n
