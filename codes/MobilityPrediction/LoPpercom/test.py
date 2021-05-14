from multiprocessing import Pool
import time


def func(i):
    time.sleep(0.5)
    arr = []
    arr.append(i)
    return i*i, i, arr


if __name__ == '__main__':
    p = Pool(5)
    ret = p.map(func, range(10))
    print(ret)
