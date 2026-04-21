from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory
import numpy as np


def write_process(shm):
    # 往共享内存写入数据
    arr = np.ndarray((3,), dtype=np.int32, buffer=shm.buf)
    arr[0] = 100
    arr[1] = 200
    arr[2] = 300
    print("子进程写入：", arr)


if __name__ == "__main__":
    # 创建共享内存：大小 = 3个int32 = 12字节
    shm = SharedMemory(create=True, size=12)

    p = Process(target=write_process, args=(shm,))
    p.start()
    p.join()

    # 主进程读取共享内存
    arr = np.ndarray((3,), dtype=np.int32, buffer=shm.buf)
    print("主进程读取：", arr)

    # 释放共享内存
    shm.close()
    shm.unlink()
