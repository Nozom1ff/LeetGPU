from multiprocessing import Process, Pipe, set_start_method
# 只能在父子进程中


# 子进程函数
def child_func(pipe):
    pipe.send("I' child_func, please report!")
    msg = pipe.recv()
    print("child have received! ", msg)
    pipe.close()


if __name__ == "__main__":
    # 强制指定 spawn
    set_start_method("spawn")

    parent_conn, child_conn = Pipe()
    p = Process(target=child_func, args=(child_conn,))
    p.start()

    # 主进程收消息
    print("主进程收到: ", parent_conn.recv())
    parent_conn.send("我是主进程，已收到！")

    p.join()  # join 主进程停下等待，直到子进程运行完。
