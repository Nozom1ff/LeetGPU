from multiprocessing import Process, Queue
import time


def producer(queue):
    for i in range(3):
        msg = f"消息 {i}"
        queue.put(msg)
        print(f"生产：{i}")
        time.sleep(0.5)


def consumer(queue):
    while True:
        msg = queue.get()
        if msg == "END":
            break
        print(f"消费：{msg}")


if __name__ == "__main__":
    q = Queue()

    p1 = Process(target=producer, args=(q,))
    p2 = Process(target=consumer, args=(q,))

    p1.start()
    p2.start()
    p1.join()  # 等待生产者完成
    q.put("END")
    p2.join()
