from multiprocessing import Process, Lock
from multiprocessing import Queue
from multiprocessing import JoinableQueue

import math, time


def calculate_square(sq_q, sqrt_q):
    while True:
        itm = sq_q.get()
        print(f"Calculating sq of: {itm}")
        square = itm * itm
        sqrt_q.put(square)
        sq_q.task_done()


def calculate_sqroot(sqrt_q, result_q):
    while True:
        itm = sqrt_q.get()  # this blocks the process unless there's a item to consume
        print(f"Calculating sqrt of: {itm}")
        sqrt = math.sqrt(itm)
        result_q.put(sqrt)
        sqrt_q.task_done()


if __name__ == '__main__':

    items = [i for i in range(5, 20)]

    sq_q = JoinableQueue()
    sqrt_q = JoinableQueue()
    result_q = JoinableQueue()

    for i in items:
        sq_q.put(i)

    p_sq = Process(target=calculate_square, args=(sq_q, sqrt_q))
    p_sqrt = Process(target=calculate_sqroot, args=(sqrt_q, result_q))

    p_sq.start()
    p_sqrt.start()

    sq_q.join()
    sqrt_q.join()
    # result_q.join() no need to join this queue

    while not result_q.empty():
        print(result_q.get())