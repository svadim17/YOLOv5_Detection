from multiprocessing import Process
import time


def worker():
    while True:
        print("Inside the worker")
        time.sleep(10)


def create_proc():
    a = Process()