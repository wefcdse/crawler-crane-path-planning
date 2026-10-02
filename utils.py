import time


class Timer:
    def __init__(self, name=None, auto_print=False):
        self.name = name
        self.sum_time = 0
        self.auto_print = auto_print

    def start(self):
        self.__enter__()

    def stop(self):
        self.__exit__()

    def __enter__(self):
        if self.name is None:
            return
        self.tstart = time.time()

    def __exit__(self, *_):  # noqa
        if self.name is None:
            return
        self.sum_time += time.time() - self.tstart
        if self.auto_print:
            self.print()

    def print(self):
        if self.name:
            print("[%s]" % self.name, end=" ")
        print("%s" % (self.sum_time))

    def value(self):
        return self.sum_time
