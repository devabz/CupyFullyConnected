import numpy as np
import cupy as cp


class Kernel:
    def __init__(self, kernel='numpy'):
        self.__kernel_str__ = kernel
        self.kernel = {'numpy': np, 'cupy': cp}.get(kernel, np)

    def set_kernel(self, kernel):
        self.__kernel_str__ = kernel
        self.kernel = {'numpy': np, 'cupy': cp}.get(kernel, np)


kernel = Kernel()
