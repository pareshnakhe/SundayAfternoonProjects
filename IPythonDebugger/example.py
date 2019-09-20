from ipd import IPythonDebugger
import os
import matplotlib.pyplot as plt
import numpy as np

ipd = IPythonDebugger()
i_shell = ipd.ipython_debugger()
b = 30


def do():
    a = 42
    print(a)
    i_shell()
    print(a)


do()

