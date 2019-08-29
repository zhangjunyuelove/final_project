# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:48:48 2019

@author: Administrator
"""

import numpy as np
import os

def setup():
    np.set_printoptions(threshold=np.inf)
    np.random.seed(7)
    if not os.path.exists(".\weights"):
        os.makedirs(".\weights")