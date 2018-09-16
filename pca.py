# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 16:23:14 2018

@author: Shanthi
"""

from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd

# You can add the parameter data_home to wherever to where you want to download your data
mnist = fetch_mldata('MNIST original')

mnist

# These are the images# These 
mnist.data.shape

# These are the labels
mnist.target.shape


# test_size: what proportion of original data is used for test set# test_ 
train_img, test_img, train_lbl, test_lbl = train_test_split(
    mnist.data, mnist.target, test_size=1/7.0, random_state=0)
