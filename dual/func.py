################################################################################
#                                                                              #
#	UJJWAL KHANDELWAL                                                      #    
#	DUAL NUMBERS AND AUTOMATIC DIFFERENTIATION                             #
#	PYTHON 3.7.10                                                          #
#                                                                              #
################################################################################

#######################   IMPORT DEPENDENCIES   ################################

import numpy as np
from ad_dual import Dual

################################################################################

def log_d(dual_number):
    dual = {}
    a = dual_number.real
    sa = np.log(a)
    for key in dual_number.dual:
        dual[key] = dual_number.dual[key]/a
    return Dual(sa, dual)

def exp_d(dual_number):
    dual = {}
    a = dual_number.real
    sa = np.exp(a)
    for key in dual_number.dual:
        dual[key] = dual_number.dual[key]*sa
    return Dual(sa, dual)

def sin_d(dual_number):
    dual = {}
    a = dual_number.real
    sa = np.sin(a)
    for key in dual_number.dual:
        dual[key] = dual_number.dual[key]*np.cos(a)
    return Dual(sa, dual)

def cos_d(dual_number):
    dual = {}
    a = dual_number.real
    sa = np.cos(a)
    for key in dual_number.dual:
        dual[key] = -np.sin(a)*dual_number.dual[key]
    return Dual(sa, dual)
    
def sigmoid_d(dual_number):
    dual = {}
    a = dual_number.real
    sa = 1 / (1 + np.exp(-a))
    for key in dual_number.dual:
        dual[key] = dual_number.dual[key]*sa*(1-sa)
    return Dual(sa, dual)
