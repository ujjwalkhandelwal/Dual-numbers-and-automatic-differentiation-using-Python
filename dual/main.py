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
from func import log_d, exp_d, sin_d, cos_d, sigmoid_d

################################# EXAMPLE-1 ####################################

x = Dual(real=1, dual={'x': 1})
y = Dual(real=2, dual={'y': 1})

f = (x**3) - 2*(x**2)*(y**2) + (y**3)
print(f)
