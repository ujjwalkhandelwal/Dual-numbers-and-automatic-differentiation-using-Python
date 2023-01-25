################################################################################
#                                                                              #
#	UJJWAL KHANDELWAL                                                            #    
#	DUAL NUMBERS AND AUTOMATIC DIFFERENTIATION                                   #
#	PYTHON 3.7.10                                                                #
#                                                                              #
################################################################################

#######################   IMPORT DEPENDENCIES   ################################

import numpy as np

###########################  DUAL CLASS  ###################################

class Dual:
    
    def __init__(self, real, dual):
        '''
        real: real number
        dual: dict (key=name_index and value=value)
        '''
        self.real = real
        self.dual = dual
        
    def __add__(self, argument):
        if isinstance(argument, Dual):
            real = self.real + argument.real
            dual = {}
            for key in self.dual:
                dual[key] = self.dual[key]
            for key in argument.dual:
                if key in dual:
                    dual[key] += argument.dual[key]
                else:
                    dual[key] = argument.dual[key]    
            return Dual(real, dual)
        else:
            return Dual(self.real + argument, self.dual)
        
    __radd__ = __add__
    
    def __sub__(self, argument):
        if isinstance(argument, Dual):
            real = self.real - argument.real
            dual = {}
            for key in self.dual:
                dual[key] = self.dual[key]
            for key in argument.dual:
                if key in dual:
                    dual[key] -= argument.dual[key]
                else:
                    dual[key] = -argument.dual[key]    
            return Dual(real, dual)
        else:
            return Dual(self.real - argument, self.dual)
        
    def __rsub__(self, argument):
            return -self + argument
    
    def __mul__(self, argument):
        if isinstance(argument, Dual):
            real = self.real * argument.real
            dual = {}
            for key in self.dual:
                dual[key] = self.dual[key] * argument.real
            for key in argument.dual:
                if key in dual:
                    dual[key] += argument.dual[key] * self.real
                else:
                    dual[key] = argument.dual[key] * self.real
            return Dual(real, dual)
        else:
            dual = {}
            for key in self.dual:
                dual[key] = self.dual[key] * argument
            return Dual(self.real * argument, dual)
        
    __rmul__ = __mul__
  
    def __truediv__(self,argument):
        if isinstance(argument, Dual):
            x = argument.real
            new_arg = self.div_neg(argument)
            num = Dual(self.real, self.dual)
            num_modified = num*new_arg
            dual = {}
            for key in num_modified.dual:
                dual[key] = num_modified.dual[key] / (x*x)
            return Dual(num_modified.real / (x*x), dual)
        else:
            dual = {}
            for key in self.dual:
                dual[key] = self.dual[key] / argument
            return Dual(self.real / argument, dual)

    def __rtruediv__(self,argument):
        x = self.real
        den = Dual(self.real, self.dual)
        new_arg = self.div_neg(den)
        num_modified = argument*new_arg
        dual = {}
        for key in num_modified.dual:
            dual[key] = num_modified.dual[key] / (x*x)
        return Dual(num_modified.real / (x*x), dual)
        
    def __pow__(self, power):
        a = self.real
        dual = {}
        for key in self.dual:
            dual[key] = power*self.dual[key]*(a**(power-1))
        return Dual(a**power,dual)
    
    def __neg__(self):
        dual = {}
        for key in self.dual:
            dual[key] = self.dual[key]*(-1)
        return Dual(-self.real,dual)

    def div_neg(self, argument):
        dual = {}
        for key in argument.dual:
            dual[key] = argument.dual[key]*(-1)
        return Dual(argument.real,dual)
    
    def __str__(self):
        s = 'f = ' + str(round(self.real,6)) + '\n'
        for key in self.dual:
            s += 'f' + key + ' = ' + str(round(self.dual[key],6)) + '\n'
        return s
