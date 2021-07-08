# Dual numbers and automatic differentiation using Python
[![GitHub license](https://img.shields.io/github/license/ujjwalkhandelwal/Dual-numbers-and-automatic-differentiation-using-Python?style=flat-square)](https://github.com/ujjwalkhandelwal/Dual-numbers-and-automatic-differentiation-using-Python/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/ujjwalkhandelwal/Dual-numbers-and-automatic-differentiation-using-Python?style=flat-square
)](https://github.com/ujjwalkhandelwal/Dual-numbers-and-automatic-differentiation-using-Python/issues)

Implemented the forward mode of automatic differentiation with the help of [dual numbers](https://en.wikipedia.org/wiki/Dual_number). We first implemented a class **Dual** with the constructor **__init__**, the functions **__add__**, **__radd__**, **__sub__**, **__rsub__**, **__mul__**, **__rmul__**, **__truediv__**, **__rtruediv__**, **__neg__** and **__pow__**. As the names suggest, those functions and properties implement basic arithmetic operations for Dual numbers:

__init__ : constructor that initialises an object of class **Dual**. Each object represents a dual number **a+εb** with real component **a** (*self.real*) and dual component **b** (*self.dual*).

__add__ : adds an argument _argument_ to the dual number, i.e. **a + εb + argument**. 

__radd__ : adds the dual number to the argument _argument_, i.e. **argument + a + εb**.

__sub__ : subtracts an argument _argument_ from the dual number. 

__rsub__ : subtracts the dual number from the argument _argument_.

__mul__ : multiplies the dual number with the argument _argument_.

__rmul__ : multiplies an argument _argument_ with the dual number. 

__truediv__ : divides the dual number by an argument _argument_.

__rtruediv__ : divides the argument _argument_ by the dual number.

__neg__ : returns the negative of the dual number **a + εb**, i.e. **-a - εb**.

__pow__ : takes the _power_-th power of the dual number. i.e. **(a + εb)<sup>power<sup>** 

Next, we implemented the following functions that are acting on dual numbers of the form **a+εb**:

__log_d__ :  log(a+εb) 

__exp_d__ :  exp(a+εb) 

__sin_d__ :  sin(a+εb) 

__cos_d__ :  cos(a+εb) 

__sigmoid_d__ :  1/1+exp(−(a+εb))

## Dependencies
    
  - Numpy (pip install numpy)

## Utilities
Once the installation is finished (downloading or cloning the files), go to the `dual` folder and follow the below simple guidelines to execute **Dual** class effectively (either write the code in command line or in a python editor with the name say `main.py`) OR you can also follow the jupyter notebook with the name `dual.ipynb`.  
```py
>>> import numpy as np
>>> from ad_dual import Dual
```

Next, import the functions (not necessarily all the functions but the one you need) using:
```py
>>> from func import log_d, exp_d, sin_d, cos_d, sigmoid_d
```

### Example-1

![eg1](https://latex.codecogs.com/gif.latex?f%28x%2Cy%2Cz%29%20%3D%20x%5E3%20-%202x%5E2y%5E2%20&plus;%20y%5E3%20%5C%5C%20%5C%5C%20f_x%20%3D%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20%5C%3Ax%7D%20%3D%203x%5E2%20-%204xy%5E2%20%5C%5C%20%5C%5C%20%5C%5C%20f_y%20%3D%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20%5C%3Ay%7D%20%3D%203y%5E2%20-%204x%5E2y)

At `x=1` and `y=2`,

f = 1, f<sub>x</sub> = -13, f<sub>y</sub> = 4

```py
x = Dual(real=1, dual={'x': 1})
y = Dual(real=2, dual={'y': 1})

f = (x**3) - 2*(x**2)*(y**2) + (y**3)
print(f)
```

You will see the following output:

```py
f = 1
fx = -13
fy = 4
```
**NOTE:** The key, value pair in the dictionary indicates the symbol with which you want to represent the variable and the value of the dual number respectively. Like **y = Dual(real=2, dual={'y': 7})** represents **y = 2 + 7ε**. In case you want to calculate the partial derivatives of `f`, keep the value of the dict as 1 (**y = Dual(real=2, dual={'y': 1})**
    
### Example-2


![eg2](https://latex.codecogs.com/gif.latex?f%28x%2Cy%2Cz%29%20%3D%20%5Cfrac%7B81x%7D%7Bx&plus;y%5E2%7D%20%5C%5C%20%5C%5C%20%5C%5C%20f_x%20%3D%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20%5C%3Ax%7D%20%3D%20%5Cfrac%7B81y%5E2%7D%7B%5Cleft%28x&plus;y%5E2%5Cright%29%5E2%7D%20%5C%5C%20%5C%5C%20%5C%5C%20f_y%20%3D%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20%5C%3Ay%7D%20%3D%20-%5Cfrac%7B162xy%7D%7B%5Cleft%28x&plus;y%5E2%5Cright%29%5E2%7D)

At `x=2` and `y=4`,

f = 9, f<sub>x</sub> = 4, f<sub>y</sub> = -4

```py
x = Dual(real=2, dual={'x': 1})
y = Dual(real=4, dual={'y': 1})
f = 81*x / (x+(y**2))
print(f)
```

You will see the following output:

```py
f = 9.0
fx = 4.0
fy = -4.0
```

### Example-3

![eg2](https://latex.codecogs.com/gif.latex?f%28x%2Cy%2Cz%29%20%3D%20%5Cfrac%7B36xz%7D%7Bx&plus;z%5E2&plus;y%5E2%7D%20%5C%5C%20%5C%5C%20%5C%5C%20f_x%20%3D%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20%5C%3Ax%7D%20%3D%20%5Cfrac%7B36z%5Cleft%28z%5E2&plus;y%5E2%5Cright%29%7D%7B%5Cleft%28x&plus;z%5E2&plus;y%5E2%5Cright%29%5E2%7D%20%5C%5C%20%5C%5C%20%5C%5C%20f_y%20%3D%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20%5C%3Ay%7D%20%3D%20-%5Cfrac%7B72xzy%7D%7B%5Cleft%28x&plus;z%5E2&plus;y%5E2%5Cright%29%5E2%7D%20%5C%5C%20%5C%5C%20%5C%5C%20f_z%20%3D%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20%5C%3Az%7D%20%3D%20%5Cfrac%7B36x%5Cleft%28-z%5E2&plus;x&plus;y%5E2%5Cright%29%7D%7B%5Cleft%28x&plus;z%5E2&plus;y%5E2%5Cright%29%5E2%7D)

At `x=1`, `y=2` and `z=1`,

f = 6, f<sub>x</sub> = 5, f<sub>y</sub> = -4, f<sub>z</sub> = 4

```py
x = Dual(1, {'x': 1})
y = Dual(2, {'y': 1})
z = Dual(1, {'z': 1})

f = 36*x*z / (x+(z**2)+(y**2))
print(f)
```

You will see the following output:

```py
f = 6.0
fx = 5.0
fz = 4.0
fy = -4.0
```

### Example-4

![eg2](https://latex.codecogs.com/gif.latex?f%28x%2Cy%2Cz%29%20%3D%20%5Cfrac%7B%5Csin%28x%29%7D%7B%5Ccos%28y%29&plus;x%5E2%7D%20%5C%5C%20%5C%5C%20%5C%5C%20f_x%20%3D%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20%5C%3Ax%7D%20%3D%20%5Cfrac%7B%5Ccos%20%5Cleft%28x%5Cright%29%5Cleft%28%5Ccos%20%5Cleft%28y%5Cright%29&plus;x%5E2%5Cright%29-2x%5Csin%20%5Cleft%28x%5Cright%29%7D%7B%5Cleft%28%5Ccos%20%5Cleft%28y%5Cright%29&plus;x%5E2%5Cright%29%5E2%7D%20%5C%5C%20%5C%5C%20%5C%5C%20f_y%20%3D%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20%5C%3Ay%7D%20%3D%20%5Cfrac%7B%5Csin%20%5Cleft%28x%5Cright%29%5Csin%20%5Cleft%28y%5Cright%29%7D%7B%5Cleft%28%5Ccos%20%5Cleft%28y%5Cright%29&plus;x%5E2%5Cright%29%5E2%7D)

At `x=π` and `y=π`,

f = 0, f<sub>x</sub> = 1/(1-π<sup>2</sup>) = −0.112744, f<sub>y</sub> = 0

```py
x = Dual(np.pi, {'x': 1})
y = Dual(np.pi, {'y': 1})

f = sin_d(x)/(cos_d(y)+(x**2))
print(f)
```

You will see the following output:

```py
f = 0.0
fx = -0.112745
fy = 0.0
```
