from casadi import *

# for more details, including different type of atomic integration check:
#https://gist.github.com/ghorn/9757680
#also
# https://groups.google.com/g/casadi-users/c/xkUCC6csmuQ/m/DKgoHzJQV-oJ


def func(a,b):
    return a+b
    pass

a = SX.sym('a',1)
b = SX.sym('b',1)
c = func(a,b)
print(c)