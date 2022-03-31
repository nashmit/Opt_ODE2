from typing import Union
from casadi import *

def test( input: Union[ SX, float ] ) -> Union[ SX, float ]:
    return input * 2 + 1

    pass

print( test(2) )
value = SX.sym('value',1)
print( test(value) )
