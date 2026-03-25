from typing import List
from typing import Container
from typing import Dict
from typing import Tuple
from typing import TypeVar

def f(word:str, dic:Container[str])->List[str]:
    pass
def add(a:str, b:int)->str:
    return a+str(b)

T = TypeVar('T', int, float)
def sum(t:Tuple[T,T])->T:
	return t[0]+t[1]
f('a',123)
add('123','4')
print(sum((2,3)))
print(sum((2.3,4)))
print(sum((2,'test')))
