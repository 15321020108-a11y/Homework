def f(type=None,*a, **kwa):
	pass

#error，默认参数不能跟在**后面
def f(*a, **kwa, type=None):
#def f(*a, type=None, **kwa): #正确的写法
	pass

#默认参数可以在*的前面或后面
def f(*a, type=None):
	pass

f()