def func(i,j=10,*args):
	print(f'j={j}, args={args}')

func(0, 2, 3, 4, 5)
#特别注意，这样的调用是错误的
#func(0, j=12, 2, 3, 4, 5)#SyntaxError: positional argument follows keyword argument
func(0)

def func_2(i:int,*args,j=10):
	print(f'j={j}, args={args}')

func_2(0, 2, 3, 4, 5)#
func_2(0, 2, 3, 4, 5, j=12)
func_2(0)