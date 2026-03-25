while False:
	pass
else:
	print("while else")

x=10
while x:
	x-=1
else:
	print(f"else in while: x={x}")

for x in range(5):
	print(f'x={x}')
else:
	print(f'else in for_1: x={x}')

for x in range(1,5):
	print(f'x={x}')
	if x % 2 == 0:
		break
else:
	print(f'else in for_2: x={x}')


def outer(x):
	def inner(y):
		nonlocal x
		x+=y
		return x
	return inner

f1=outer(10)
print(f1(1))
print(f1(2))
print(f1(3))

print(outer(10)(1))
print(outer(10)(2))


