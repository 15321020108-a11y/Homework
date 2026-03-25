def append_ele(a,l=[]):
	'''
	演示默认值累积
	'''
	print(hex(id(l)))
	l.append(a)
	return l

print(append_ele(1))
print(append_ele(2))
print(append_ele(3))