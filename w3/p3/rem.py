def get_summary_statistics(nl:list)->list:
	'''
	该函数用来表示一个可能返回多个（超过三个）结果的场景
	且其功能可能有极大的变化可能
	这里演示时仅返回所有归一并排完序的元素，调用时仅捕获最大值和最小值
	'''
	aver=sum(nl)/len(nl)
	nls=[e/aver for e in nl]
	nls.sort(reverse=True)
	return nls

def main():
	nl=list(range(101))
	smax,*middle,smin=get_summary_statistics(nl)
	print(f'scaled_max={smax}, scaled_min={smin}')

if __name__=='__main__':main()
