students = eval(input())
students.sor(key=lambda x:x[2],reverse=True)
for student in students:
    print(student[0], student[1], student[2])
