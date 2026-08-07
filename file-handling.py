file =  open("student.txt", "r")
data = file.read()
print(data)
file.close()

file = open("student.csv","w")
data = file.write("sanskar\n")
print(data)
file.close()

#with moder code 

with open("school.csv","r") as file 
     data = file.read()
     print(data)