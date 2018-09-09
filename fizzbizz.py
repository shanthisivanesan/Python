x=0
while (x!="q"):
	x = int(input("enter a number"))
	if (x=="q"):
		print("exit")
		break;
	else:
		x = int(x)
		if(x%5==0):
			print("Fizz")
		elif(x%3==0):
			print("Bizz")
	