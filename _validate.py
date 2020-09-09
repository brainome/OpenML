import os

predictors=[pred for pred in next(os.walk("./Predictors/"))[2]]

for pred in predictors:
	print("-"*50)
	if pred[:len("binary.")]=="binary.":
		data=pred[len("binary."):].replace(".py", ".csv")
	elif pred[:len("multi.")]=="multi.":
		data=pred[len("multi."):].replace(".py", ".csv")
	else:
		data=pred.replace(".py", ".csv")
	res=os.system(f"python3 Predictors/{pred} Data/{data} -validate")
	if res!=0:
		print("error encountered")
		break
