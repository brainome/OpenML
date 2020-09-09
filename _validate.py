import os

predictors=[(pred, pred.replace("binary.", "").replace("multi.", "").replace(".py", ".csv")) for pred in [x for x in next(os.walk("./Predictors/"))[2]]]

for pred,data in predictors:
	print("-"*50)
	res=os.system(f"python3 Predictors/{pred} Data/{data} -validate")
	if res!=0:
		print("error encountered")
		break
