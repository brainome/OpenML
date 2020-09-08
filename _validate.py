import os

predictors=[("Predictors/"+pred, pred.replace("binary.", "").replace("multi.", "").replace(".py", ".csv")) for pred in [x for x in next(os.walk("./Predictors/"))[2]]]

for pred,data,ncorrect in predictors:
	os.system(f"python3 {pred} Data/{data} -validate")
