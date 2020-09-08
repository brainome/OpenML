import os
import pickle

d=pickle.load(open("all_predictors.pickle", "rb"))
predictors=[("Predictors/"+pred, pred.replace("binary.", "").replace("multi.", "").replace(".py", ".csv"), int(d["ncorrect"][pred.replace(".py", "")])) for pred in [x for x in next(os.walk("./Predictors/"))[2]]]

for pred,data,ncorrect in predictors:
	os.system(f"python3 {pred} Data/{data} -validate > dummy.txt")


	with open("dummy.txt") as f:
		text=f.read()
	start=text.find("Model accuracy:")
	subtext=text[start:]
	end=subtext.find("correct")+len("correct")
	subtext=subtext[:end].strip()
	res=int(subtext.split()[-2].replace("(", "").split("/")[0])
	if res!=ncorrect:
		print(data)
		raise Exception