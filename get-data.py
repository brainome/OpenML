import os

with open("datalinks.txt") as f:
	lines=[line.strip().split(",") for line in f]

if not os.path.exists("Data/"):
	os.mkdir("Data/")

for link,name in lines:
	print(f"### Downloading {name} from openml.org")
	if not os.path.exists(f"Data/{name}"):
		res=os.system(f"wget -q -O Data/{name}.csv {link}")
	if res!=0:
		print("An error occured.")
		break
	if not os.path.exists(f"Data/{name}.csv"):
		print("### ERROR - $Name.csv did not download")