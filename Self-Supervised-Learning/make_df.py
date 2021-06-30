import pandas as pd 
import os


train_folder = "images3/val"
train_files = os.listdir(train_folder)

print("no of files: ",len(train_files))

#data = pd.DataFrame(columns = ['id','label'])
#data['id'] = 0
#data['label'] = 0
data = []
i = 0
for file_name in train_files:
	file_name = file_name[:-5]
	if file_name[:7] == "lungaca":
		i = i+1
		data.append([file_name, 0])
		continue
	if file_name[:5] == "lungn":
		i = i+1
		data.append([file_name,1])
		continue

	if file_name[:7] == "lungscc":
		data.append([file_name, 2])
		i = i+1
		continue

final_df = pd.DataFrame(data, columns = ['id','label'])

final_df.to_csv('val.csv',index = False)