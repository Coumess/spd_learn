import pickle
import os 
import glob
import json

base_dir = "C:/Users/coumesa/Documents/BCI/data/"

folds_paths = ["processed_data_BNCI2014001", "processed_data_BNCI2014004", "processed_data_Cho2017", "processed_data_GrosseWentrup2009", "processed_data_Lee2019_MI",
                "processed_data_schirr", "processed_data_Weibo2014", "processed_data_Zhou2016"
                ]

database_info = {}

for path in folds_paths:
    final_path = os.path.join(base_dir, path)
    
    files = glob.glob(os.path.join(final_path, "**/*.pkl"), recursive=True)

    with open(files[0], "rb") as f:
        data = pickle.load(f)

        dim = {}
        for key in data.keys():
            dim[key] = list(data[key][0].shape)
    
        database_info[path] = {
            'Dim': dim,
            
        }
with open('database_info.json', 'w') as j:
    json.dump(database_info, j, indent=4)