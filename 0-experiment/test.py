import os
import shutil

dir_path_data = "C:/Users/HP/Desktop/Warwick/Hackathon/data/SYTO-9_WGA647_BF_ECOLI_STAPH_1_10000/pos_0"
dir_path = "C:/Users/HP/Desktop/Warwick/Hackathon/BioImage/data/pos_84"
for file in os.listdir(dir_path_data):
    if file.endswith(".tif"):
        print(file)
        shutil.copyfile(dir_path_data, dir_path)