import os
import shutil

dir_path_data = "C:/Users/HP/Desktop/Warwick/Hackathon/data/SYTO-9_WGA_STAPH_PLL/pos_0"
dir_path = "C:/Users/HP/Desktop/Warwick/Hackathon/BioImage/data/pos_102"
for file in os.listdir(dir_path_data):
    if file.endswith(".tif"):
        print(file)
        shutil.copyfile(dir_path_data, dir_path)
