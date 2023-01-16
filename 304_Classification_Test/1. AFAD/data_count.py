import glob
import os

path = './AFAD_remove/'

for i in range(2, 10):
    all_data_path = glob.glob(os.path.join(path, str(i) + '*', '112', '*'))
    print(i*10, len(all_data_path))
