from msilib import sequence
from os import listdir
from os.path import isfile, join
from pathlib import Path
from os import walk
import random

cwd = Path.cwd()

train_path = Path(cwd, "data", "train")
test_path = Path(cwd, "data", "test")

apnea_src_path = Path(cwd, "apnea")
normal_src_path = Path(cwd, "normal")

apnea_imgs = []
normal_imgs = []

for (dirpath, dirnames, filenames) in walk(apnea_src_path):
    apnea_imgs.extend(filenames)
    break

for (dirpath, dirnames, filenames) in walk(normal_src_path):
    normal_imgs.extend(filenames)
    break


imgs_max_quantity = len(apnea_imgs)

print(imgs_max_quantity)

apnea_imgs.sort()
normal_imgs.sort()

random.seed(42)

chosen_normal_imgs = random.sample(normal_imgs, k=imgs_max_quantity)
chosen_apnea_imgs = apnea_imgs

random.shuffle(chosen_normal_imgs)
random.shuffle(chosen_apnea_imgs)

split = int(0.8 * imgs_max_quantity)

# print(split)

# print(chosen_normal_imgs[:split])

# Path("path/to/current/file.foo").rename("path/to/new/destination/for/file.foo")

for filename in chosen_normal_imgs[:split]:
#     # print(Path(normal_src_path, filename))
    Path(Path(normal_src_path, filename).as_posix()).rename(Path(train_path, "normal", filename).as_posix())

for filename in chosen_normal_imgs[split:]:
#     # print(Path(normal_src_path, filename))
    Path(Path(normal_src_path, filename).as_posix()).rename(Path(test_path, "normal", filename).as_posix())


for filename in chosen_apnea_imgs[:split]:
#     # print(Path(normal_src_path, filename))
    Path(Path(apnea_src_path, filename).as_posix()).rename(Path(train_path, "apnea", filename).as_posix())

for filename in chosen_apnea_imgs[split:]:
#     # print(Path(normal_src_path, filename))
    Path(Path(apnea_src_path, filename).as_posix()).rename(Path(test_path, "apnea", filename).as_posix())

# print()