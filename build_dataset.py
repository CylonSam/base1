from msilib import sequence
from os import listdir
from os.path import isfile, join
from pathlib import Path
from os import walk
import random

cwd = Path.cwd()

data_path = Path(cwd, "data")

apnea_src_path = Path(cwd, "apnea2")
normal_src_path = Path(cwd, "normal2")

apnea_imgs = []
normal_imgs = []

for (dirpath, dirnames, filenames) in walk(apnea_src_path):
    apnea_imgs.extend(filenames)
    break

for (dirpath, dirnames, filenames) in walk(normal_src_path):
    normal_imgs.extend(filenames)
    break

apnea_imgs.sort()
normal_imgs.sort()

random.seed(3)

# chosen_normal_imgs = random.sample(normal_imgs, k=imgs_max_quantity)
# chosen_apnea_imgs = apnea_imgs

# random.shuffle(chosen_normal_imgs)
# random.shuffle(chosen_apnea_imgs)


# for filename in chosen_normal_imgs:
# #     # print(Path(normal_src_path, filename))
#     Path(Path(normal_src_path, filename).as_posix()).rename(Path(data_path, "normal", filename).as_posix())


# for filename in chosen_apnea_imgs:
# #     # print(Path(normal_src_path, filename))
#     Path(Path(apnea_src_path, filename).as_posix()).rename(Path(data_path, "apnea", filename).as_posix())



current_img_quantity = 0
# para cada conjunto de imagens de um paciente
for p in range(22):
    print(f"Processing images from patient #{p}")
    apnea_imgs_p = list(filter(lambda x : x.split('_')[0] == str(p), apnea_imgs))

    current_img_quantity = len(list(apnea_imgs_p))

    print(f"Moving {current_img_quantity} apnea images from patient #{p}...")

    for filename in apnea_imgs_p:
        Path(Path(apnea_src_path, filename).as_posix()).rename(Path(data_path, "apnea", filename).as_posix())

    print("Done.")

    print(f"Choosing {current_img_quantity} random normal images from patient...")

    normal_imgs_p = list(filter(lambda x : x.split('_')[0] == str(p), normal_imgs))
    chosen_normal_imgs = random.sample(normal_imgs_p, k=(current_img_quantity if current_img_quantity <= len(normal_imgs_p) else len(normal_imgs_p)))

    print(f"Moving {current_img_quantity} normal images from patient #{p}...")

    for filename in chosen_normal_imgs:
        Path(Path(normal_src_path, filename).as_posix()).rename(Path(data_path, "normal", filename).as_posix())

    print("Done.")
