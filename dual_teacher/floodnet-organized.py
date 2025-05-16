import os
import shutil

# Define the base path
base_path = '/home/ubuntu/Dataset/FloodNet/Train'

# Define original image and label directories
train_org_img = os.path.join(base_path, 'train-org-img')
train_label_img = os.path.join(base_path, 'train-label-img')

# Define the new labeled/unlabeled output directories
dirs_to_create = {
    'labeled': {
        'img': os.path.join(base_path, 'train-org-img-l'),
        'mask': os.path.join(base_path, 'train-label-img-l')
    },
    'unlabeled': {
        'img': os.path.join(base_path, 'train-org-img-u'),
        'mask': os.path.join(base_path, 'train-label-img-u')
    }
}

# Create directories if they do not exist
for dset in dirs_to_create.values():
    os.makedirs(dset['img'], exist_ok=True)
    os.makedirs(dset['mask'], exist_ok=True)

# Function to move files
def move_files(txt_path, dest_dirs):
    with open(txt_path, 'r') as file:
        for line in file:
            base_name = line.strip()
            img_filename = f"{base_name}.jpg"
            mask_filename = f"{base_name}_lab.png"

            img_src = os.path.join(train_org_img, img_filename)
            mask_src = os.path.join(train_label_img, mask_filename)

            img_dst = os.path.join(dest_dirs['img'], img_filename)
            mask_dst = os.path.join(dest_dirs['mask'], mask_filename)

            if os.path.exists(img_src):
                shutil.move(img_src, img_dst)
            else:
                print(f"[Missing] Image not found: {img_src}")

            if os.path.exists(mask_src):
                shutil.move(mask_src, mask_dst)
            elif 'blank' in base_name:
                open(mask_dst, 'a').close()  # Create empty mask
                print(f"[Blank] Created empty mask: {mask_dst}")
            else:
                print(f"[Missing] Mask not found: {mask_src}")

# Process both labeled and unlabeled sets
move_files('/home/ubuntu/Dataset/FloodNet/floodnet-labeled.txt', dirs_to_create['labeled'])
move_files('/home/ubuntu/Dataset/FloodNet/floodnet-unlabeled.txt', dirs_to_create['unlabeled'])

print("✅ All images and masks have been moved.")
