import os
valid_names = []

with open('rescuenet-labeled.txt', "r") as f:
    valid_names = set(line.strip() for line in f)

# Prepare output lists
matched = []
unmatched = []

datasets = ['RescueNet']
for dataset in datasets:
    for image in os.listdir(f'../../../../data/jpk322/{dataset}/train-set/train-org-img'):
        if image[:-4] in valid_names:
            matched.append(f'train-set/train-org-img/{image} train-set/train-label-img/{image[:-4]}_lab.png')
        else:
            unmatched.append(f'train-set/train-org-img/{image} train-set/train-unlabel-img/blank.png')
# Save results
with open("rescuenet/train-set/labeled.txt", "w") as f:
    f.write("\n".join(matched))

with open("rescuenet/train-set/unlabeled.txt", "w") as f:
    f.write("\n".join(unmatched))

print(f"Labeled: {len(matched)} paths")
print(f"Unlabeled: {len(unmatched)} paths")

val = []

for dataset in datasets:
    for image in os.listdir(f'../../../../data/jpk322/{dataset}/val/val-org-img'):
        val.append(f'val/val-org-img/{image} val/val-label-img/{image[:-4]}_lab.png')

with open("rescuenet/val.txt", "w") as f:
    f.write("\n".join(val))

print(f"validation: {len(val)} paths")