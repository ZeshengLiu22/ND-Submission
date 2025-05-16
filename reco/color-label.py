import os
import numpy as np
from PIL import Image

# import your colormap constructors
from build_data import create_floodnet_label_colormap, create_rescuenet_label_colormap

def color_encode_folder(label_dir, output_dir, colormap_fn, exts=('.png', '.jpg')):
    """
    Reads every file in label_dir with extension in exts,
    treats it as a 2D array of class indices, applies the colormap,
    and saves a colorized PNG in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    colormap = colormap_fn()  # should be an array of shape [num_classes, 3]

    for fname in os.listdir(label_dir):
        if not fname.lower().endswith(exts):
            continue
        # load label mask as numpy array
        lbl = np.array(Image.open(os.path.join(label_dir, fname)))
        # apply colormap: shape (H, W, 3)
        colored = colormap[lbl]
        # convert back to PIL and save
        out = Image.fromarray(colored.astype(np.uint8))
        out.save(os.path.join(output_dir, fname))

if __name__ == "__main__":
    # Example usage for FloodNet labels
    color_encode_folder(
        label_dir='dataset/floodnet/validationset/val-label-img',
        output_dir='results/floodnet/colored_labels',
        colormap_fn=create_floodnet_label_colormap
    )

    # # Example usage for RescueNet labels
    # color_encode_folder(
    #     label_dir='dataset/rescuenet/validationset/val-label-img',
    #     output_dir='results/rescuenet/colored_labels',
    #     colormap_fn=create_rescuenet_label_colormap
    # )
