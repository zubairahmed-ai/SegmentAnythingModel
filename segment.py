import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="./models/sam_vit_b_01ec64.pth",
	help="path to the model")
ap.add_argument("-mk", "--model_key", type=str, default="vit_b",
	help="path to the model key required by SAM")
ap.add_argument("-i", "--image", type=str, default="./images/IMG_2176.jpg",
	help="path to the input image")
ap.add_argument("-p", "--points_per_side", type=int, default=32,
	help="points per side")
ap.add_argument("-pr", "--pred_iou_thresh", type=float, default=0.86,
	help="prediction iou threshold")
ap.add_argument("-s", "--stability_score_thresh", type=float, default=0.95,
	help="path to the input image")
ap.add_argument("-c", "--crop_n_layers", type=int, default=1,
	help="path to the input image")
ap.add_argument("-cr", "--crop_n_points_downscale_factor", type=int, default=2,
	help="path to the input image")
ap.add_argument("-mm", "--min_mask_region_area", type=int, default=100,
	help="path to the input image")
args = vars(ap.parse_args())

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

sam = sam_model_registry[args["model_key"]](checkpoint=args["model"])
mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=args["points_per_side"],
    pred_iou_thresh=args["pred_iou_thresh"],
    stability_score_thresh=args["stability_score_thresh"],
    crop_n_layers=args["crop_n_layers"],
    crop_n_points_downscale_factor=args["crop_n_points_downscale_factor"],
    min_mask_region_area=args["min_mask_region_area"],)  # Requires open-cv to run post-processing)

# load the original input image and display it to our screen
image = cv2.imread(args["image"])
# cv2.waitKey(0)

masks = mask_generator.generate(image)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 

# Uncomment the following to view each mask invidually overlayed on image

# for i, mask_data in enumerate(masks):
#     # cv2.imshow("Original", image)
#     mask = mask_data["segmentation"]
#     iou = mask_data["predicted_iou"]
#     score = mask_data["stability_score"]
#     # show_mask(mask, image)
#     plt.figure(figsize=(10,10))
#     plt.imshow(image)
#     show_mask(mask, plt.gca())    
#     plt.title(f"Mask {i+1}, Score: {score:.3f} , IoU: {iou:.3f}", fontsize=18)
#     plt.axis('off')
#     plt.show()      
