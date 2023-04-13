# Python Script for Facebook Research's Segment Anything Model
This script uses command line argument to show [Automatically generating object masks with SAM](https://github.com/facebookresearch/segment-anything) with mask generation options support using arguments

![SAM](https://user-images.githubusercontent.com/1317442/231848591-7cf0f597-4298-4088-961b-8099b6a101b5.png)

And with arguments support for generating masks

```
mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)
```


![download (1)](https://user-images.githubusercontent.com/1317442/231849479-e6d53b30-cba6-4599-8e22-c421dda319c0.png)

With support for viewing individual Predicted IOU, Stability score with the segmenation

You can use it on command line with 

`python segment.py --image ./<image path>`
