import os, sys

import argparse
import random
import warnings
import json

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy import ndimage
from PIL import Image
from huggingface_hub import hf_hub_download
from segments.utils import bitmap2file

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
)
from GroundingDINO.groundingdino.util.inference import annotate, predict

# segment anything
from segment_anything import build_sam, SamPredictor

# CLIPSeg
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from pinmask import maskpinner
def load_model_hf(model_config_path, filename, device):
    args = SLConfig.fromfile(model_config_path)
    model = build_model(args)
    args.device = device
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    model = model.to(device)
    return model


def load_image_for_dino(image):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dino_image, _ = transform(image, None)
    return dino_image


def dino_detection(
    model,
    image,
    image_array,
    category_names,
    category_name_to_id,
    box_threshold,
    text_threshold,
    device,
    visualize=False,
):
    detection_prompt = " . ".join(category_names)
    dino_image = load_image_for_dino(image)
    dino_image = dino_image.to(device)
    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=model,
            image=dino_image,
            caption=detection_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )
    # category_ids=[]
    # for phrase in phrases:
    #     if phrase in category_name_to_id.keys():
    #         category_ids.append(category_name_to_id[phrase])
    category_ids = [category_name_to_id[phrase] for phrase in phrases]

    if visualize:
        annotated_frame = annotate(
            image_source=image_array, boxes=boxes, logits=logits, phrases=phrases
        )
        annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
        visualization = Image.fromarray(annotated_frame)
        return boxes, category_ids, visualization
    else:
        return boxes, category_ids, phrases


def sam_masks_from_dino_boxes(predictor, image_array, boxes, device):
    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_array.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_xyxy, image_array.shape[:2]
    ).to(device)
    thing_masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return thing_masks


def preds_to_semantic_inds(preds, threshold):
    flat_preds = preds.reshape((preds.shape[0], -1))
    # Initialize a dummy "unlabeled" mask with the threshold
    flat_preds_with_treshold = torch.full(
        (preds.shape[0] + 1, flat_preds.shape[-1]), threshold
    )
    flat_preds_with_treshold[1 : preds.shape[0] + 1, :] = flat_preds

    # Get the top mask index for each pixel
    semantic_inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices.reshape(
        (preds.shape[-2], preds.shape[-1])
    )

    return semantic_inds


def clipseg_segmentation(
    processor, model, image, category_names, background_threshold, device
):
    inputs = processor(
        text=category_names,
        images=[image] * len(category_names),
        padding="max_length",
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(0)
    # resize the outputs
    upscaled_logits = nn.functional.interpolate(
        logits.unsqueeze(1),
        size=(image.size[1], image.size[0]),
        mode="bilinear",
    )
    preds = torch.sigmoid(upscaled_logits.squeeze(dim=1))
    semantic_inds = preds_to_semantic_inds(preds, background_threshold)
    return preds, semantic_inds


def semantic_inds_to_shrunken_bool_masks(
    semantic_inds, shrink_kernel_size, num_categories
):
    shrink_kernel = np.ones((shrink_kernel_size, shrink_kernel_size))

    bool_masks = torch.zeros((num_categories, *semantic_inds.shape), dtype=bool)
    for category in range(num_categories):
        binary_mask = semantic_inds == category
        shrunken_binary_mask_array = (
            ndimage.binary_erosion(binary_mask.numpy(), structure=shrink_kernel)
            if shrink_kernel_size > 0
            else binary_mask.numpy()
        )
        bool_masks[category] = torch.from_numpy(shrunken_binary_mask_array)

    return bool_masks


def clip_and_shrink_preds(semantic_inds, preds, shrink_kernel_size, num_categories):
    # convert semantic_inds to shrunken bool masks
    bool_masks = semantic_inds_to_shrunken_bool_masks(
        semantic_inds, shrink_kernel_size, num_categories
    ).to(preds.device)

    sizes = [
        torch.sum(bool_masks[i].int()).item() for i in range(1, bool_masks.size(0))
    ]
    max_size = max(sizes)
    relative_sizes = [size / max_size for size in sizes] if max_size > 0 else sizes

    # use bool masks to clip preds
    clipped_preds = torch.zeros_like(preds)
    for i in range(1, bool_masks.size(0)):
        float_mask = bool_masks[i].float()
        clipped_preds[i - 1] = preds[i - 1] * float_mask

    return clipped_preds, relative_sizes


def sample_points_based_on_preds(preds, N):
    height, width = preds.shape
    weights = preds.ravel()
    indices = np.arange(height * width)

    # Randomly sample N indices based on the weights
    sampled_indices = random.choices(indices, weights=weights, k=N)

    # Convert the sampled indices into (col, row) coordinates
    sampled_points = [(index % width, index // width) for index in sampled_indices]

    return sampled_points


def upsample_pred(pred, image_source):
    pred = pred.unsqueeze(dim=0)
    original_height = image_source.shape[0]
    original_width = image_source.shape[1]

    larger_dim = max(original_height, original_width)
    aspect_ratio = original_height / original_width

    # upsample the tensor to the larger dimension
    upsampled_tensor = F.interpolate(
        pred, size=(larger_dim, larger_dim), mode="bilinear", align_corners=False
    )

    # remove the padding (at the end) to get the original image resolution
    if original_height > original_width:
        target_width = int(upsampled_tensor.shape[3] * aspect_ratio)
        upsampled_tensor = upsampled_tensor[:, :, :, :target_width]
    else:
        target_height = int(upsampled_tensor.shape[2] * aspect_ratio)
        upsampled_tensor = upsampled_tensor[:, :, :target_height, :]
    return upsampled_tensor.squeeze(dim=1)


def sam_mask_from_points(predictor, image_array, points):
    points_array = np.array(points)
    # we only sample positive points, so labels are all 1
    points_labels = np.ones(len(points))
    # we don't use predict_torch here cause it didn't seem to work...
    _, _, logits = predictor.predict(
        point_coords=points_array,
        point_labels=points_labels,
    )
    # max over the 3 segmentation levels
    total_pred = torch.max(torch.sigmoid(torch.tensor(logits)), dim=0)[0].unsqueeze(
        dim=0
    )
    # logits are 256x256 -> upsample back to image shape
    upsampled_pred = upsample_pred(total_pred, image_array)
    return upsampled_pred


def inds_to_segments_format(
    panoptic_inds, thing_category_ids, stuff_category_names, category_name_to_id
):
    panoptic_inds_array = panoptic_inds.numpy().astype(np.uint32)
    bitmap_file = bitmap2file(panoptic_inds_array, is_segmentation_bitmap=True)
    segmentation_bitmap = Image.open(bitmap_file)

    stuff_category_ids = [
        category_name_to_id[stuff_category_name]
        for stuff_category_name in stuff_category_names
    ]

    unique_inds = np.unique(panoptic_inds_array)
    stuff_annotations = [
        {"id": i+5000, "category_id": stuff_category_ids[i - 1]}
        for i in range(1, len(stuff_category_names) + 1)
        if i in unique_inds
    ]
    thing_annotations = [
        {"id": len(stuff_category_names) + 1 + i+5000, "category_id": thing_category_id}
        for i, thing_category_id in enumerate(thing_category_ids)
    ]
    annotations = stuff_annotations + thing_annotations

    return segmentation_bitmap, annotations


def generate_panoptic_mask(
    imagebatch,
    thing_category_names_string,
    stuff_category_names_string,
    maskbatch_input,
    dino_box_threshold=0.3,
    dino_text_threshold=0.25,
    segmentation_background_threshold=0.1,
    shrink_kernel_size=20,
    num_samples_factor=1000,
    task_attributes_json="",
):
    thing_category_names = [
        thing_category_name.strip()
        for thing_category_name in thing_category_names_string.split(",")
    ]
        # parse inputs
    stuff_category_names = [
        stuff_category_name.strip()
        for stuff_category_name in stuff_category_names_string.split(",")
    ]
    category_names = thing_category_names + stuff_category_names
    category_name_to_id = {
        category_name: i for i, category_name in enumerate(category_names)
    }         #category:id
    pinner=maskpinner()
    colordict=pinner.colordict(maskbatch_input)
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    name=imagebatch[0].split('/')[-3]
    out = cv2.VideoWriter(f'/mnt/ve_share/liushuai/panoptic-segment-anything/results/{name}.mp4', fourcc, 5, (1280,800))
    import tqdm
    for image,mask in tqdm.tqdm(zip(imagebatch,maskbatch_input)):
        imagepath=image
        image=Image.open(image)
        image = image.convert("RGB")
        image_array = np.asarray(image)
        # compute SAM image embedding
        sam_predictor.set_image(image_array)
        # detect boxes for "thing" categories using Grounding DINO
        thing_category_ids = []
        thing_masks = []
        thing_boxes = []
        detected_thing_category_names = []
        if len(thing_category_names) > 0:
            thing_boxes, thing_category_ids, detected_thing_category_names = dino_detection(
            dino_model,
            image,
            image_array,
            thing_category_names,
            category_name_to_id,
            dino_box_threshold,
            dino_text_threshold,
            device,
        )
        if len(thing_boxes) > 0:
            # get segmentation masks for the thing boxes
            thing_masks = sam_masks_from_dino_boxes(
                sam_predictor, image_array, thing_boxes, device
            )  #由dino得到thing的mask

        if len(stuff_category_names) > 0:
            # get rough segmentation masks for "stuff" categories using CLIPSeg
            clipseg_preds, clipseg_semantic_inds = clipseg_segmentation(
                clipseg_processor,
                clipseg_model,
                image,
                stuff_category_names,
                segmentation_background_threshold,
                device,
            )
            # remove things from stuff masks
            clipseg_semantic_inds_without_things = clipseg_semantic_inds.clone()
            if len(thing_boxes) > 0:
                combined_things_mask = torch.any(thing_masks, dim=0) #如果这些位置有object 没有stuff
                clipseg_semantic_inds_without_things[combined_things_mask[0]] = 0
            # clip CLIPSeg preds based on non-overlapping semantic segmentation inds (+ optionally shrink the mask of each category)
            # also returns the relative size of each category
            clipsed_clipped_preds, relative_sizes = clip_and_shrink_preds(
                clipseg_semantic_inds_without_things,
                clipseg_preds,
                shrink_kernel_size,
                len(stuff_category_names) + 1,
            )
            # get finer segmentation masks for the "stuff" categories using SAM
            sam_preds = torch.zeros_like(clipsed_clipped_preds)
            for i in range(clipsed_clipped_preds.shape[0]):
                clipseg_pred = clipsed_clipped_preds[i]
                # for each "stuff" category, sample points in the rough segmentation mask
                num_samples = int(relative_sizes[i] * num_samples_factor)
                if num_samples == 0:
                    continue
                points = sample_points_based_on_preds(
                    clipseg_pred.cpu().numpy(), num_samples
                )
                if len(points) == 0:
                    continue
                # use SAM to get mask for points
                pred = sam_mask_from_points(sam_predictor, image_array, points)
                sam_preds[i] = pred
            sam_semantic_inds = preds_to_semantic_inds(
                sam_preds, segmentation_background_threshold
            )
        panoptic_inds = (
            sam_semantic_inds.clone()
            if len(stuff_category_names) > 0
            else torch.zeros(image_array.shape[0], image_array.shape[1], dtype=torch.long)
        )
        ind = len(stuff_category_names) + 1
        for thing_mask in thing_masks:
                # overlay thing mask on panoptic inds
            panoptic_inds[thing_mask.squeeze(dim=0)] = ind
            ind += 1
        
        panoptic_bool_masks = (
            semantic_inds_to_shrunken_bool_masks(panoptic_inds, 0, ind + 1)
            .numpy()
            .astype(int)
        )
        panoptic_names = (
            ["unlabeled"] + stuff_category_names + detected_thing_category_names
        )
        subsection_label_pairs = [
            (panoptic_bool_masks[i], panoptic_name)
            for i, panoptic_name in enumerate(panoptic_names)
        ]

        segmentation_bitmap, annotations = inds_to_segments_format(
            panoptic_inds, thing_category_ids, stuff_category_names, category_name_to_id
        )
        panoptic_inds+=5000

        vis=lambda x:(panoptic_inds.numpy()==x).astype("uint8")*255
        #3,  4,  9, 10, 11, 12, 13, 14
        panopticmask=panoptic_inds.numpy()
        panopticmask[panopticmask==5000]=0
        pinner.updatecolordict(panopticmask,annotations,category_names)
        pinner.getpinnedmask(panopticmask,mask)
        pinnedimage=pinner.drawmask(imagepath)
        out.write(pinnedimage)
    out.release()


config_file = "/mnt/ve_share/liushuai/panoptic-segment-anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
ckpt_filename = "/mnt/ve_share/liushuai/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
sam_checkpoint = "/mnt/ve_share/liushuai/SegmentAnyRGBD-main/sam_vit_h_4b8939.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
device="cpu"
print("Using device:", device)

if device != "cpu":
    try:
        from GroundingDINO.groundingdino import _C
    except:
        warnings.warn(
            "Failed to load custom C++ ops. Running on CPU mode Only in groundingdino!"
        )

# initialize groundingdino model
dino_model = load_model_hf(config_file, ckpt_filename, device)

# initialize SAM
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)

clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
    "CIDAS/clipseg-rd64-refined"
)
clipseg_model.to(device)

if __name__ == "__main__":
    things="person, car, motorcycle, truck, bird, dog, handbag, suitcase, bottle, cup, bowl, chair, potted plant, bed, dining table, tv, laptop, cell phone, bag, bin, box, door, road barrier, stick"
    stuff="floor, ground, wall, window, stair, fence, grass, sky"
    RGBroot="/mnt/ve_share/liushuai/panoptic-segment-anything/sol_5_mcs_1/images"
    maskroot="/mnt/ve_share/liushuai/panoptic-segment-anything/sol_5_mcs_1/visible"
    getbatch=lambda x:sorted(os.path.join(x,i) for i in os.listdir(x))
    rgbbatch_input=getbatch(RGBroot)
    maskbatch_input=getbatch(maskroot)
    # inputbatch=["/mnt/ve_share/liushuai/panoptic-segment-anything/sol_5_mcs_1/images/000000.bmp"]
    # maskbatch_input=["/mnt/ve_share/liushuai/panoptic-segment-anything/sol_5_mcs_1/visible/000000.npy"]
    generate_panoptic_mask(rgbbatch_input,things,stuff,maskbatch_input)

