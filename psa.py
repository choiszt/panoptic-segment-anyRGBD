import random
import requests

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from scipy import ndimage
from PIL import Image
from huggingface_hub import hf_hub_download
from segments import SegmentsClient
from segments.export import colorize
from segments.utils import bitmap2file
from getpass import getpass

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, predict
device = "cuda" if torch.cuda.is_available() else "cpu"
# segment anything
from segment_anything import build_sam, SamPredictor

# CLIPSeg
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
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
def generate_panoptic_mask(
    image,
    thing_category_names,
    stuff_category_names,
    category_name_to_id,
    dino_model,
    sam_predictor,
    clipseg_processor,
    clipseg_model,
    device,
    dino_box_threshold=0.3,
    dino_text_threshold=0.25,
    segmentation_background_threshold=0.1,
    shrink_kernel_size=20,
    num_samples_factor=1000,
):
    image = image.convert("RGB")
    image_array = np.asarray(image)

    # compute SAM image embedding
    sam_predictor.set_image(image_array)

    # detect boxes for "thing" categories using Grounding DINO
    thing_category_ids = []
    thing_masks = []
    thing_boxes = []
    if len(thing_category_names) > 0:
        thing_boxes, thing_category_ids, _ = dino_detection(
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
            )

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
            combined_things_mask = torch.any(thing_masks, dim=0)
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

    # combine the thing inds and the stuff inds into panoptic inds
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

    return panoptic_inds, thing_category_ids