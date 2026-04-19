import os

from .inference import predict_probabilities
from .io import save_binary_mask, save_prediction_outputs
from .metrics import dice_score, evaluate_against_gt, load_binary_mask
from .postprocess import binarize_prediction, postprocess_binary_mask, restore_prediction_to_full
from .preprocess import crop_image_with_roi, load_image_rgb, normalize_roi_box
from .types import PredictionResult
from .visualize import build_preview_arrays, save_prediction_preview


def predict_image_result(
    model,
    image_path,
    device,
    *,
    threshold=0.5,
    apply_postprocessing=True,
    roi_box=None,
    mode="tile",
    input_size=256,
    tile_overlap=32,
    tile_batch_size=4,
    gt_mask_path=None,
    stop_checker=None,
):
    """Run prediction and return arrays/metrics without saving any files."""
    full_img = load_image_rgb(image_path)
    cropped_img, roi_box = crop_image_with_roi(full_img, roi_box)

    pred = predict_probabilities(
        model,
        cropped_img,
        device,
        mode=mode,
        input_size=input_size,
        tile_overlap=tile_overlap,
        tile_batch_size=tile_batch_size,
        stop_checker=stop_checker,
    )
    binary_mask = binarize_prediction(pred, threshold)
    binary_mask = postprocess_binary_mask(binary_mask, apply_postprocessing)

    pred, binary_mask = restore_prediction_to_full(pred, binary_mask, roi_box, full_img.size)
    gt_mask, dice = evaluate_against_gt(binary_mask, gt_mask_path, target_size=full_img.size)

    return PredictionResult(
        image_path=image_path,
        image=full_img,
        pred=pred,
        binary_mask=binary_mask,
        roi_box=roi_box,
        dice=dice,
        gt_mask=gt_mask,
        gt_mask_path=gt_mask_path,
        original_size=full_img.size,
    )


def predict_image(
    model,
    image_path,
    device,
    threshold=0.5,
    output_dir="results",
    apply_postprocessing=True,
    output_basename=None,
    gt_mask_path=None,
    gt_expected=False,
    return_details=False,
    roi_box=None,
    mode="tile",
    input_size=256,
    tile_overlap=32,
    tile_batch_size=4,
    stop_checker=None,
):
    """Run crack segmentation on a single image using a trained model."""
    img_name = os.path.basename(image_path)
    base_name = output_basename or os.path.splitext(img_name)[0]

    result = predict_image_result(
        model,
        image_path,
        device,
        threshold=threshold,
        apply_postprocessing=apply_postprocessing,
        roi_box=roi_box,
        mode=mode,
        input_size=input_size,
        tile_overlap=tile_overlap,
        tile_batch_size=tile_batch_size,
        gt_mask_path=gt_mask_path,
        stop_checker=stop_checker,
    )

    output_path, mask_path = save_prediction_outputs(
        result,
        output_dir,
        base_name,
        threshold=threshold,
        gt_expected=gt_expected,
    )

    print(f"Prediction saved to: {output_path}")
    print(f"Binary mask saved to: {mask_path}")
    if gt_mask_path and result.dice is not None:
        print(f"Ground truth: {gt_mask_path}")
        print(f"Dice: {result.dice:.4f}")
    elif gt_expected:
        if gt_mask_path:
            print(f"Warning: GT mask not found/invalid: {gt_mask_path}")
        else:
            print("Warning: GT mask not provided (gt_expected=True)")
    if return_details:
        return {
            "output_path": output_path,
            "mask_path": mask_path,
            "dice": result.dice,
            "gt_mask_path": gt_mask_path,
            "roi_box": result.roi_box,
        }
    return output_path
