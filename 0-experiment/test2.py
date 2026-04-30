#%%
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import logging
from typing import Union

from skimage import io
from skimage.measure import regionprops
from skimage import filters, morphology, measure, exposure
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt

from cellpose_omni import models


def run_omnipose(
    images: np.array,
    model_name: str,
    gpu_option: bool,
    chans: tuple or None = None,
    filter_options: dict = {}
) -> Union[np.array, np.array]:
    """
    Run the specified Omnipose/Cellpose model on the input image(s).
    """

    if chans is None:
        chans = [0, 0]

    model = models.CellposeModel(
        gpu=gpu_option,
        model_type=model_name,
        nchan=2
    )

    params = {
        "channels": chans,
        "diameter": 4,
        "rescale": None,
        "mask_threshold": -3,
        "flow_threshold": 0.0,
        "transparency": True,
        "omni": True,
        "cluster": True,
        "resample": True,
        "tile": True,
        "niter": None,
        "augment": False,
        "affinity_seg": False,
    }

    tic = time.time()

    if len(images.shape) == 2:
        try:
            masks, flows, styles = model.eval(images, **params)
        except Exception as e:
            logging.error(f"Error in Omnipose segmentation: {e}")
            logging.info("A blank mask will be returned.")
            masks = np.zeros_like(images, dtype=np.uint16)
            flows = np.zeros_like(images)
    else:
        n_images = len(images)
        n = range(n_images)

        masks_list, flows_list, styles = model.eval(
            [images[i] for i in n],
            **params
        )

        masks = np.dstack(masks_list)
        masks = np.moveaxis(masks, -1, 0)

        flows = np.copy(masks)

        for i, sublist in enumerate(flows_list):
            flows[i] = sublist[-1]

    net_time = time.time() - tic
    logging.info(f"Total segmentation time: {net_time}s")

    if filter_options:
        masks = filter_mask(masks, filter_options)

    return masks, flows


def segment_bright_bacteria(
    image: np.ndarray,
    min_area: int = 3,
    max_area: int = 200,
    sigma_bg: float = 10,
    threshold_scale: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment bright fluorescent bacteria using classical image processing.

    This method is often better for sparse, bright, small bacteria-like targets.

    Parameters
    ----------
    image:
        Input 2D fluorescence image.
    min_area:
        Minimum object area in pixels.
    max_area:
        Maximum object area in pixels.
    sigma_bg:
        Gaussian sigma used for background subtraction.
    threshold_scale:
        Multiplier for Otsu threshold.
        Lower values detect more objects.
        Try 0.6, 0.7, 0.8, 1.0.

    Returns
    -------
    filtered_mask:
        Labelled segmentation mask.
    corrected:
        Background-corrected and normalized image.
    """

    image = image.astype(np.float32)

    # Background subtraction
    background = filters.gaussian(image, sigma=sigma_bg)
    corrected = image - background
    corrected[corrected < 0] = 0

    # Normalize to 0-1
    corrected = exposure.rescale_intensity(
        corrected,
        in_range="image",
        out_range=(0, 1)
    )

    # Threshold
    otsu_threshold = filters.threshold_otsu(corrected)
    threshold = otsu_threshold * threshold_scale
    binary = corrected > threshold

    # Clean small noise
    binary = morphology.remove_small_objects(binary, min_size=min_area)
    binary = morphology.remove_small_holes(binary, area_threshold=2)

    # Optional: lightly close tiny gaps
    binary = morphology.binary_closing(binary, morphology.disk(1))

    # Label connected objects
    labelled = measure.label(binary)

    # Filter by object area
    filtered_mask = np.zeros_like(labelled, dtype=np.uint16)
    new_label = 1

    for region in measure.regionprops(labelled):
        if min_area <= region.area <= max_area:
            filtered_mask[labelled == region.label] = new_label
            new_label += 1

    return filtered_mask, corrected


def filter_mask(mask: np.array, options: dict = {}) -> np.array:
    """
    Filter segmentation mask based on area, length, and width.
    """

    min_area = options.get("min_area", None)
    max_area = options.get("max_area", None)

    min_length = options.get("min_length", None)
    max_length = options.get("max_length", None)

    min_width = options.get("min_width", None)
    max_width = options.get("max_width", None)

    label_mask = np.copy(mask)

    if len(label_mask.shape) == 2:
        regions = regionprops(label_mask)

        bacteria_indices = []
        filtered_mask = np.zeros_like(mask)

        for region in regions:
            area = region.area
            length = region.axis_major_length
            width = region.axis_minor_length

            if (
                (min_area is None or area >= min_area)
                and (max_area is None or area <= max_area)
                and (min_length is None or length >= min_length)
                and (max_length is None or length <= max_length)
                and (min_width is None or width >= min_width)
                and (max_width is None or width <= max_width)
            ):
                bacteria_indices.append(region.label)

        for i in bacteria_indices:
            filtered_mask[label_mask == i] = i

    elif len(label_mask.shape) == 3:
        filtered_mask = np.zeros_like(mask)

        for i, current_mask in enumerate(label_mask):
            current_filtered_mask = filter_mask(current_mask, options=options)
            filtered_mask[i] = current_filtered_mask

    else:
        raise ValueError("Mask must be 2D or 3D.")

    return filtered_mask


def show_overlay(image, masks):
    from skimage.segmentation import find_boundaries
    import matplotlib.pyplot as plt

    boundaries = find_boundaries(masks, mode="outer")

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray")
    plt.contour(boundaries, colors="red", linewidths=0.5)
    plt.axis("off")
    plt.title("Segmentation Overlay")
    plt.show()


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    image_path = input("Enter the path to the image file: ").strip()

    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        exit(1)

    image = io.imread(image_path)

    print(f"Loaded image with shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image min/max before processing: {image.min()} / {image.max()}")

    # If image has more than 2 dimensions, take the first channel/slice.
    # Your current image is already 2D, so this will usually not be used.
    if image.ndim > 2:
        print("Image has more than 2 dimensions. Using the first channel/slice.")
        image = image[..., 0]

    # ------------------------------------------------------------------
    # Choose segmentation method
    # ------------------------------------------------------------------
    # Recommended for your current fluorescence image:
    method = "classical"

    # Alternative:
    # method = "omnipose"
    # ------------------------------------------------------------------

    if method == "classical":
        masks, corrected = segment_bright_bacteria(
            image,
            min_area=3,
            max_area=200,
            sigma_bg=10,
            threshold_scale=0.8
        )

        flows = None

    elif method == "omnipose":
        image_float = image.astype(np.float32)
        image_float = exposure.rescale_intensity(
            image_float,
            in_range="image",
            out_range=(0, 1)
        )

        model_name = "bact_fluor"
        gpu_option = False

        masks, flows = run_omnipose(
            image_float,
            model_name,
            gpu_option
        )

        corrected = image_float

    else:
        raise ValueError("method must be either 'classical' or 'omnipose'.")

    print(f"Segmentation complete. Masks shape: {masks.shape}")
    print("Unique mask values:", np.unique(masks))
    print("Number of objects:", len(np.unique(masks)) - 1)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    base_path = os.path.splitext(image_path)[0]

    mask_output_path = base_path + "_mask.tif"
    corrected_output_path = base_path + "_corrected.tif"
    overlay_output_path = base_path + "_overlay.png"

    io.imsave(mask_output_path, masks.astype(np.uint16))

    corrected_display = exposure.rescale_intensity(
        corrected.astype(np.float32),
        in_range="image",
        out_range=(0, 255)
    ).astype(np.uint8)

    io.imsave(corrected_output_path, corrected_display)

    show_overlay(image, masks, overlay_output_path)

    print(f"Mask saved to: {mask_output_path}")
    print(f"Corrected image saved to: {corrected_output_path}")
    print(f"Overlay saved to: {overlay_output_path}")