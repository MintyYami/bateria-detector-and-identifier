import numpy as np
from skimage import filters, morphology, measure, exposure
import cv2

# processing all 3 channels of img
def process_all(function, imgs):
    imgs_processed = imgs.copy()
    for i in range(3):
        img = function(imgs_processed[i])
        imgs_processed[i] = img
    return imgs_processed

# segmentation on bright areas (bateria)
def __segment_bright_bacteria(
    image: np.ndarray,
    min_area: int = 3,
    max_area: int = 200,
    sigma_bg: float = 10,
    threshold_scale: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:

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

    # Clean objects
    binary = morphology.remove_small_objects(binary, max_size=min_area - 1)
    binary = morphology.remove_small_holes(binary, max_size=1)  # since old was area_threshold=2

    # Lightly close small gaps
    binary = morphology.closing(binary, morphology.disk(1))

    # Label objects
    labelled = measure.label(binary)

    # Filter by object area
    filtered_mask = np.zeros_like(labelled, dtype=np.uint16)
    new_label = 1

    for region in measure.regionprops(labelled):
        if min_area <= region.area <= max_area:
            filtered_mask[labelled == region.label] = new_label
            new_label += 1

    return filtered_mask, corrected

def getSegments(imgs):
    ret = []
    for i in range(3):
        masks, corrected = __segment_bright_bacteria(
            imgs[i],
            min_area=3,
            max_area=200,
            sigma_bg=10,
            threshold_scale=0.6
        )
        ret.append([masks, corrected])
    return ret

# def get