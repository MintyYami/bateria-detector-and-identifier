import numpy as np
from skimage import filters, morphology, measure, exposure
from skimage.measure import regionprops
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# processing all 3 channels of img
def process_all(function, imgs):
    imgs_processed = imgs.copy()
    for i in range(2):
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
    for i in range(2):
        masks, corrected = __segment_bright_bacteria(
            imgs[i],
            min_area=3,
            max_area=200,
            sigma_bg=10,
            threshold_scale=0.6
        )
        ret.append([masks, corrected])
    return ret

def getStats(maskCorrects, imgs):
    ret = []
    for i in range(2):
        measurements = []

        regions = regionprops(maskCorrects[i][0], intensity_image=imgs[i])

        for region in regions:
            label_id = region.label
            area = region.area
            y, x = region.centroid

            major_length = region.axis_major_length
            minor_length = region.axis_minor_length

            measurements.append({
                "label": label_id,
                "area": area,
                "x": x,
                "y": y,
                "major_length": major_length,
                "minor_length": minor_length,
            })
        ret.append(pd.DataFrame(measurements))
    return ret

def match_two_channels_greedy(
    df_a,
    df_b,
    tolerance=3,
    area_ratio_warning=3.0,
    channel_a_name="640",
    channel_b_name="488",
    class_shared="shared",
    class_a_only=None,
    class_b_only=None,
):
    """
    Match two-channel bacteria measurements by centroid distance.

    Returns:
        shared: matched object table
        a_only: objects only in channel A
        b_only: objects only in channel B
        df_a_labeled: original df_a with matching labels
        df_b_labeled: original df_b with matching labels
        combined: one-row-per-candidate summary table
    """

    if class_a_only is None:
        class_a_only = f"{channel_a_name}_only"

    if class_b_only is None:
        class_b_only = f"{channel_b_name}_only"

    df_a = df_a.copy().reset_index(drop=True)
    df_b = df_b.copy().reset_index(drop=True)

    required_cols = {"label", "x", "y", "area"}
    missing_a = required_cols - set(df_a.columns)
    missing_b = required_cols - set(df_b.columns)

    if missing_a:
        raise ValueError(f"df_a is missing columns: {missing_a}")

    if missing_b:
        raise ValueError(f"df_b is missing columns: {missing_b}")

    pts_a = df_a[["x", "y"]].to_numpy()
    pts_b = df_b[["x", "y"]].to_numpy()

    tree_b = cKDTree(pts_b)
    nearby = tree_b.query_ball_point(pts_a, r=tolerance)

    pairs = []

    for i_a, b_candidates in enumerate(nearby):
        for i_b in b_candidates:
            d = np.linalg.norm(pts_a[i_a] - pts_b[i_b])
            pairs.append((d, i_a, i_b))

    pairs = sorted(pairs, key=lambda x: x[0])

    used_a = set()
    used_b = set()
    matches = []

    for d, i_a, i_b in pairs:
        if i_a in used_a or i_b in used_b:
            continue

        row_a = df_a.iloc[i_a]
        row_b = df_b.iloc[i_b]

        area_a = row_a["area"]
        area_b = row_b["area"]
        area_ratio = max(area_a, area_b) / (min(area_a, area_b) + 1e-8)

        matches.append({
            f"{channel_a_name}_index": i_a,
            f"{channel_b_name}_index": i_b,
            f"{channel_a_name}_label": row_a["label"],
            f"{channel_b_name}_label": row_b["label"],

            "x": (row_a["x"] + row_b["x"]) / 2,
            "y": (row_a["y"] + row_b["y"]) / 2,

            f"x_{channel_a_name}": row_a["x"],
            f"y_{channel_a_name}": row_a["y"],
            f"x_{channel_b_name}": row_b["x"],
            f"y_{channel_b_name}": row_b["y"],

            "distance": d,

            f"area_{channel_a_name}": area_a,
            f"area_{channel_b_name}": area_b,
            "area_ratio": area_ratio,
            "area_warning": area_ratio > area_ratio_warning,

            f"in_{channel_a_name}": True,
            f"in_{channel_b_name}": True,
            "category": class_shared,
        })

        used_a.add(i_a)
        used_b.add(i_b)

    shared = pd.DataFrame(matches)

    a_only = df_a.loc[
        [i for i in range(len(df_a)) if i not in used_a]
    ].copy()

    b_only = df_b.loc[
        [i for i in range(len(df_b)) if i not in used_b]
    ].copy()

    a_only[f"in_{channel_a_name}"] = True
    a_only[f"in_{channel_b_name}"] = False
    a_only["category"] = class_a_only

    b_only[f"in_{channel_a_name}"] = False
    b_only[f"in_{channel_b_name}"] = True
    b_only["category"] = class_b_only

    # ------------------------------------------------------------
    # 1. Write labels back into the two original dataframes
    # ------------------------------------------------------------

    df_a_labeled = df_a.copy()
    df_b_labeled = df_b.copy()

    for df_labeled in [df_a_labeled, df_b_labeled]:
        df_labeled["match_category"] = "unmatched"
        df_labeled["matched_label"] = np.nan
        df_labeled["match_distance"] = np.nan
        df_labeled["area_ratio"] = np.nan
        df_labeled["area_warning"] = False

    if len(shared) > 0:
        for _, row in shared.iterrows():
            a_idx = int(row[f"{channel_a_name}_index"])
            b_idx = int(row[f"{channel_b_name}_index"])

            df_a_labeled.loc[a_idx, "match_category"] = class_shared
            df_a_labeled.loc[a_idx, "matched_label"] = row[f"{channel_b_name}_label"]
            df_a_labeled.loc[a_idx, "match_distance"] = row["distance"]
            df_a_labeled.loc[a_idx, "area_ratio"] = row["area_ratio"]
            df_a_labeled.loc[a_idx, "area_warning"] = row["area_warning"]

            df_b_labeled.loc[b_idx, "match_category"] = class_shared
            df_b_labeled.loc[b_idx, "matched_label"] = row[f"{channel_a_name}_label"]
            df_b_labeled.loc[b_idx, "match_distance"] = row["distance"]
            df_b_labeled.loc[b_idx, "area_ratio"] = row["area_ratio"]
            df_b_labeled.loc[b_idx, "area_warning"] = row["area_warning"]

    df_a_labeled.loc[list(a_only.index), "match_category"] = class_a_only
    df_b_labeled.loc[list(b_only.index), "match_category"] = class_b_only

    # ------------------------------------------------------------
    # 2. Build one combined summary table
    # ------------------------------------------------------------

    combined_rows = []

    if len(shared) > 0:
        for _, row in shared.iterrows():
            combined_rows.append({
                "object_class": class_shared,
                "x": row["x"],
                "y": row["y"],

                f"{channel_a_name}_label": row[f"{channel_a_name}_label"],
                f"{channel_b_name}_label": row[f"{channel_b_name}_label"],

                f"in_{channel_a_name}": True,
                f"in_{channel_b_name}": True,

                "distance": row["distance"],
                "area_ratio": row["area_ratio"],
                "area_warning": row["area_warning"],

                f"area_{channel_a_name}": row[f"area_{channel_a_name}"],
                f"area_{channel_b_name}": row[f"area_{channel_b_name}"],
            })

    for _, row in a_only.iterrows():
        combined_rows.append({
            "object_class": class_a_only,
            "x": row["x"],
            "y": row["y"],

            f"{channel_a_name}_label": row["label"],
            f"{channel_b_name}_label": np.nan,

            f"in_{channel_a_name}": True,
            f"in_{channel_b_name}": False,

            "distance": np.nan,
            "area_ratio": np.nan,
            "area_warning": False,

            f"area_{channel_a_name}": row["area"],
            f"area_{channel_b_name}": np.nan,
        })

    for _, row in b_only.iterrows():
        combined_rows.append({
            "object_class": class_b_only,
            "x": row["x"],
            "y": row["y"],

            f"{channel_a_name}_label": np.nan,
            f"{channel_b_name}_label": row["label"],

            f"in_{channel_a_name}": False,
            f"in_{channel_b_name}": True,

            "distance": np.nan,
            "area_ratio": np.nan,
            "area_warning": False,

            f"area_{channel_a_name}": np.nan,
            f"area_{channel_b_name}": row["area"],
        })

    combined = pd.DataFrame(combined_rows)

    return shared, a_only, b_only, df_a_labeled, df_b_labeled, combined

def show_label(image, compared):
    shared,A_only,B_only = compared["shared"], compared["only_640"], compared["only_488"]
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray")
    plt.scatter(
        shared["x"],
        shared["y"],
        s=30,
        facecolors="none",
        edgecolors="lime",
        linewidths=1.5,
        label="shared"
    )

    plt.scatter(
        A_only["x"],
        A_only["y"],
        s=20,
        facecolors="none",
        edgecolors="red",
        linewidths=1,
        label="640 only"
    )

    plt.scatter(
        B_only["x"],
        B_only["y"],
        s=20,
        facecolors="none",
        edgecolors="cyan",
        linewidths=1,
        label="488 only"
    )

    plt.legend()
    plt.axis("off")
    plt.show()