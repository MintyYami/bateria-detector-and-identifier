import numpy as np
from skimage import filters, morphology, measure, exposure
from skimage.measure import regionprops
import pandas as pd
from scipy.spatial import cKDTree

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
    min_area: int = 0,
    max_area: int = 200,
    sigma_bg: float = 10,
    threshold_scale: float = 0.1
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

def length_ratio(row):
    return row["major_length"] / (row["minor_length"] + 1e-8)
def getStats(maskCorrects, imgs):
    ret = []
    for i in range(2):
        measurements = []

        regions = regionprops(maskCorrects[i][0], intensity_image=imgs[i])
        for region in regions:
            label_id = region.label
            area = region.area
            y, x = region.centroid
            # y_min, x_min, y_max, x_max = region.bbox
            major_length = region.axis_major_length
            minor_length = region.axis_minor_length
            eccentricity = region.eccentricity
            solidity = region.solidity
            mean_intensity = region.intensity_mean
            max_intensity = region.intensity_max

            measurements.append({
                "label": label_id,
                "area": area,
                "x": x,
                "y": y,
                "major_length": major_length,
                "minor_length": minor_length,
                "eccentricity": eccentricity,
                "solidity": solidity,
                "mean_intensity": mean_intensity,
                "max_intensity": max_intensity,
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

            f"major_length_{channel_a_name}": row_a["major_length"],
            f"major_length_{channel_b_name}": row_b["major_length"],
            f"minor_length_{channel_a_name}": row_a["minor_length"],
            f"minor_length_{channel_b_name}": row_b["minor_length"],

            f"length_ratio_{channel_a_name}": length_ratio(row_a),
            f"length_ratio_{channel_b_name}": length_ratio(row_b),

            f"eccentricity_{channel_a_name}": row_a["eccentricity"],
            f"eccentricity_{channel_b_name}": row_b["eccentricity"],
            f"solidity_{channel_a_name}": row_a["solidity"],
            f"solidity_{channel_b_name}": row_b["solidity"],
            f"mean_intensity_{channel_a_name}": row_a["mean_intensity"],
            f"mean_intensity_{channel_b_name}": row_b["mean_intensity"],
            f"max_intensity_{channel_a_name}": row_a["max_intensity"],
            f"max_intensity_{channel_b_name}": row_b["max_intensity"],
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
    # 2. Build one combined summary table for the raw data
    # ------------------------------------------------------------

    combined_rows = []
    if len(shared) > 0:
        # region showed in both
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

                f"major_length_{channel_a_name}": row[f"major_length_{channel_a_name}"],
                f"major_length_{channel_b_name}": row[f"major_length_{channel_b_name}"],
                f"minor_length_{channel_a_name}": row[f"minor_length_{channel_a_name}"],
                f"minor_length_{channel_b_name}": row[f"minor_length_{channel_b_name}"],
                f"length_ratio_{channel_a_name}": row[f"length_ratio_{channel_a_name}"],
                f"length_ratio_{channel_b_name}": row[f"length_ratio_{channel_b_name}"],
                f"eccentricity_{channel_a_name}": row[f"eccentricity_{channel_a_name}"],
                f"eccentricity_{channel_b_name}": row[f"eccentricity_{channel_b_name}"],
                f"solidity_{channel_a_name}": row[f"solidity_{channel_a_name}"],
                f"solidity_{channel_b_name}": row[f"solidity_{channel_b_name}"],
                f"mean_intensity_{channel_a_name}": row[f"mean_intensity_{channel_a_name}"],
                f"mean_intensity_{channel_b_name}": row[f"mean_intensity_{channel_b_name}"],
                f"max_intensity_{channel_a_name}": row[f"max_intensity_{channel_a_name}"],
                f"max_intensity_{channel_b_name}": row[f"max_intensity_{channel_b_name}"],
            })
    # region showed in 640
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

            f"major_length_{channel_a_name}": row["major_length"],
            f"major_length_{channel_b_name}": np.nan,
            f"minor_length_{channel_a_name}": row["minor_length"],
            f"minor_length_{channel_b_name}": np.nan,
            f"length_ratio_{channel_a_name}": length_ratio(row),
            f"length_ratio_{channel_b_name}": np.nan,
            f"eccentricity_{channel_a_name}": row["eccentricity"],
            f"eccentricity_{channel_b_name}": np.nan,
            f"solidity_{channel_a_name}": row["solidity"],
            f"solidity_{channel_b_name}": np.nan,
            f"mean_intensity_{channel_a_name}": row["mean_intensity"],
            f"mean_intensity_{channel_b_name}": np.nan,
            f"max_intensity_{channel_a_name}": row["max_intensity"],
            f"max_intensity_{channel_b_name}": np.nan,
        })
    # region showed in 488
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

            f"major_length_{channel_a_name}": np.nan,
            f"major_length_{channel_b_name}": row["major_length"],
            f"minor_length_{channel_a_name}": np.nan,
            f"minor_length_{channel_b_name}": row["minor_length"],
            f"length_ratio_{channel_a_name}": np.nan,
            f"length_ratio_{channel_b_name}": length_ratio(row),
            f"eccentricity_{channel_a_name}": np.nan,
            f"eccentricity_{channel_b_name}": row["eccentricity"],
            f"solidity_{channel_a_name}": np.nan,
            f"solidity_{channel_b_name}": row["solidity"],
            f"mean_intensity_{channel_a_name}": np.nan,
            f"mean_intensity_{channel_b_name}": row["mean_intensity"],
            f"max_intensity_{channel_a_name}": np.nan,
            f"max_intensity_{channel_b_name}": row["max_intensity"],
        })

    combined = pd.DataFrame(combined_rows)

    return shared, a_only, b_only, df_a_labeled, df_b_labeled, combined

# get clean trainable data (no local info)
def extract_unified_features(combined, channel_a_name="640", channel_b_name="488"):
    rows = []

    for _, row in combined.iterrows():
        out = {
            f"in_{channel_a_name}": row.get(f"in_{channel_a_name}", False),
            f"in_{channel_b_name}": row.get(f"in_{channel_b_name}", False),
        }

        # helper: pick value from A if exists, else B
        def pick(field):
            a_val = row.get(f"{field}_{channel_a_name}", np.nan)
            b_val = row.get(f"{field}_{channel_b_name}", np.nan)

            if not pd.isna(b_val) and not pd.isna(a_val):
                return np.average([a_val, b_val])
            if not pd.isna(b_val):
                return b_val
            if not pd.isna(a_val):
                return a_val
            return np.nan

        # unified fields
        for field in [
            "major_length",
            "minor_length",
            "length_ratio",
            "eccentricity",
            "solidity",
            "mean_intensity",
            "max_intensity",
        ]:
            out[field] = pick(field)

        rows.append(out)

    return pd.DataFrame(rows)