import os
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from config import DDI_IMAGES_DIR, DDI_METADATA_PATH, DDI_MASKS_DIR, DDI_PATCHES_DIR

# DDI
IMAGE_COLUMN = "DDI_file"
LABEL_COLUMN = "skin_tone"
PATCH_SIZE = 224
SEED = 42

os.makedirs(DDI_PATCHES_DIR, exist_ok=True)

def process_entry(
    row_data,
    image_column,
    label_column,
    image_folder,
    mask_folder,
    output_folder,
    patch_size
):
    cv.setNumThreads(0)

    filename_raw = row_data[image_column]
    label = row_data[label_column]
    filename = str(filename_raw).strip()

    image_stem = os.path.splitext(filename)[0].split("/")[-1]

    img_path = os.path.join(
        image_folder,
        filename,
    )
    mask_path = os.path.join(mask_folder, image_stem + ".png")

    img = cv.imread(img_path)
    if img is None:
        print(f"Error reading image for {img_path}")
        return []

    height = img.shape[0]
    width = img.shape[1]
    ratio = max(height, width) / 1024

    img = cv.resize(
        img, (int(width / ratio), int(height / ratio)), interpolation=cv.INTER_AREA
    )
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error reading mask for {mask_path}")
        return []

    _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)

    mask = cv.resize(
        mask, (int(width / ratio), int(height / ratio)), interpolation=cv.INTER_NEAREST
    )

    results = []
    patch = extract_healthy_patch(patch_size, img, mask)
    patch_name = f"{image_stem}_patch.png"
    save_path = os.path.join(output_folder, patch_name)
    if patch is not None:
        cv.imwrite(save_path, patch)
        results.append(
            {
                "filename": patch_name,
                "original_image": filename,
                label_column: label,
            }
        )

    if not results:
        patch = extract_less_lesioned_patch(patch_size, img, mask)
        patch_name = f"{image_stem}_patch_fallback.png"
        save_path = os.path.join(output_folder, patch_name)
        cv.imwrite(save_path, patch)
        results.append(
            {
                "filename": patch_name,
                "original_image": filename,
                label_column: label,
            }
        )

    return results

def extract_healthy_skin_patches(
    image_column,
    label_column,
    csv_path,
    image_folder,
    mask_folder,
    output_folder,
    patch_size,
):
    df = pd.read_csv(csv_path)

    rows = df.to_dict('records')

    worker_func = partial(
        process_entry,
        image_column=image_column,
        label_column=label_column,
        image_folder=image_folder,
        mask_folder=mask_folder,
        output_folder=output_folder,
        patch_size=patch_size,
    )

    metadata = []

    num_cores = min(cpu_count(), 12)
    print(f"Processing with {num_cores} cores...")

    with Pool(processes=num_cores) as pool:
        for result in tqdm(pool.imap(worker_func, rows), total=len(rows)):
            if result is not None:
                metadata.extend(result)

    pd.DataFrame(metadata).to_csv(f"ddi_patches_metadata.csv", index=False)


def extract_healthy_patch(patch_size, img, mask):
    kernel = np.ones((patch_size, patch_size), np.uint8)

    valid_locations_map = cv.erode(mask, kernel, anchor=(0, 0), iterations=1)

    valid_points = cv.findNonZero(valid_locations_map)

    if valid_points is None or len(valid_points) == 0:
        return None

    points = valid_points[:, 0, :]

    H, W = img.shape[:2]

    # Center of the lower left quadrant
    target = np.array([(W // 4) - (patch_size // 2), (3 * H // 4) - (patch_size // 2)])

    distances = np.sum((points - target) ** 2, axis=1)

    best_point = None
    for idx in np.argsort(distances):
        x, y = points[idx]
        if 0 <= x <= W - patch_size and 0 <= y <= H - patch_size:
            best_point = (x, y)
            break

    if best_point is None:
        return None

    x, y = best_point
    return img[y : y + patch_size, x : x + patch_size]


def extract_less_lesioned_patch(patch_size, img, mask, stride=5):
    best_score = -1
    best_coordinates = (0, 0)

    H, W = img.shape[:2]
    for y in range(0, H - patch_size, stride):
        for x in range(0, W - patch_size, stride):
            mask_patch = mask[y : y + patch_size, x : x + patch_size]
            score = cv.countNonZero(mask_patch)

            if score > best_score:
                best_score = score
                best_coordinates = (x, y)

    x, y = best_coordinates

    patch = img[y : y + patch_size, x : x + patch_size]

    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv.resize(patch, (patch_size, patch_size), interpolation=cv.INTER_AREA)

    return patch


if __name__ == "__main__":
    extract_healthy_skin_patches(
        image_column=IMAGE_COLUMN,
        label_column=LABEL_COLUMN,
        csv_path=DDI_METADATA_PATH,
        image_folder=DDI_IMAGES_DIR,
        mask_folder=DDI_MASKS_DIR,
        output_folder=DDI_PATCHES_DIR,
        patch_size=PATCH_SIZE,
    )
