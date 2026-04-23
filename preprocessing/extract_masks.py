import os
import torch
import cv2 as cv
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import Sam3Processor, Sam3Model
from config import DDI_IMAGES_DIR, DDI_METADATA_PATH, DDI_MASKS_DIR

# DDI
IMAGE_COLUMN = 'DDI_file'

BATCH_SIZE = 8
IMG_SIZE = (1024, 1024) 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(DDI_MASKS_DIR, exist_ok=True)

class MaskDataset(Dataset):
    def __init__(self, metadata_path, root_dir, image_column):
        self.root_dir = root_dir
        self.df = pd.read_csv(metadata_path)

        self.files = self.df[image_column].apply(lambda x: os.path.join(self.root_dir, str(x).strip()) if self.root_dir is not None else str(x).strip())
        self.image_column = image_column

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files.iloc[idx]
        filename = self.df[self.image_column].iloc[idx]
        original_image = Image.open(path).convert("RGB")
        original_height, original_width = original_image.size
        resized_image = original_image.resize(IMG_SIZE)

        return filename, np.array(resized_image), original_height, original_width

def segment(model, processor, images, prompt, fallback, threshold=0.5):
    prompts = [prompt] * len(images)

    inputs = processor(
        images=images, 
        text=prompts, 
        return_tensors="pt", 
    ).to(DEVICE)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(outputs, threshold=threshold, target_sizes=[(img.shape[0], img.shape[1]) for img in images])

    batch_masks = []
    for i, res in enumerate(results):
        h, w = images[i].shape[0], images[i].shape[1]
        
        if len(res['masks']) > 0:
            best_idx = np.argmax(res['scores'].cpu().numpy())
            mask = res['masks'][best_idx].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
        else:
            if fallback == 'ones':
                mask = np.ones((h, w), dtype=np.uint8) * 255
            else:
                mask = np.zeros((h, w), dtype=np.uint8)
                
        batch_masks.append(mask)

    return batch_masks

if __name__ == "__main__":
    model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    dataset = MaskDataset(DDI_METADATA_PATH, DDI_IMAGES_DIR, IMAGE_COLUMN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

    for batch in tqdm(dataloader):
        filenames, resized_images, original_heights, original_widths = batch
        skin_masks = segment(model, processor, resized_images, prompt="skin", fallback='ones')

        lesion_masks = segment(model, processor, resized_images, prompt="mole", fallback='zeros')
        
        for i in range(len(filenames)):
            not_lesion = cv.bitwise_not(lesion_masks[i])

            healthy_skin_mask = cv.bitwise_and(skin_masks[i], not_lesion)

            original_size = (int(original_heights[i]), int(original_widths[i]))
            if healthy_skin_mask.shape[:2] != original_size:
                healthy_skin_mask = cv.resize(healthy_skin_mask, original_size, interpolation=cv.INTER_NEAREST)

            cv.imwrite(os.path.join(DDI_MASKS_DIR, filenames[i].split('/')[-1].split('.')[0] + '.png'), healthy_skin_mask)