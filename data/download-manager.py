import os
import requests
from tqdm import tqdm
import tarfile

import cv2
import random
import pandas as pd
from pathlib import Path

def download_file(url, output_path, desc=None):
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    with open(output_path, 'wb') as file, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def untar_file(tar_path, extract_path):
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_path)

def extract_frames(csv_path, video_dir, output_dir):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    for _, row in df.iterrows():
        label = row['label'].replace(' ', '_')
        youtube_id = row['youtube_id']
        time_start = int(row['time_start'])
        time_end = int(row['time_end'])
        video_file = f"{youtube_id}_{time_start:06d}_{time_end:06d}.mp4"
        video_path = Path(video_dir) / video_file
        if not video_path.exists():
            continue
        label_dir = Path(output_dir) / label
        os.makedirs(label_dir, exist_ok=True)
        video = cv2.VideoCapture(str(video_path))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            continue
        frame_indices = random.sample(range(total_frames), min(3, total_frames))
        for i, idx in enumerate(frame_indices):
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = video.read()
            if success:
                output_path = label_dir / f"{video_path.stem}_frame{i+1}.jpg"
                cv2.imwrite(str(output_path), frame)
        video.release()

if __name__ == "__main__":
    root_dl = "k400"
    tgz_train = f'{root_dl}/tgz/train'
    csv_train = f'{root_dl}/csv/train'
    video_train = f'{root_dl}/video/train'
    frame_train = f'{root_dl}/frame/train'

    
    os.makedirs(root_dl, exist_ok=True)
    os.makedirs(tgz_train, exist_ok=True)
    os.makedirs(csv_train, exist_ok=True)
    os.makedirs(video_train, exist_ok=True)
    
    download_list_url = "https://s3.amazonaws.com/kinetics/400/train/k400_train_path.txt"
    
    train_download_list = requests.get(download_list_url).text.splitlines()
    first_url = train_download_list[0]

    output_path = f"{tgz_train}/{os.path.basename(first_url)}"
    download_file(first_url, output_path, desc="Downloading first train item")

    csv_file_url = "https://s3.amazonaws.com/kinetics/400/annotations/train.csv"

    csv_output = f'{csv_train}/{os.path.basename(csv_file_url)}'
    download_file(csv_file_url, csv_output, desc="Downloading train csv")

    untar_file(output_path,video_train)


    extract_frames('./k400/csv/train/train.csv',video_train,frame_train)
