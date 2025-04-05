import os
import requests
from tqdm import tqdm
import tarfile

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

if __name__ == "__main__":
    root_dl = "k400"
    tgz_train = f'{root_dl}/tgz/train'
    csv_train = f'{root_dl}/csv/train'
    video_train = f'{root_dl}/video/train'

    
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
