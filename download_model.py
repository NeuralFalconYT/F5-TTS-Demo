import os
import urllib.request
import urllib.error
from tqdm import tqdm
import shutil
def conditional_download(url, download_file_path, redownload=False):
    print(f"Downloading {os.path.basename(download_file_path)}")
    base_path = os.path.dirname(download_file_path)

    # Create the directory if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Skip download if the file exists and redownload is False
    if os.path.exists(download_file_path) and not redownload:
        print(f"File {download_file_path} already exists. Skipping download.")
        return

    # If redownload is True, remove the existing file
    if os.path.exists(download_file_path) and redownload:
        os.remove(download_file_path)

    # Try opening the URL and get the total file size
    try:
        request = urllib.request.urlopen(url)
        total = int(request.headers.get('Content-Length', 0))
    except urllib.error.URLError as e:
        print(f"Error: Unable to open the URL - {url}")
        print(f"Reason: {e.reason}")
        return

    # Start downloading with a progress bar
    with tqdm(total=total, desc=f"Downloading {os.path.basename(download_file_path)}", unit='B', unit_scale=True, unit_divisor=1024) as progress:
        try:
            urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))
        except urllib.error.URLError as e:
            print(f"Error: Failed to download the file from the URL - {url}")
            print(f"Reason: {e.reason}")
            return

    print(f"Download successful! Saved at: {download_file_path}")

 
# Step 3: Download the models
def download_models(base_path, redownload=False):
    print("Starting model downloads...")
    model_urls = [
        ("https://huggingface.co/SWivid/E2-TTS/resolve/main/E2TTS_Base/model_1200000.pt", f"{base_path}/ckpts/E2TTS_Base/model_1200000.pt"),
        # ("https://huggingface.co/SWivid/E2-TTS/resolve/main/E2TTS_Base/model_1200000.safetensors", f"{base_path}/ckpts/E2TTS_Base/model_1200000.safetensors"),
        ("https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_Base/model_1200000.pt", f"{base_path}/ckpts/F5TTS_Base/model_1200000.pt"),
        # ("https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_Base/model_1200000.safetensors", f"{base_path}/ckpts/F5TTS_Base/model_1200000.safetensors"),
        ("https://huggingface.co/charactr/vocos-mel-24khz/resolve/main/pytorch_model.bin", f"{base_path}/ckpts/vocos-mel-24khz/pytorch_model.bin"),
        ("https://huggingface.co/charactr/vocos-mel-24khz/resolve/main/config.yaml", f"{base_path}/ckpts/vocos-mel-24khz/config.yaml")
    ]

    for url, path in model_urls:
        conditional_download(url, path, redownload=redownload)

    print("All models downloaded successfully.")


base_path = "."
# base_path = "/content"     
download_models(base_path, redownload=True)
