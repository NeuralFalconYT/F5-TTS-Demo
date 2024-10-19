
# Official Demo From [F5-TTS](https://github.com/SWivid/F5-TTS)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuralFalconYT/F5-TTS-Demo/blob/main/Official_F5_TTS_Demo.ipynb) <br>



https://github.com/user-attachments/assets/476f966d-946e-4cce-b677-36942a926fca



## Local Set Up
#### Step 1: Create and activate a virtual environment

##### Create a virtual environment
```
python -m venv f5-tts-venv
```
##### Activate the virtual environment (Linux/Mac)
```
source f5-tts-venv/bin/activate
```
##### Activate the virtual environment (Windows)
```
f5-tts-venv\Scripts\activate
```
#### Step 2: Clone the repository and navigate to the folder
```
git clone https://github.com/SWivid/F5-TTS.git
```
```
cd F5-TTS
```
#### Step 3: Check CUDA version (if needed)
```
nvcc --version
```
#### Step 4: Install PyTorch and Torchaudio with CUDA [pytorch.org](https://pytorch.org/get-started/locally/)
```
pip install torch==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
#### Step 5: Install other dependencies
```
pip install -r requirements.txt
```
#### Step 6: Run the application
```
python gradio_app.py
```
You can specify the port/host:
```
python gradio_app.py --port 7860 --host 0.0.0.0
```
Or launch a share link:
```
python gradio_app.py --share
```
#### Step 7: Deactivate the virtual environment when done
```
deactivate
```

Load on the GPU if it is currently loading on the CPU
```
state_dict = torch.load(model_path, map_location="cpu")
```
Make sure you are inside  ```F5-TTS``` folder
```
python -c "import requests; exec(requests.get('https://raw.githubusercontent.com/NeuralFalconYT/F5-TTS-Demo/refs/heads/main/download_model.py').text)"

```
replace
```
# vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
```
with
```
vocos_local_path = "./ckpts/vocos-mel-24khz"
vocos = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
state_dict = torch.load(f"{vocos_local_path}/pytorch_model.bin", map_location=device)
vocos.load_state_dict(state_dict)
vocos.eval()
```


# Unofficial F5-TTS-Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuralFalconYT/F5-TTS-Demo/blob/main/F5_TTS_Demo.ipynb) <br>

## 1. Run on Google Colab Cell
![Colab Cell](https://github.com/user-attachments/assets/195f0e51-9bd8-48bb-9378-b422fc8c399d)
## 2. Run Using Gradio App
![Gradio](https://github.com/user-attachments/assets/93d2785f-c134-44d8-89f3-331bb0eb5bc4)
## 3. SRT to Audio Generation (Subtitle Dubbing)
![Subtitle](https://github.com/user-attachments/assets/da76f0d2-cd1a-409a-a6d8-0622986ef264)

# F5-TTS Video Dubbing from Any Languages to English or Chinese (Only Single Speaker Supported)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuralFalconYT/F5-TTS-Demo/blob/main/F5_TTS_Video_Dubbing_Single_Speaker.ipynb) <br>

![dubbing](https://github.com/user-attachments/assets/ddc12f4c-7457-45b1-a6ff-90ca699b7711)


## Credit
[F5-TTS](https://github.com/SWivid/F5-TTS) <br>
```
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```
