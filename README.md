
# Official Demo From [F5-TTS](https://github.com/SWivid/F5-TTS)
Official one:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuralFalconYT/F5-TTS-Demo/blob/main/F5_TTS_Latest.ipynb) <br>

For Hindi:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuralFalconYT/F5-TTS-Demo/blob/main/F5_TTS_Hindi_Small.ipynb) <br>



https://github.com/user-attachments/assets/476f966d-946e-4cce-b677-36942a926fca



## Local Set Up
#### Step 1: Create and activate a virtual environment
## Using virtualenv
##### Create a virtual environment
```
python -m venv f5-tts
```
##### Activate the virtual environment (Linux/Mac)
```
source f5-tts/bin/activate
```
##### Activate the virtual environment (Windows)
```
f5-tts\Scripts\activate
```
## Using conda
##### Create a python 3.10 conda env (you could also use virtualenv)
```
conda create -n f5-tts python=3.10
conda activate f5-tts
```
#### Step 2: Check CUDA version (if needed)
```
nvcc --version
```
#### Step 3: Install PyTorch and Torchaudio with CUDA [pytorch.org](https://pytorch.org/get-started/locally/)
```
pip install torch==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
#### Step 4: Install other dependencies
```
pip install git+https://github.com/SWivid/F5-TTS.git
```
#### Step 6: Run the application

##### Launch a Gradio app (web interface)
```
f5-tts_infer-gradio
```
##### Specify the port/host
```
f5-tts_infer-gradio --port 7860 --host 0.0.0.0
```
##### Launch a share link
```
f5-tts_infer-gradio --share
```
#### Step 7: Deactivate the virtual environment when done
```
deactivate
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
Follow this Colab Notebook to run it on local device too
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
