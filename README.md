
# Official Demo From [F5-TTS](https://github.com/SWivid/F5-TTS)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuralFalconYT/F5-TTS-Demo/blob/main/Official_F5_TTS_Demo.ipynb) <br>



https://github.com/user-attachments/assets/476f966d-946e-4cce-b677-36942a926fca


# Unofficial F5-TTS-Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuralFalconYT/F5-TTS-Demo/blob/main/F5_TTS_Demo.ipynb) <br>

## 1. Run on Google Colab Cell
![Colab Cell](https://github.com/user-attachments/assets/195f0e51-9bd8-48bb-9378-b422fc8c399d)
## 2. Run Using Gradio App
![Gradio](https://github.com/user-attachments/assets/93d2785f-c134-44d8-89f3-331bb0eb5bc4)
## 3. SRT to Audio Generation (Subtitle Dubbing)
![Subtitle](https://github.com/user-attachments/assets/da76f0d2-cd1a-409a-a6d8-0622986ef264)
## Local Set Up
```
git clone https://github.com/SWivid/F5-TTS.git
```
```
cd F5-TTS
```
[Skip this if you already have Torch installed] Install torch with your CUDA version, e.g. :
```
pip install torch==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
```
pip install -r requirements.txt
```
```
python gradio_app.py
```
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
