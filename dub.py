
base_path = "."
# base_path = "/content"

#@title import Model
# import locale
# locale.getpreferredencoding = lambda: "UTF-8"
import os
install_path=f"{base_path}/"
os.chdir(install_path)
import os
import re
import torch
import torchaudio
from einops import rearrange
from vocos import Vocos
from model import CFM, UNetT, DiT
from model.utils import (
    load_checkpoint,
    get_tokenizer,
    convert_char_to_pinyin,
    save_spectrogram,
)
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import librosa
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
import re
import uuid
from tqdm.notebook import tqdm
import shutil
from IPython.display import clear_output
import gc
import time
import subprocess
from IPython.display import Audio

import torch

def get_max_gpu_memory():
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        max_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        max_memory_gb = max_memory / (1024 ** 3)  # Convert bytes to GB
        return max_memory_gb -1
    else:
        return 0  # No GPU available



def is_gpu_memory_over_limit(limit_gb=14.5):
    limit_gb=get_max_gpu_memory()
    # Run nvidia-smi and capture the output
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                            stdout=subprocess.PIPE, text=True)

    # Split the result into lines (for each GPU if there are multiple)
    memory_used_mb_list = result.stdout.strip().splitlines()

    # Convert memory used from MB to GB and check each GPU's memory usage
    for i, memory_used_mb in enumerate(memory_used_mb_list):
        memory_used_gb = int(memory_used_mb) / 1024.0
        # print(f"GPU {i}: Current memory allocated: {memory_used_gb:.2f} GB")
        if memory_used_gb > limit_gb:
            # print(f"GPU {i} memory usage exceeds {limit_gb} GB.")
            return True

    # print("GPU memory usage is within safe limits.")
    return False



# Load Whisper model
def load_whisper():
    global whisper_pipe,whisper_model
    try:
        if whisper_pipe is not None:
          del whisper_pipe
          whisper_pipe=None
        if whisper_model is not None:
          del whisper_model
          whisper_model=None
        gc.collect()
        torch.cuda.empty_cache()
        # print("Free GPU memeory")
        time.sleep(2)
    except:
      pass
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    whisper_model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return whisper_pipe,whisper_model

# Initialize vocoder and model function
def initialize_vocoder_and_model(
    exp_name="F5TTS_Base",
    ckpt_step=1200000,
    device="cuda",
    target_sample_rate=24000,
    n_mel_channels=100,
    hop_length=256,
    dataset_name="Emilia_ZH_EN",
    tokenizer="pinyin",
    ode_method='euler',
    use_ema=True
):
    global vocos,model
    try:
      if vocos is not None:
        del vocos
        vocos=None
      if model is not None:
        del model
        model=None
      gc.collect()
      torch.cuda.empty_cache()
      # print("Free GPU memeory")
      time.sleep(2)
    except:
      pass
    # Set model configuration based on experiment name
    if exp_name == "F5TTS_Base":
        model_cls = DiT
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    elif exp_name == "E2TTS_Base":
        model_cls = UNetT
        model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)

    # Initialize vocoder
    vocos_local_path = "./ckpts/vocos-mel-24khz"
    vocos = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
    state_dict = torch.load(f"{vocos_local_path}/pytorch_model.bin", map_location=device)
    vocos.load_state_dict(state_dict)
    vocos.eval()

    # Initialize tokenizer
    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

    # Initialize model
    model = CFM(
        transformer=model_cls(
            **model_cfg,
            text_num_embeds=vocab_size,
            mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    # Load the model checkpoint
    ckpt_path = f"ckpts/{exp_name}/model_{ckpt_step}.pt"
    model = load_checkpoint(model, ckpt_path, device, use_ema=use_ema)

    return vocos, model

def merge_audio(audio_list, save_path):
    # Initialize an empty audio segment
    merged_audio = AudioSegment.empty()

    # Loop through the list of audio files
    for audio_file in audio_list:
        # Load each audio file
        audio_segment = AudioSegment.from_wav(audio_file)
        # Append to the merged audio segment
        merged_audio += audio_segment

    # Export the merged audio to the specified save path
    merged_audio.export(save_path, format="wav")

def chunks_sentences(paragraph, join_limit=2):
    sentences = sent_tokenize(paragraph)
    # Initialize an empty list to store the new sentences
    new_sentences = []

    # Iterate through the list of sentences in steps of 'join_limit'
    for i in range(0, len(sentences), join_limit):
        # Join the sentences with a space between them
        new_sentence = ' '.join(sentences[i:i + join_limit])
        new_sentences.append(new_sentence)
    return new_sentences


def clean_file_name(file_path):
    # Get the base file name and extension
    file_name = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(file_name)

    # Replace non-alphanumeric characters with an underscore
    cleaned = re.sub(r'[^a-zA-Z\d]+', '_', file_name)

    # Remove any multiple underscores
    clean_file_name = re.sub(r'_+', '_', cleaned).strip('_')

    # Generate a random UUID for uniqueness
    random_uuid = uuid.uuid4().hex[:6]

    # Combine cleaned file name with the original extension
    clean_file_path = os.path.join(os.path.dirname(file_path), clean_file_name + f"_{random_uuid}" + file_extension)

    return clean_file_path


def tts_file_name(text):
    if text.endswith("."):
        text = text[:-1]
    text = text.lower()
    text = text.strip()
    text = text.replace(" ","_")
    truncated_text = text[:25] if len(text) > 25 else text if len(text) > 0 else "empty"
    random_string = uuid.uuid4().hex[:8].upper()
    file_name = f"{base_path}/f5_Voice/{truncated_text}_{random_string}.wav"
    file_name=clean_file_name(file_name)
    return file_name



import os
from pydub import AudioSegment


def is_audio_duration_greater_than_30s(audio_path,max_duration=30):
    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return False

    # Get the duration in seconds
    duration = len(audio) / 1000  # pydub works in milliseconds

    # Check if the duration is greater than 30 seconds
    return duration > max_duration

def trim_audio(input_audio_path,max_duration=30):
    # Create output folder if it doesn't exist
    global base_path
    output_folder=f"{base_path}/trim_audio"
    os.makedirs(output_folder, exist_ok=True)

    # Load the audio file
    audio = AudioSegment.from_file(input_audio_path)

    # Check the duration of the audio in seconds
    duration = len(audio) / 1000  # pydub works in milliseconds

    # Trim the audio if it exceeds the max_duration
    if duration > max_duration:
        trimmed_audio = audio[:max_duration * 1000]  # Trim to max_duration
    else:
        trimmed_audio = audio

    # Generate a new file name
    base_name = os.path.splitext(os.path.basename(input_audio_path))[0]
    output_file = f"{output_folder}/{base_name}_trimmed.wav"
    # output_file=clean_file_name(output_file)
    trimmed_audio.export(output_file, format="wav")

    return output_file


def process_audio(reference_audio, max_duration=15):
    global old_trim_audio,base_path

    # Check if the audio duration exceeds max_duration
    if is_audio_duration_greater_than_30s(reference_audio, max_duration):
        f_base_name = os.path.basename(reference_audio)
        f_name, f_extension = os.path.splitext(f_base_name)
        trimmed_audio_path = f"{base_path}/trim_audio/{f_name}_trimmed.wav"

        # Check if we've already trimmed this audio
        if old_trim_audio == trimmed_audio_path:
            reference_audio = trimmed_audio_path  # Use existing trimmed audio
            # print("skipping because same trim audio")
        else:
            reference_audio = trim_audio(reference_audio, max_duration)  # Trim the audio
            old_trim_audio = reference_audio  # Update the old trimmed audio path

    return reference_audio


# Voice cloning function

def voice_clone(reference_audio, text, output_dir="", target_sample_rate=24000,remove_silence = False,fix_duration=None,chunks=0,exp_name="F5TTS_Base",progress_bar=True):
    global device,old_audio_path,old_ref_text,old_exp_name
    global whisper_pipe,whisper_model,vocos,model
    global seed
    reference_audio=process_audio(reference_audio, max_duration=15)

    if old_exp_name==exp_name:
      pass
    else:
      vocos, model = initialize_vocoder_and_model(device=device,exp_name=exp_name)
      old_exp_name=exp_name
    if is_gpu_memory_over_limit():
      whisper_pipe,whisper_model = load_whisper()
      device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
      vocos, model = initialize_vocoder_and_model(device=device,exp_name=exp_name)
    # clear_output()
    # seed = None
    final_audio_path= tts_file_name(text)
    output_dir=f"{base_path}/f5_Voice/temp"
    nfe_step = 32
    cfg_strength = 2.
    ode_method = 'euler'  # euler | midpoint
    speed = 1.
    # target_sample_rate = 24000
    # fix_duration=27
    target_rms = 0.1
    hop_length = 256  # Ensure hop_length is define
    tokenizer="pinyin"
    sway_sampling_coef = -1.

    fix_duration = fix_duration  # None (will linear estimate. if code-switched, consider fix) | float (total in seconds, include ref audio)

    # Get the reference audio text
    if old_audio_path==reference_audio:
      ref_text=old_ref_text
      # print("skipping because same audio file")
    else:
      ref_text = whisper_pipe(reference_audio)['text'].strip()
      old_audio_path=reference_audio
      old_ref_text=ref_text



    # Ensure output directory exists
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Load audio
    audio, sr = torchaudio.load(reference_audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Normalize audio
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms

    # Resample audio if necessary
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    if chunks==0:
      prompts=[text]
    else:
      prompts=chunks_sentences(text, join_limit=chunks)
    audio_list=[]

    number_of_prompts=len(prompts)
    iterable = tqdm(enumerate(prompts), total=len(prompts), desc="Processing Prompts") if progress_bar else enumerate(prompts)
    # for i,text in enumerate(prompts):
    # for i, text in tqdm(enumerate(prompts), total=len(prompts), desc="Processing Prompts"):
    for i, text in iterable:
      gen_text = text.strip()
      # Prepare text
      text_list = [ref_text + gen_text]
      if tokenizer == "pinyin":
          final_text_list = convert_char_to_pinyin(text_list)
      else:
          final_text_list = [text_list]
      # print(f"text  : {text_list}")
      # print(f"pinyin: {final_text_list}")

      # Calculate duration
      ref_audio_len = audio.shape[-1] // hop_length
      if fix_duration is not None:
          if number_of_prompts==1:
            duration = int(fix_duration * target_sample_rate / hop_length)
      else:
          zh_pause_punc = r"。，、；：？！"
          ref_text_len = len(ref_text) + len(re.findall(zh_pause_punc, ref_text))
          gen_text_len = len(gen_text) + len(re.findall(zh_pause_punc, gen_text))
          duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

      # Inference
      with torch.inference_mode():
          generated, _ = model.sample(
              cond=audio,
              text=final_text_list,
              duration=duration,
              steps=nfe_step,
              cfg_strength=cfg_strength,
              sway_sampling_coef=sway_sampling_coef,
              seed=seed,
          )

      generated = generated[:, ref_audio_len:, :]
      generated_mel_spec = rearrange(generated, '1 n d -> 1 d n')
      generated_wave = vocos.decode(generated_mel_spec.cpu())
      if rms < target_rms:
          generated_wave = generated_wave * rms / target_rms

      if isinstance(generated_wave, torch.Tensor):
          generated_wave = generated_wave.squeeze().cpu().numpy()

      # Remove silence

      if remove_silence:
          # Detect non-silent intervals
          non_silent_intervals = librosa.effects.split(generated_wave, top_db=30)

          # Concatenate non-silent parts
          non_silent_wave = np.array([])
          for interval in non_silent_intervals:
              start, end = interval
              non_silent_wave = np.concatenate([non_silent_wave, generated_wave[start:end]])

          # Replace generated_wave with the non-silent version
          generated_wave = non_silent_wave
          generated_wave_tensor = torch.tensor(generated_wave).unsqueeze(0)
      else:
          generated_wave_tensor = torch.tensor(generated_wave).unsqueeze(0)

      # Ensure that generated_wave_tensor is 2D (batch size, channels)
      if len(generated_wave_tensor.shape) == 1:
          generated_wave_tensor = generated_wave_tensor.unsqueeze(0)

      # Save the generated audio
      save_audio_path = f"{output_dir}/{i}.wav"
      torchaudio.save(save_audio_path, generated_wave_tensor, target_sample_rate)
      audio_list.append(save_audio_path)
    if len(audio_list)==1:
      shutil.copy(audio_list[-1],final_audio_path)
    elif len(audio_list)>1:
      merge_audio(audio_list, final_audio_path)
    else:
      final_audio_path=None
    return final_audio_path

whisper_pipe = None
whisper_model=None
vocos = None
model = None
seed = None
whisper_pipe,whisper_model = load_whisper()
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
vocos, model = initialize_vocoder_and_model(device=device,exp_name="F5TTS_Base")
old_audio_path=""
old_ref_text=""
old_trim_audio=""
old_exp_name="F5TTS_Base"
os.makedirs(f"{base_path}/f5_Voice", exist_ok=True)
clear_output()
print("Model Import Complete")

# #@title Generate TTS
# Reference_Audio_Path= '/content/F5-TTS/tests/ref_audio/test_en_1_ref_short.wav'  # @param {type: "string"}
# TTS_Text = 'A cat named Luna loved watching the rain from her window. One day, a small bird landed on the sill, chirping happily despite the storm. Luna, curious but gentle, tapped the glass with her paw. The bird fluttered away, leaving Luna to wonder what it was like to fly. As the rain continued, she dreamed of soaring through the clouds.'  # @param {type: "string"}
# Remove_Silence = True  # @param {type: "boolean"}
# # Split_Sentences = 0  # @param {type: "number"}
# if len(TTS_Text)<=135:
#   Split_Sentences=0
# if len(TTS_Text)>135:
#   Split_Sentences=2
# Choose_Model = "F5TTS_Base" # @param ['F5TTS_Base', 'E2TTS_Base']
# seed = None
# cloned_voice_path=voice_clone(Reference_Audio_Path, TTS_Text,remove_silence = Remove_Silence,chunks=Split_Sentences,exp_name=Choose_Model)
# clear_output()
# print(f"TTS Save at {cloned_voice_path}")
# Audio(cloned_voice_path)

#@title Utils
language_dict = {
    "Akan": {"lang_code": "aka", "meta_code": "aka_Latn"},
    "Albanian": {"lang_code": "sq", "meta_code": "als_Latn"},
    "Amharic": {"lang_code": "am", "meta_code": "amh_Ethi"},
    "Arabic": {"lang_code": "ar", "meta_code": "arb_Arab"},
    "Armenian": {"lang_code": "hy", "meta_code": "hye_Armn"},
    "Assamese": {"lang_code": "as", "meta_code": "asm_Beng"},
    "Azerbaijani": {"lang_code": "az", "meta_code": "azj_Latn"},
    "Basque": {"lang_code": "eu", "meta_code": "eus_Latn"},
    "Bashkir": {"lang_code": "ba", "meta_code": "bak_Cyrl"},
    "Bengali": {"lang_code": "bn", "meta_code": "ben_Beng"},
    "Bosnian": {"lang_code": "bs", "meta_code": "bos_Latn"},
    "Bulgarian": {"lang_code": "bg", "meta_code": "bul_Cyrl"},
    "Burmese": {"lang_code": "my", "meta_code": "mya_Mymr"},
    "Catalan": {"lang_code": "ca", "meta_code": "cat_Latn"},
    "Chinese": {"lang_code": "zh", "meta_code": "zh_Hans"},
    "Croatian": {"lang_code": "hr", "meta_code": "hrv_Latn"},
    "Czech": {"lang_code": "cs", "meta_code": "ces_Latn"},
    "Danish": {"lang_code": "da", "meta_code": "dan_Latn"},
    "Dutch": {"lang_code": "nl", "meta_code": "nld_Latn"},
    "English": {"lang_code": "en", "meta_code": "eng_Latn"},
    "Estonian": {"lang_code": "et", "meta_code": "est_Latn"},
    "Faroese": {"lang_code": "fo", "meta_code": "fao_Latn"},
    "Finnish": {"lang_code": "fi", "meta_code": "fin_Latn"},
    "French": {"lang_code": "fr", "meta_code": "fra_Latn"},
    "Galician": {"lang_code": "gl", "meta_code": "glg_Latn"},
    "Georgian": {"lang_code": "ka", "meta_code": "kat_Geor"},
    "German": {"lang_code": "de", "meta_code": "deu_Latn"},
    "Greek": {"lang_code": "el", "meta_code": "ell_Grek"},
    "Gujarati": {"lang_code": "gu", "meta_code": "guj_Gujr"},
    "Haitian Creole": {"lang_code": "ht", "meta_code": "hat_Latn"},
    "Hausa": {"lang_code": "ha", "meta_code": "hau_Latn"},
    "Hebrew": {"lang_code": "he", "meta_code": "heb_Hebr"},
    "Hindi": {"lang_code": "hi", "meta_code": "hin_Deva"},
    "Hungarian": {"lang_code": "hu", "meta_code": "hun_Latn"},
    "Icelandic": {"lang_code": "is", "meta_code": "isl_Latn"},
    "Indonesian": {"lang_code": "id", "meta_code": "ind_Latn"},
    "Italian": {"lang_code": "it", "meta_code": "ita_Latn"},
    "Japanese": {"lang_code": "ja", "meta_code": "jpn_Jpan"},
    "Kannada": {"lang_code": "kn", "meta_code": "kan_Knda"},
    "Kazakh": {"lang_code": "kk", "meta_code": "kaz_Cyrl"},
    "Korean": {"lang_code": "ko", "meta_code": "kor_Hang"},
    "Kurdish": {"lang_code": "ckb", "meta_code": "ckb_Arab"},
    "Kyrgyz": {"lang_code": "ky", "meta_code": "kir_Cyrl"},
    "Lao": {"lang_code": "lo", "meta_code": "lao_Laoo"},
    "Lithuanian": {"lang_code": "lt", "meta_code": "lit_Latn"},
    "Luxembourgish": {"lang_code": "lb", "meta_code": "ltz_Latn"},
    "Macedonian": {"lang_code": "mk", "meta_code": "mkd_Cyrl"},
    "Malay": {"lang_code": "ms", "meta_code": "ms_Latn"},
    "Malayalam": {"lang_code": "ml", "meta_code": "mal_Mlym"},
    "Maltese": {"lang_code": "mt", "meta_code": "mlt_Latn"},
    "Maori": {"lang_code": "mi", "meta_code": "mri_Latn"},
    "Marathi": {"lang_code": "mr", "meta_code": "mar_Deva"},
    "Mongolian": {"lang_code": "mn", "meta_code": "khk_Cyrl"},
    "Nepali": {"lang_code": "ne", "meta_code": "npi_Deva"},
    "Norwegian": {"lang_code": "no", "meta_code": "nob_Latn"},
    "Norwegian Nynorsk": {"lang_code": "nn", "meta_code": "nno_Latn"},
    "Pashto": {"lang_code": "ps", "meta_code": "pbt_Arab"},
    "Persian": {"lang_code": "fa", "meta_code": "pes_Arab"},
    "Polish": {"lang_code": "pl", "meta_code": "pol_Latn"},
    "Portuguese": {"lang_code": "pt", "meta_code": "por_Latn"},
    "Punjabi": {"lang_code": "pa", "meta_code": "pan_Guru"},
    "Romanian": {"lang_code": "ro", "meta_code": "ron_Latn"},
    "Russian": {"lang_code": "ru", "meta_code": "rus_Cyrl"},
    "Serbian": {"lang_code": "sr", "meta_code": "srp_Cyrl"},
    "Sinhala": {"lang_code": "si", "meta_code": "sin_Sinh"},
    "Slovak": {"lang_code": "sk", "meta_code": "slk_Latn"},
    "Slovenian": {"lang_code": "sl", "meta_code": "slv_Latn"},
    "Somali": {"lang_code": "so", "meta_code": "som_Latn"},
    "Spanish": {"lang_code": "es", "meta_code": "spa_Latn"},
    "Sundanese": {"lang_code": "su", "meta_code": "sun_Latn"},
    "Swahili": {"lang_code": "sw", "meta_code": "swa_Latn"},
    "Swedish": {"lang_code": "sv", "meta_code": "swe_Latn"},
    "Tamil": {"lang_code": "ta", "meta_code": "tam_Taml"},
    "Telugu": {"lang_code": "te", "meta_code": "tel_Telu"},
    "Thai": {"lang_code": "th", "meta_code": "tha_Latn"},
    "Turkish": {"lang_code": "tr", "meta_code": "tur_Latn"},
    "Ukrainian": {"lang_code": "uk", "meta_code": "ukr_Cyrl"},
    "Urdu": {"lang_code": "ur", "meta_code": "urd_Arab"},
    "Uzbek": {"lang_code": "uz", "meta_code": "uzb_Latn"},
    "Vietnamese": {"lang_code": "vi", "meta_code": "vie_Latn"},
    "Welsh": {"lang_code": "cy", "meta_code": "cym_Latn"},
    "Yiddish": {"lang_code": "yi", "meta_code": "yi_Hebr"},
    "Yoruba": {"lang_code": "yo", "meta_code": "yo_Latn"},
    "Zulu": {"lang_code": "zu", "meta_code": "zul_Latn"},
}
available_language=['English','Hindi','Bengali','Akan', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Assamese', 'Azerbaijani', 'Basque', 'Bashkir', 'Bengali', 'Bosnian', 'Bulgarian', 'Burmese', 'Catalan', 'Chinese', 'Croatian', 'Czech', 'Danish', 'Dutch', 'English', 'Estonian', 'Faroese', 'Finnish', 'French', 'Galician', 'Georgian', 'German', 'Greek', 'Gujarati', 'Haitian Creole', 'Hausa', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Indonesian', 'Italian', 'Japanese', 'Kannada', 'Kazakh', 'Korean', 'Kurdish', 'Kyrgyz', 'Lao', 'Lithuanian', 'Luxembourgish', 'Macedonian', 'Malay', 'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Mongolian', 'Nepali', 'Norwegian', 'Norwegian Nynorsk', 'Pashto', 'Persian', 'Polish', 'Portuguese', 'Punjabi', 'Romanian', 'Russian', 'Serbian', 'Sinhala', 'Slovak', 'Slovenian', 'Somali', 'Spanish', 'Sundanese', 'Swahili', 'Swedish', 'Tamil', 'Telugu', 'Thai', 'Turkish', 'Ukrainian', 'Urdu', 'Uzbek', 'Vietnamese', 'Welsh', 'Yiddish', 'Yoruba', 'Zulu']
import math
import torch
import gc
import time
import subprocess
from faster_whisper import WhisperModel
import os
import mimetypes
import shutil
import re
import uuid
from pydub import AudioSegment
from transformers import pipeline


def get_language_name(lang_code):
    global language_dict
    # Iterate through the language dictionary
    for language, details in language_dict.items():
        # Check if the language code matches
        if details["lang_code"] == lang_code:
            return language  # Return the language name
    return None

def clean_file_name(file_path):
    # Get the base file name and extension
    file_name = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(file_name)

    # Replace non-alphanumeric characters with an underscore
    cleaned = re.sub(r'[^a-zA-Z\d]+', '_', file_name)

    # Remove any multiple underscores
    clean_file_name = re.sub(r'_+', '_', cleaned).strip('_')

    # Generate a random UUID for uniqueness
    random_uuid = uuid.uuid4().hex[:6]

    # Combine cleaned file name with the original extension
    clean_file_path = os.path.join(os.path.dirname(file_path), clean_file_name + f"_{random_uuid}" + file_extension)

    return clean_file_path

def get_audio_file(uploaded_file):
    global base_path
    # ,device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Detect the file type (audio/video)
    mime_type, _ = mimetypes.guess_type(uploaded_file)
    # Create the folder path to store audio files
    audio_folder = f"{base_path}/subtitle_audio"
    os.makedirs(audio_folder, exist_ok=True)
    # Initialize variable for the audio file path
    audio_file_path = ""
    if mime_type and mime_type.startswith('audio'):
        # If it's an audio file, save it as is
        audio_file_path = os.path.join(audio_folder, os.path.basename(uploaded_file))
        audio_file_path=clean_file_name(audio_file_path)
        shutil.copy(uploaded_file, audio_file_path)  # Move file to audio folder

    elif mime_type and mime_type.startswith('video'):
        # If it's a video file, extract the audio
        audio_file_name = os.path.splitext(os.path.basename(uploaded_file))[0] + ".mp3"
        audio_file_path = os.path.join(audio_folder, audio_file_name)
        audio_file_path=clean_file_name(audio_file_path)

        # Extract the file extension from the uploaded file
        file_extension = os.path.splitext(uploaded_file)[1]  # Includes the dot, e.g., '.mp4'

        # Generate a random UUID and create a new file name with the same extension
        random_uuid = uuid.uuid4().hex[:6]
        new_file_name = random_uuid + file_extension

        # Set the new file path in the subtitle_audio folder
        new_file_path = os.path.join(audio_folder, new_file_name)

        # Copy the original video file to the new location with the new name
        shutil.copy(uploaded_file, new_file_path)
        if device=="cuda":
          command = f"ffmpeg -hwaccel cuda -i {new_file_path} {audio_file_path} -y"
        else:
          command = f"ffmpeg -i {new_file_path} {audio_file_path} -y"

        subprocess.run(command, shell=True)
        if os.path.exists(new_file_path):
          os.remove(new_file_path)
    # Return the saved audio file path
    audio = AudioSegment.from_file(audio_file_path)
    # Get the duration in seconds
    duration_seconds = len(audio) / 1000.0  # pydub measures duration in milliseconds
    return audio_file_path,duration_seconds

def format_segments(segments):
    saved_segments = list(segments)
    sentence_timestamp = []
    words_timestamp = []
    speech_to_text = ""

    for i in saved_segments:
        temp_sentence_timestamp = {}
        # Store sentence information in sentence_timestamp
        text = i.text.strip()
        sentence_id = len(sentence_timestamp)  # Get the current index for the new entry
        sentence_timestamp.append({
            "id": sentence_id,  # Use the index as the id
            "text": text,
            "start": i.start,
            "end": i.end,
            "words": []  # Initialize words as an empty list within the sentence
        })
        speech_to_text += text + " "

        # Process each word in the sentence
        for word in i.words:
            word_data = {
                "word": word.word.strip(),
                "start": word.start,
                "end": word.end
            }

            # Append word timestamps to the sentence's word list
            sentence_timestamp[sentence_id]["words"].append(word_data)

            # Optionally, add the word data to the global words_timestamp list
            words_timestamp.append(word_data)

    return sentence_timestamp, words_timestamp, speech_to_text

def combine_word_segments(words_timestamp, max_words_per_subtitle=8, min_silence_between_words=0.5):
    before_translate = {}
    id = 1
    text = ""
    start = None
    end = None
    word_count = 0
    last_end_time = None

    for i in words_timestamp:
        try:
            word = i['word']
            word_start = i['start']
            word_end = i['end']

            # Check for sentence-ending punctuation
            is_end_of_sentence = word.endswith(('.', '?', '!'))

            # Check for conditions to create a new subtitle
            if ((last_end_time is not None and word_start - last_end_time > min_silence_between_words)
                or word_count >= max_words_per_subtitle
                or is_end_of_sentence):

                # Store the previous subtitle if there's any
                if text:
                    before_translate[id] = {
                        "text": text,
                        "start": start,
                        "end": end
                    }
                    id += 1

                # Reset for the new subtitle segment
                text = word
                start = word_start  # Set the start time for the new subtitle
                word_count = 1
            else:
                if word_count == 0:  # First word in the subtitle
                    start = word_start  # Ensure the start time is set
                text += " " + word
                word_count += 1

            end = word_end  # Update the end timestamp
            last_end_time = word_end  # Update the last end timestamp

        except KeyError as e:
            print(f"KeyError: {e} - Skipping word")
            pass

    # After the loop, make sure to add the last subtitle segment
    if text:
        before_translate[id] = {
            "text": text,
            "start": start,
            "end": end
        }

    return before_translate


def convert_time_to_srt_format(seconds):
    """ Convert seconds to SRT time format (HH:MM:SS,ms) """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"
def write_subtitles_to_file(subtitles, filename="subtitles.srt"):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    # Open the file with UTF-8 encoding
    with open(filename, 'w', encoding='utf-8') as f:
        for id, entry in subtitles.items():
            # Write the subtitle index
            f.write(f"{id}\n")
            if entry['start'] is None or entry['end'] is None:
              print(id)
            # Write the start and end time in SRT format
            start_time = convert_time_to_srt_format(entry['start'])
            end_time = convert_time_to_srt_format(entry['end'])
            f.write(f"{start_time} --> {end_time}\n")

            # Write the text and speaker information
            f.write(f"{entry['text']}\n\n")


def translate_subtitle(word_level_segments,Source_Language,Destination_Language):
    global language_dict
    store_text=""
    meta_translator = pipeline(
    task="translation",
    model="facebook/nllb-200-distilled-600M",
    torch_dtype=torch.bfloat16,device='cuda')

    translated_subtitles = {}

    for id, entry in word_level_segments.items():
        # Access the complete text for each subtitle block
        full_text = entry['text']
        # Translate the entire text of the subtitle block
        text_translated = meta_translator(full_text,
                          src_lang=language_dict[Source_Language]["meta_code"],
                          tgt_lang=language_dict[Destination_Language]["meta_code"])
        translated_text=text_translated[0]["translation_text"]
        # Reconstruct the subtitle with the translated text
        translated_subtitles[id] = {
            "text": translated_text,
            "start": entry['start'],
            "end": entry['end']
        }
        store_text+=translated_text.strip()+" "
    del meta_translator
    gc.collect()
    torch.cuda.empty_cache()
    return translated_subtitles,store_text



def whisper_subtitle(uploaded_file,Source_Language,Destination_Language):
  global language_dict,base_path
  #setup srt file names
  base_name = os.path.basename(uploaded_file).rsplit('.', 1)[0][:30]
  save_name = f"{base_path}/generated_subtitle/{base_name}_{Source_Language}.srt"
  original_srt_name=clean_file_name(save_name)
  translated_srt_name=save_name.replace(Source_Language,Destination_Language)
  original_txt_name=original_srt_name.replace(".srt",".txt")
  translated_txt_name=translated_srt_name.replace(".srt",".txt")
  #Load model
  faster_whisper_model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2")
  audio_path,audio_duration=get_audio_file(uploaded_file)

  if Source_Language=="Automatic":
      segments,d = faster_whisper_model.transcribe(audio_path, word_timestamps=True)
      lang_code=d.language
      src_lang=get_language_name(lang_code)
  else:
    lang=language_dict[Source_Language]['lang_code']
    segments,d = faster_whisper_model.transcribe(audio_path, word_timestamps=True,language=lang)
    src_lang=Source_Language
  # if os.path.exists(audio_path):
  #   os.remove(audio_path)


  sentence_timestamp,words_timestamp,text=format_segments(segments)
  del faster_whisper_model
  gc.collect()
  torch.cuda.empty_cache()

  word_segments=combine_word_segments(words_timestamp, max_words_per_subtitle=8, min_silence_between_words=0.5)
  write_subtitles_to_file(word_segments, filename=original_srt_name)
  with open(original_txt_name, 'w', encoding='utf-8') as f1:
    f1.write(text)
  if Destination_Language=="Same as Source Language":
    dest_lang=src_lang
  else:
    dest_lang=Destination_Language
  if src_lang!=dest_lang:
    translated_word_segments,translated_text=translate_subtitle(word_segments,src_lang,dest_lang)
    write_subtitles_to_file(translated_word_segments, filename=translated_srt_name)
    with open(translated_txt_name, 'w', encoding='utf-8') as f2:
      f2.write(translated_text)
  else:
    translated_srt_name=original_srt_name
    translated_txt_name=original_txt_name
  return original_srt_name,translated_srt_name,original_txt_name,translated_txt_name,audio_path

#@title Using Gradio Interface
# import gradio as gr
# def subtitle_maker(Audio_or_Video_File,Source_Language,Destination_Language):
#   try:
#     original_srt_file,translated_srt_file,original_text_file,translated_text_file,used_audio_path=whisper_subtitle(Audio_or_Video_File,Source_Language,Destination_Language)
#   except:
#     original_srt_file,translated_srt_file,original_text_file,translated_text_file,used_audio_path=None,None,None,None,None
#   # return original_srt_file,original_text_file,translated_srt_file,translated_text_file
#   return translated_srt_file





# source_lang_list=['Automatic']
# source_lang_list.extend(available_language)
# destination_lang_list=["Same as Source Language"]
# destination_lang_list.extend(available_language)
# # demo_examples = [["/content/audio/a.mp3"]]
# gradio_inputs=[gr.File(label="Upload Audio or Video File"),
#                gr.Dropdown(label="Source Language",choices=source_lang_list,value="Automatic"),
#                gr.Dropdown(label="Destination Language",choices=destination_lang_list,value="Same as Source Language")]
# gradio_outputs=[gr.File(label="Original SRT File",show_label=True),
#                 gr.File(label="Original Text File",show_label=True),
#                 gr.File(label="Translated SRT File",show_label=True),
#                 gr.File(label="Translated Text File",show_label=True)]
# demo = gr.Interface(fn=subtitle_maker, inputs=gradio_inputs,outputs=gradio_outputs , title="Whisper-Large-V3-Turbo-Ct2 Subtitle Maker")#,examples=demo_examples)
# demo.launch(debug=True,share=True)

# import locale
# locale.getpreferredencoding = lambda: "UTF-8"

# !pwd

# %cd $install_path

# actual_duration

#@title Generate Audio File From Subtitle
from tqdm.notebook import tqdm
import subprocess
import json
import pysrt
import os
from pydub import AudioSegment
import shutil
import uuid
import re
import time

os.chdir(install_path)

Reference_Audio_File,Clone_Method,Seed,Remove_Silence_From_TTS=None,None,None,None
def your_tts(text,audio_path,language,actual_duration):
  global Reference_Audio_File,Clone_Method,Seed,Remove_Silence_From_TTS
  if len(text)<=135:
    Split_Sentences=0
  if len(text)>135:
    Split_Sentences=2
  # actual_duration= abs(actual_duration)
  # actual_duration=None if actual_duration==0 else actual_duration
  actual_duration=None
  cloned_voice_path=voice_clone(Reference_Audio_File, text,remove_silence = Remove_Silence_From_TTS,fix_duration=actual_duration,chunks=Split_Sentences,exp_name=Clone_Method,progress_bar=False)
  shutil.copy(cloned_voice_path,audio_path)




def get_subtitle_Dub_path(srt_file_path,Language):
  file_name = os.path.splitext(os.path.basename(srt_file_path))[0]
  if not os.path.exists(f"{base_path}/TTS_DUB"):
    os.mkdir(f"{base_path}/TTS_DUB")
  random_string = str(uuid.uuid4())[:6]
  new_path=f"{base_path}/TTS_DUB/{file_name}_{random_string}.wav"
  return new_path







def get_video_duration(video_path):
    try:
        # Run ffmpeg command to get video information in JSON format
        result = subprocess.run(
            ['ffmpeg', '-i', video_path, '-f', 'ffmetadata', '-'],
            stderr=subprocess.PIPE,
            text=True
        )

        # Parse the duration from the stderr output
        for line in result.stderr.split('\n'):
            if 'Duration' in line:
                duration_str = line.split('Duration: ')[1].split(',')[0]
                h, m, s = duration_str.split(':')
                duration = int(h) * 3600 + int(m) * 60 + float(s)
                return duration
    except Exception as e:
        print(f"Error: {e}")
        return None


# def replace_audio(video_path,audio_path):
#   if not video_path.lower().endswith(".mp4"):
#     return
#   tts_audio = AudioSegment.from_file(dub_save_path)
#   audio_duration = len(tts_audio)/1000
#   video_duration=get_video_duration(video_path)
#   slience_duration=video_duration-audio_duration
#   audio_segment = AudioSegment.from_file(audio_path)
#   slience_Segment= AudioSegment.silent(duration=slience_duration)
#   marge_audio=audio_segment+slience_Segment
#   marge_audio.export(f"{base_path}/new_audio.wav", format="wav")
#   command=f"ffmpeg -i {video_path}  -i {base_path}/new_audio.wav -map 0:v -map 1:a -c:v copy -shortest {base_path}/output.mp4 -y"
#   var=os.system(command)
#   if var==0:
#     if os.path.exists("/content/gdrive/MyDrive/upload"):
#       file_name = os.path.basename(video_path)
#       shutil.copy("/content/output.mp4", f"/content/gdrive/MyDrive/upload/change_audio_{file_name}")
#       print(f"Copied at drive '/content/gdrive/MyDrive/upload/change_audio_{file_name}'")
#       return f"/content/gdrive/MyDrive/upload/change_audio_{file_name}"
#   else:
#     print(command)
#     return None







def clean_srt(input_path):
    file_name = os.path.basename(input_path)
    output_folder = f"{base_path}/save_srt"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    output_path = f"{output_folder}/{file_name}"

    def clean_srt_line(text):
        bad_list = ["[", "]", "♫", "\n"]
        for i in bad_list:
            text = text.replace(i, "")
        return text.strip()

    # Load the subtitle file
    subs = pysrt.open(input_path)

    # Iterate through each subtitle and print its details
    with open(output_path, "w", encoding='utf-8') as file:
        for sub in subs:
            file.write(f"{sub.index}\n")
            file.write(f"{sub.start} --> {sub.end}\n")
            file.write(f"{clean_srt_line(sub.text)}\n")
            file.write("\n")
        file.close()
    # print(f"Clean SRT saved at: {output_path}")
    return output_path
# Example usage




import shutil
import subprocess
from pydub import AudioSegment

class SRTDubbing:
    def __init__(self):
        pass
    @staticmethod
    def text_to_speech(text, audio_path, language, actual_duration):
        tts_filename = "temp.wav"
        your_tts(text, tts_filename, language, actual_duration)
        
        # Check the duration of the generated TTS audio
        tts_audio = AudioSegment.from_file(tts_filename)
        tts_duration = len(tts_audio)
    
        if actual_duration == 0:
            # If actual duration is zero, use the original TTS audio without modifications
            shutil.move(tts_filename, audio_path)
            return
    
        # If TTS audio duration is longer than actual duration, speed up the audio
        if tts_duration > actual_duration:
            speedup_factor = tts_duration / actual_duration
            speedup_filename = "speedup_temp.wav"
    
            # Use ffmpeg to change audio speed
            subprocess.run([
                "ffmpeg",
                "-i", tts_filename,
                "-filter:a", f"atempo={speedup_factor}",
                speedup_filename,
            "-y"], check=True)
    
            # Replace the original TTS audio with the sped-up version
            shutil.move(speedup_filename, audio_path)
        elif tts_duration < actual_duration:
            # If TTS audio duration is less than actual duration, add silence to match the duration
            silence_gap = actual_duration - tts_duration
            silence = AudioSegment.silent(duration=int(silence_gap))
            silence = silence.set_frame_rate(tts_audio.frame_rate)
            silence = silence.set_channels(tts_audio.channels)
    
            new_audio = tts_audio + silence
    
            # Save the new audio with added silence
            new_audio.export(audio_path, format="wav")
        else:
            # If TTS audio duration is equal to actual duration, use the original TTS audio
            shutil.move(tts_filename, audio_path)


    # @staticmethod
    # def text_to_speech(text, audio_path, language, actual_duration):
    #     tts_filename = "temp.wav"
    #     your_tts(text,tts_filename,language,actual_duration)
    #     # Check the duration of the generated TTS audio
    #     tts_audio = AudioSegment.from_file(tts_filename)
    #     tts_duration = len(tts_audio)

    #     if actual_duration == 0:
    #         # If actual duration is zero, use the original TTS audio without modifications
    #         shutil.move(tts_filename, audio_path)
    #         return

    #     # If TTS audio duration is longer than actual duration, speed up the audio
    #     if tts_duration > actual_duration:
    #         speedup_factor = tts_duration / actual_duration
    #         speedup_filename = "speedup_temp.wav"

    #         # Use ffmpeg to change audio speed
    #         subprocess.run([
    #             "ffmpeg",
    #             "-i", tts_filename,
    #             "-filter:a", f"atempo={speedup_factor}",
    #             speedup_filename
    #         ], check=True)

    #         # Replace the original TTS audio with the sped-up version
    #         shutil.move(speedup_filename, audio_path)
    #     elif tts_duration < actual_duration:
    #         # If TTS audio duration is less than actual duration, add silence to match the duration
    #         silence_gap = actual_duration - tts_duration
    #         silence = AudioSegment.silent(duration=int(silence_gap))
    #         new_audio = tts_audio + silence

        #     # Save the new audio with added silence
        #     new_audio.export(audio_path, format="wav")
        # else:
        #     # If TTS audio duration is equal to actual duration, use the original TTS audio
        #     shutil.move(tts_filename, audio_path)

    @staticmethod
    def make_silence(pause_time, pause_save_path):
        silence = AudioSegment.silent(duration=pause_time)
        silence.export(pause_save_path, format="wav")
        return pause_save_path

    @staticmethod
    def create_folder_for_srt(srt_file_path):
        srt_base_name = os.path.splitext(os.path.basename(srt_file_path))[0]
        random_uuid = str(uuid.uuid4())[:4]
        dummy_folder_path = f"{base_path}/dummy"
        if not os.path.exists(dummy_folder_path):
            os.makedirs(dummy_folder_path)
        folder_path = os.path.join(dummy_folder_path, f"{srt_base_name}_{random_uuid}")
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    @staticmethod
    def concatenate_audio_files(audio_paths, output_path):
        concatenated_audio = AudioSegment.silent(duration=0)
        for audio_path in audio_paths:
            audio_segment = AudioSegment.from_file(audio_path)
            concatenated_audio += audio_segment
        concatenated_audio.export(output_path, format="wav")

    def srt_to_dub(self, srt_file_path, dub_save_path,language='en'):
        result = self.read_srt_file(srt_file_path)
        new_folder_path = self.create_folder_for_srt(srt_file_path)
        join_path = []
        for i in tqdm(result):
        # for i in result:
            text = i['text']
            actual_duration = i['end_time'] - i['start_time']
            pause_time = i['pause_time']
            slient_path = f"{new_folder_path}/{i['previous_pause']}"
            self.make_silence(pause_time, slient_path)
            join_path.append(slient_path)
            tts_path = f"{new_folder_path}/{i['audio_name']}"
            self.text_to_speech(text, tts_path, language, actual_duration)
            join_path.append(tts_path)
        self.concatenate_audio_files(join_path, dub_save_path)

    @staticmethod
    def convert_to_millisecond(time_str):
      if isinstance(time_str, str):
          hours, minutes, second_millisecond = time_str.split(':')
          seconds, milliseconds = second_millisecond.split(",")

          total_milliseconds = (
              int(hours) * 3600000 +
              int(minutes) * 60000 +
              int(seconds) * 1000 +
              int(milliseconds)
          )

          return total_milliseconds
    @staticmethod
    def read_srt_file(file_path):
        entries = []
        default_start = 0
        previous_end_time = default_start
        entry_number = 1
        audio_name_template = "{}.wav"
        previous_pause_template = "{}_before_pause.wav"

        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # print(lines)
            for i in range(0, len(lines), 4):
                time_info = re.findall(r'(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)', lines[i + 1])
                start_time = SRTDubbing.convert_to_millisecond(time_info[0][0])
                end_time = SRTDubbing.convert_to_millisecond(time_info[0][1])

                current_entry = {
                    'entry_number': entry_number,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': lines[i + 2].strip(),
                    'pause_time': start_time - previous_end_time if entry_number != 1 else start_time - default_start,
                    'audio_name': audio_name_template.format(entry_number),
                    'previous_pause': previous_pause_template.format(entry_number),
                }

                entries.append(current_entry)
                previous_end_time = end_time
                entry_number += 1

        return entries

def srt_process(srt_file_path,ref_audio_path,clone_method="F5TTS_Base",dest_language="English",silence=True):
  global Reference_Audio_File,Clone_Method,Seed,Remove_Silence_From_TTS
  Reference_Audio_File=ref_audio_path
  Clone_Method=clone_method
  Seed=100000
  Remove_Silence_From_TTS = silence
  srt_dubbing = SRTDubbing()
  dub_save_path=get_subtitle_Dub_path(srt_file_path,dest_language)
  srt_dubbing.srt_to_dub(srt_file_path, dub_save_path,dest_language)
  return dub_save_path

# srt_file_path="/content/test.srt"
# ref_audio_path="/content/F5-TTS/tests/ref_audio/test_en_1_ref_short.wav"
# srt_process(srt_file_path,ref_audio_path,clone_method="F5TTS_Base",dest_language="English")



import gradio as gr

os.chdir(install_path)
def subtitle_maker(Audio_or_Video_File,Source_Language,Destination_Language):
  try:
    original_srt_file,translated_srt_file,original_text_file,translated_text_file,used_audio_path=whisper_subtitle(Audio_or_Video_File,Source_Language,Destination_Language)
  except:
    original_srt_file,translated_srt_file,original_text_file,translated_text_file,used_audio_path=None,None,None,None,None
  # return original_srt_file,original_text_file,translated_srt_file,translated_text_file
  return translated_srt_file

def clone_from_srt(Audio_or_Video_File,Source_Language,Destination_Language,ref_audio_path,clone_method):
  srt_path=subtitle_maker(Audio_or_Video_File,Source_Language,Destination_Language)
  srt_path=clean_srt(srt_path)
  dub_save_path=srt_process(srt_path,ref_audio_path,clone_method=clone_method)
  return dub_save_path,srt_path

source_lang_list=['Automatic']
source_lang_list.extend(available_language)
gradio_inputs=[gr.File(label="Upload Audio or Video File"),
               gr.Dropdown(label="Source Language",choices=source_lang_list,value="Automatic"),
               gr.Dropdown(label="Destination Language",choices=['English','Chinese'],value="English"),
               gr.Audio(label="Reference Audio", type="filepath"),
               gr.Dropdown(label="Choose TTS Model",choices=['F5TTS_Base', 'E2TTS_Base'],value="F5TTS_Base")]
gradio_outputs=[gr.File(label="Clone dubbing Voice",show_label=True),
                gr.File(label="Translated SRT File",show_label=True),
                ]
demo = gr.Interface(fn=clone_from_srt, inputs=gradio_inputs,outputs=gradio_outputs , title="F5-TTS Single Speaker Video Dubbing")#,examples=demo_examples)
demo.launch(debug=True,share=True)

