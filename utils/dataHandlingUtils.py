import os
import torchaudio 
import torch
import matplotlib.pyplot as plt
import mimetypes  
from IPython.display import Audio
from torchvision import transforms
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
import os
import random

## Tested
# Load audio file
# audio_file: path to the audio file
# return: waveform, sample_rate
def load(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)
    waveform = waveform.to(torch.float32) # Convert to float tensor (it is to make sure, otherwise it does not required)
    return waveform, sample_rate

## Tested
# Check if audio is mono or stereo
# audio_file: path to the audio file
# return: None
def is_mono(audio_file):
    waveform, _ = torchaudio.load(audio_file)  # Load waveform (sample_rate not needed here)
    waveform.shape[0] == 1

    if waveform.shape[0] == 1:
        print("Audio is mono.")
    else:
        print("Audio is stereo.")

# Save audio file
# audio_file: path to the audio file
# waveform: waveform to save
# sample_rate: sample rate of the waveform
# return: None
def save(audio_file, waveform, sample_rate):
    torchaudio.save(audio_file, waveform, sample_rate)

# get audio files
# directory: path to the directory
# return: audio_files
def get_audio_files(directory):
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files

# ================== Resampling ==================
## Tested
# increase sample rate and save on same path with same name
# audio_file: path to the audio file
# new_sample_rate: new sample rate
# return: None
def reshape_SampleRate_write(audio_file, new_sample_rate):
    waveform, sample_rate = load(audio_file)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    waveform = resampler(waveform)
    torchaudio.save(audio_file, waveform, new_sample_rate)
    
## Tested
# increase sample rate and save on different path with same name in a folder
def reshape_SampleRate_write_folder(new_sample_rate, folder_path):
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath) and filename.endswith('.wav'):
            if get_sample_rate(filepath) != new_sample_rate:
                reshape_SampleRate_write(filepath, new_sample_rate)

# increase sample rate and return waveform
# audio_file: path to the audio file
# new_sample_rate: new sample rate
# return: waveform
def reshape_SampleRate_read(audio_file, new_sample_rate):
    waveform, sample_rate = load(audio_file)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    waveform = resampler(waveform)
    return waveform




## Tested
# plot waveform
# audio_file: path to the audio file
# sample_rate: sample rate of the waveform
# title: title of the plot
# xlim: x limit of the plot
# return: None
def plot_spectrogram(audio_file, title="Spectrogram", xlim=None):
    waveform, sample_rate = load(audio_file)
    spectrogram = torchaudio.transforms.Spectrogram()(waveform)
    plt.figure()
    plt.imshow(spectrogram.log2()[0, :, :].numpy(), cmap='viridis', origin='lower', aspect='auto')
    plt.title(title)
    plt.xlabel("Frequency bin")
    plt.ylabel("Frame")

    if xlim:
        plt.xlim(xlim)

    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

## Tested
# play audio
# audio_file: path to the audio file
# sample_rate: sample rate of the waveform
# return: None
def play_audio(audio_file):
    waveform, sample_rate = load(audio_file)
    return Audio(waveform.numpy(), rate=sample_rate)

## Tested
# get sample rate of the audio file
# audio_file: path to the audio file
# return: sample rate
def get_sample_rate(audio_file):
    metadata = torchaudio.info(audio_file)
    return metadata.sample_rate

## Tested
# count files in a folder
# folder_path: path to the folder
# return: number of files in the folder
def count_files_in_folder(folder_path):
    return len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])


# check if a folder contains only audio files
# folder_path: path to the folder
# return: True if all files are audio, False otherwise
def is_audio_folder(folder_path):
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            mime_type, _ = mimetypes.guess_type(filepath)
            if not mime_type or not mime_type.startswith('audio/'):
                return False  # Not an audio file
    return True  # All files are audio


def check_sample_rate_consistency(folder_path):
    sample_rates = set()
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            sample_rate = get_sample_rate(filepath)
            sample_rates.add(sample_rate)
    
    if len(sample_rates) == 1:
        return True, sample_rates.pop()  # All have the same sample rate
    else:
        return False, None  # Different sample rates


# AudioDataset class
# mixed_dir: path to the directory containing mixed audio files
# clean_dir: path to the directory containing clean audio files
# return: mixed_spec, clean_spec

class AudioDataset(Dataset):
    def __init__(self, mixed_dir, clean_dir, cache_dir="preprocessed_data"):
        self.mixed_files = [os.path.join(mixed_dir, f) for f in os.listdir(mixed_dir) if f.endswith('.wav')]
        self.clean_files = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.wav')]
        self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)  # Create cache directory if it doesn't exist
        # for idx in range(len(self.mixed_files)):
        #     mixed_cache_file = os.path.join(self.cache_dir, f"mixed_{idx}.pt")
        #     clean_cache_file = os.path.join(self.cache_dir, f"clean_{idx}.pt")
        #     # Load and preprocess if not cached
        #     mixed_waveform, _ = torchaudio.load(self.mixed_files[idx])
        #     clean_waveform, _ = torchaudio.load(self.clean_files[idx])

        #     mixed_spec = torchaudio.transforms.Spectrogram(n_fft=1022, win_length=1022, hop_length=256)(mixed_waveform)
        #     clean_spec = torchaudio.transforms.Spectrogram(n_fft=1022, win_length=1022, hop_length=256)(clean_waveform)

        #     # Random cropping
        #     mixed_spec = transforms.RandomCrop(size=(256, 512),pad_if_needed=True)(mixed_spec) 
        #     clean_spec = transforms.RandomCrop(size=(256, 512),pad_if_needed=True)(clean_spec) 

        #     # Save to cache
        #     torch.save(mixed_spec, mixed_cache_file)
        #     torch.save(clean_spec, clean_cache_file)

    def __len__(self):
        return len(self.mixed_files)

    def __getitem__(self, idx):
        mixed_cache_file = os.path.join(self.cache_dir, f"mixed_{idx}.pt")
        clean_cache_file = os.path.join(self.cache_dir, f"clean_{idx}.pt")
        mixed_spec = None
        clean_spec = None

        # Check if cached data exists
        if os.path.exists(mixed_cache_file) and os.path.exists(clean_cache_file):
            mixed_spec = torch.load(mixed_cache_file)
            clean_spec = torch.load(clean_cache_file)
        else:
            # Load and preprocess if not cached
            mixed_waveform, _ = torchaudio.load(self.mixed_files[idx])
            clean_waveform, _ = torchaudio.load(self.clean_files[idx])

            mixed_spec = torchaudio.transforms.Spectrogram(n_fft=1022, win_length=1022, hop_length=256)(mixed_waveform)
            clean_spec = torchaudio.transforms.Spectrogram(n_fft=1022, win_length=1022, hop_length=256)(clean_waveform)

            # Random cropping
            mixed_spec = transforms.RandomCrop(size=(256, 512),pad_if_needed=True)(mixed_spec) 
            clean_spec = transforms.RandomCrop(size=(256, 512),pad_if_needed=True)(clean_spec) 

            # Save to cache
            torch.save(mixed_spec, mixed_cache_file)
            torch.save(clean_spec, clean_cache_file)

        return mixed_spec, clean_spec





