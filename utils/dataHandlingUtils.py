import os
import torchaudio 
import torch
import matplotlib.pyplot as plt
import mimetypes  
from IPython.display import Audio




def load(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)
    waveform = waveform.to(torch.float32)
    return waveform[0, :]  

def save(audio_file, waveform, sample_rate):
    torchaudio.save(audio_file, waveform, sample_rate)

def get_audio_files(directory):
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files

def increase_SampleRate_write(audio_file, new_sample_rate):
    waveform, sample_rate = torchaudio.load(audio_file)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    waveform = resampler(waveform)
    torchaudio.save(audio_file, waveform, new_sample_rate)


def increase_SampleRate_read(audio_file, new_sample_rate):
    waveform, sample_rate = torchaudio.load(audio_file)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    waveform = resampler(waveform)
    return waveform


def plot_spectrogram(waveform, sample_rate, title="Spectrogram", xlim=None):
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

def play_audio(waveform, sample_rate):
    return Audio(waveform.numpy(), rate=sample_rate)


def get_sample_rate(audio_file):
    metadata = torchaudio.info(audio_file)
    return metadata.sample_rate


def count_files_in_folder(folder_path):
    return len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])


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


