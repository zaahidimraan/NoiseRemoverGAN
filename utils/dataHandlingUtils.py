import os
import numpy as np
import torchaudio as ta
import torch

def load(audio_file):
    waveform, sample_rate = ta.load(audio_file)
    waveform = waveform.to(torch.float32)
    return waveform[0, :]  

def save(audio_file, waveform, sample_rate):
    ta.save(audio_file, waveform, sample_rate)

def get_audio_files(directory):
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files

def increase_SampleRate_write(audio_file, new_sample_rate):
    waveform, sample_rate = ta.load(audio_file)
    resampler = ta.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    waveform = resampler(waveform)
    ta.save(audio_file, waveform, new_sample_rate)


def increase_SampleRate_read(audio_file, new_sample_rate):
    waveform, sample_rate = ta.load(audio_file)
    resampler = ta.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    waveform = resampler(waveform)
    return waveform

