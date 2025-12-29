import numpy as np
import soundfile as sf
import os

def construct_mix_matrix(file_list):
    """
    Construct an m x n matrix from multiple audio files
    where m is the max length of audio files and n is the number of files
    Shorter files are zero-padded to match the longest file
    """
    audio_data_list = []
    max_length = 0
    
    # Read each file and store the float data
    for file in file_list:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, 'output', file)
        data, samplerate = sf.read(data_path, dtype='float32')
        # If stereo, convert to mono by averaging channels
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        audio_data_list.append(data)
        if len(data) > max_length:
            max_length = len(data)
    
    # Create matrix with zero padding for shorter files
    mix_matrix = np.zeros((max_length, len(file_list)), dtype=np.float32)
    for i, data in enumerate(audio_data_list):
        mix_matrix[:len(data), i] = data
    
    output_file = 'mix.dat'
    np.savetxt(output_file, mix_matrix, fmt='%.8f')
    return mix_matrix

# Example usage:
file_list = ['mph1.wav','mph2.wav']
mix_matrix = construct_mix_matrix(file_list)
print(f"Matrix shape: {mix_matrix.shape}")  # (m, n) format
print(f"Saved to mix.dat")
