import numpy as np
import soundfile as sf

sr = 22050
t = np.linspace(0, 10, sr*10)

tone = 0.5*np.sin(2*np.pi*440*t)

sf.write("song.wav", tone, sr)

print("test audio created: song.wav")