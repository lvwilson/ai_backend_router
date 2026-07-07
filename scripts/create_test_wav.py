import wave, struct, os

sample_rate = 16000
num_samples = sample_rate
wav_path = os.path.expanduser("~/roundtable_en.wav")

with wave.open(wav_path, "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    for i in range(num_samples):
        value = 16383 if (i % 2) == 0 else -16384
        wf.writeframes(struct.pack("<h", value))

print(f"Created {wav_path}, size={os.path.getsize(wav_path)} bytes")
