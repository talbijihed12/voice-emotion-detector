import pyaudio

def list_audio_devices():
    p = pyaudio.PyAudio()

    print("Available audio playback devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info["maxOutputChannels"] > 0:
            print(f"Device ID: {device_info['index']}")
            print(f"  Name: {device_info['name']}")
            print(f"  Max Output Channels: {device_info['maxOutputChannels']}")
            print(f"  Default Sample Rate: {device_info['defaultSampleRate']}\n")

    p.terminate()

if __name__ == "__main__":
    list_audio_devices()