import numpy
import sounddevice


class AudioRecorder:
    def __init__(self, sample_rate: int, channels: int):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames = []

    def record(self):
        self.frames = []
        input("\n[Type 'Enter' before asking]")
        print("HEARING...", end="")
        stream = sounddevice.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        with stream:
            input()
        if not self.frames:
            return numpy.array([], dtype=numpy.float32)
        return numpy.concatenate(self.frames, axis=0).squeeze()

    def _callback(self, indata, frame_count, time_info, status):
        if status:
            print(f"[STATUS];{status}")
        self.frames.append(indata.copy())
