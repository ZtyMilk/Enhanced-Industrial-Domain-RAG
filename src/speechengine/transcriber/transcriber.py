import numpy
from funasr.auto.auto_model import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


class SenseVoiceTranscriber:
    def __init__(
        self,
        voice_model: AutoModel,
    ):
        self.voice_model = voice_model

    def transcribe(self, audio_data: numpy.ndarray):
        INJECTION_HOTWORDS = "型腔 保压时间 模具温度 残余应力 浇口冻结 翘曲 气孔 缩水率 压铸"
        res = self.voice_model.generate(
            input=audio_data,
            language="zh",
            hotword=INJECTION_HOTWORDS,
            use_itn=True,
        )
        if not res or "text" not in res[0]:
            return ""
        raw_text = res[0]["text"]
        cleaned_text = rich_transcription_postprocess(raw_text)
        return cleaned_text.replace(" ", "").strip()
