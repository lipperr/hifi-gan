from src.metrics.base_metric import BaseMetric
from src.metrics.mos.utils import Wav2Vec2MOS
import torchaudio

import os
import torch

class MOS(BaseMetric):
    def __init__(self, *args, **kwargs):
        path = os.path.join(os.path.expanduser('~'), ".cache/wv_mos/wv_mos.ckpt")
        self.model =  Wav2Vec2MOS(path)
        self.resample = torchaudio.transforms.Resample(orig_freq= 22050, new_freq= 16000)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.resample.to(self.device)
        super(MOS, self).__init__()

    def __call__(self, audio_pred, length=None, **kwargs):
        metric = 0
        for i, wav in enumerate(audio_pred):
            audio = torch.clone(wav).squeeze(0)
            out = self.resample(audio)

            if length is not None:
                out = out[:length].clone()
            elif "audio_len" in kwargs:
                out = out[:kwargs["audio_len"][i]].clone()
            
            metric += self.model(out)
        metric /= audio_pred.shape[0]
        return metric.item()
        