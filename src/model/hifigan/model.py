import torch.nn as nn

from src.model.hifigan.generator import Generator
from src.model.hifigan.discriminator import MPD, MSD
from src.datasets.mel_spec import MelSpectrogram, MelSpectrogramConfig

class HiFiGan(nn.Module):
    def __init__(self, hidden_dim=128):
        super(HiFiGan, self).__init__()
        self.gen = Generator(hidden_dim=hidden_dim)
        self.mpd = MPD()
        self.msd = MSD()
        self.melspec = MelSpectrogram(MelSpectrogramConfig)

    def generator(self, melspectrogram, **batch):
        
        audio_pred = self.gen(melspectrogram, **batch)
        melspectrogram_pred = self.melspec(audio_pred)

        return {"audio_pred": audio_pred, "melspectrogram_pred": melspectrogram_pred}

    def discriminator(self, audio, audio_pred, **batch):

        rlog_mpd, flog_mpd, rfmap_mpd, ffmap_mpd = self.mpd(audio, audio_pred)
        rlog_msd, flog_msd, rfmap_msd, ffmap_msd = self.msd(audio, audio_pred)

        return {"rlog_mpd": rlog_mpd,
                "flog_mpd": flog_mpd,
                "rfmap_mpd": rfmap_mpd,
                "ffmap_mpd": ffmap_mpd, 
                "rlog_msd": rlog_msd, 
                "flog_msd": flog_msd, 
                "rfmap_msd": rfmap_msd, 
                "ffmap_msd": ffmap_msd}


    def remove_weight_norm(self):
        self.gen.remove_weight_norm()
        self.mpd.remove_weight_norm()
        self.msd.remove_weight_norm()

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info