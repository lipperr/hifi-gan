from pathlib import Path
import torchaudio
import torch
from src.datasets.base_dataset import BaseDataset


class LJSpeechDataset(BaseDataset):
    def __init__(self, audio_dir, cut=True, *args, **kwargs):
        self.type="audio"
        self.cut = cut
        data = []
        for path in Path(audio_dir + "/wavs").iterdir():
            entry = {}
            if path.suffix == ".wav":
                entry["path"] = str(path)
                entry["utterance_id"] = str(path.stem)
                t_info = torchaudio.info(str(path))
                length = t_info.num_frames / t_info.sample_rate
                entry["audio_len"] = length
            if len(entry) > 0:
                data.append(entry)

        if len(data) == 0:
            print("WARNING: no audio files provided for training.")

        super().__init__(data, *args, **kwargs)

    def __getitem__(self, idx):

        entry = self._index[idx]
        audio = self.load_audio(entry["path"])
        length = audio.shape[1]
        if self.cut == True:
            wav_len = 8192
            audio_start = torch.randint(low=0, high=audio.shape[1] - wav_len, size=(1,))
            audio = audio[:, audio_start: audio_start + wav_len]
            

        instance_data = {
            "audio_len": length,
            "audio": audio,
            "path": entry["path"],
            "utterance_id": entry["utterance_id"]
        }

        instance_data = self.preprocess_data(instance_data)

        return instance_data