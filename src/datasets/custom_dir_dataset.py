from pathlib import Path
import os
from tqdm import tqdm
from src.datasets.base_dataset import BaseDataset
from src.datasets.mel_spec import MelSpectrogramText

class CustomDirDataset(BaseDataset):
    """
    Dataset of text samples to synthesize audio.
    """
    def __init__(self, transcription_dir, query="", save_path="", *args, **kwargs):
        self.type = "text"
        data = []
        self.melspectrogram = MelSpectrogramText()
        if query != "":
            data.append({"utterance_id": "my_query", "text": query, "text_len": len(query)})
            super().__init__(data, *args, **kwargs)
            return
        
        if not Path(transcription_dir + "/transcriptions").exists():
            print("WARNING: no folder 'transcriptions'. Searching for .csv file to create index...")
            assert Path(transcription_dir + "/metadata.csv").exists(), "Can't synthesize audio without transcriptions."

            transcriptions_file_name = transcription_dir + "/metadata.csv"
            with open(transcriptions_file_name, 'r', encoding='utf-8') as f:
                lines = f.read().split('\n')
                ids_trs = [ (x.split('|')[0], x.split('|')[2]) for x in lines if len(x) > 0]

            print("Found transcriptions in .csv file!")
            if os.access(transcription_dir, os.W_OK) or os.access(save_path, os.W_OK):
                transcriptions_dir = os.path.join(transcription_dir, "transcriptions") if os.access(transcription_dir, os.W_OK) else os.path.join(save_path, "transcriptions")
                os.makedirs(transcriptions_dir, exist_ok=True)
                for (id, text) in tqdm(ids_trs, desc=f"Preparing transcriptions folder..."):
                    file_path = os.path.join(transcriptions_dir, f"{id}.txt")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(text)
                
            else:
                print("Can't create transcriptions folder, preparing dataset from .csv file.")
                
                for id, text in ids_trs:
                    entry = {"utterance_id": id, "text": text, "text_len": len(text)}
                    data.append(entry)

        if Path(transcription_dir + "/transcriptions").exists() or Path(save_path + "/transcriptions").exists():
            dir = transcription_dir + "/transcriptions" if Path(transcription_dir + "/transcriptions").exists() else save_path + "/transcriptions"
            for path in Path(dir).iterdir():
                entry = {}
                if path.suffix == ".txt":
                    entry["utterance_id"] = path.stem
                    with path.open() as f:
                        entry["text"] = f.read().strip()
                        entry["text_len"] = len(entry["text"])

                    if Path(transcription_dir + "/wavs/" + path.stem + ".wav").exists():
                        audio_path = transcription_dir + "/wavs/" + path.stem + ".wav"
                        entry["audio_path"] = audio_path

                if len(entry) > 0:
                    data.append(entry)

        if len(data) == 0:
            print("WARNING: no transcriptions provided for synthesis.")
        super().__init__(data, *args, **kwargs)
        
    def __getitem__(self, idx):
        entry = self._index[idx]
        melspec = self.melspectrogram(entry["text"])

        instance_data = {
            "melspectrogram": melspec,
            "utterance_id": entry["utterance_id"],
            "text": entry["text"],
            "text_len": entry["text_len"]
            }
        
        if "audio_path" in entry.keys():
            audio = self.load_audio(entry["audio_path"])
            instance_data.update({"audio_gt": audio})
        
        return instance_data
        