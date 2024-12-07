import torch
from torch.nn.utils.rnn import pad_sequence
from src.datasets.mel_spec import MelSpectrogram, MelSpectrogramConfig, MelSpectrogramText

class Collate:
    def __init__(self, melspec_type):
        if melspec_type == "audio":
            self.mel_config = MelSpectrogramConfig()
            self.melspectrogram = MelSpectrogram(self.mel_config)
        else:
            self.melspectrogram = MelSpectrogramText()

    def __call__(self, dataset_items: list[dict]):
        """
        Collate and pad fields in the dataset items.
        Converts individual items into a batch.

        Args:
            dataset_items (list[dict]): list of objects from
                dataset.__getitem__.
        Returns:
            result_batch (dict[Tensor]): dict, containing batch-version
                of the tensors.
        """
        result_batch = {}
        keys = dataset_items[0].keys()

        for key in keys:
            if  key == "audio":
                result_batch[key] = pad_sequence(
                    [item[key].squeeze(0).t() for item in dataset_items],
                    batch_first=True,
                )
                result_batch["melspectrogram"] = self.melspectrogram(result_batch["audio"])
                result_batch["audio"] = result_batch["audio"].unsqueeze(1)
                
            elif key == "text":
                result_batch["melspectrogram"] = pad_sequence(
                    [self.melspectrogram(item[key]).squeeze(0).t() for item in dataset_items],
                    batch_first=True, padding_value=-11.5129251
                )
                result_batch["melspectrogram"] = result_batch["melspectrogram"].permute(0, 2, 1)
                result_batch[key] = [item[key] for item in dataset_items]
            else:
                result_batch[key] = [item[key] for item in dataset_items]

        return result_batch