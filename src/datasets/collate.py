import torch
from torch.nn.utils.rnn import pad_sequence
from src.datasets.mel_spec import MelSpectrogram, MelSpectrogramConfig

class Collate:
    def __init__(self):
        self.mel_config = MelSpectrogramConfig()
        self.melspectrogram = MelSpectrogram(self.mel_config)

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

            else:
                result_batch[key] = [item[key] for item in dataset_items]

        result_batch["audio"] = result_batch["audio"].unsqueeze(1)
        return result_batch