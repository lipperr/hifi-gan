import torch

from src.logger.utils import plot_melspectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.datasets.mel_spec import MelSpectrogramConfig


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster
        metric_funcs = self.metrics["inference"]

        outputs_gen = self.model.generator(**batch)
        batch.update(outputs_gen)

        if self.is_train:
            metric_funcs = self.metrics["train"]

            self.optimizer[0].zero_grad()
            outputs_disc = self.model.discriminator(batch["audio"], batch["audio_pred"].detach())
            batch.update(outputs_disc)
            loss_disc = self.criterion.disc(**batch)
            batch.update(loss_disc)
            batch["loss_disc"].backward()
            self._clip_grad_norm(self.model.mpd)
            self._clip_grad_norm(self.model.msd)
            self.optimizer[0].step()
            if self.lr_scheduler[0] is not None:
                self.lr_scheduler[0].step()

            torch.autograd.set_detect_anomaly(True)
            self.optimizer[1].zero_grad()
            outputs_disc = self.model.discriminator(**batch)
            batch.update(outputs_disc)
            loss_gen = self.criterion.gen(**batch)
            batch.update(loss_gen)
            batch["loss_gen"].backward()
            self._clip_grad_norm(self.model.gen)
            self.optimizer[1].step()
            if self.lr_scheduler[1] is not None:
                self.lr_scheduler[1].step()

            # update metrics for each loss (in case of multiple losses)
            for loss_name in self.config.writer.loss_names:
                metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        self.log_audio(mode, **batch)


    def log_audio(self, mode, **batch):
        if "audio" in batch:
            self.writer.add_audio(f"audio", batch["audio"][0], 22050)
        if "audio_pred" in batch:
            self.writer.add_audio(f"audio_pred", batch["audio_pred"][0], 22050)


    def log_melspectrogram(self, melspectrogram, melspectrogram_pred, **batch):

        melspectrogram_for_plot = melspectrogram[0].detach().cpu()
        image = plot_melspectrogram(melspectrogram_for_plot, sr=22050, hop_length=MelSpectrogramConfig.hop_length)
        self.writer.add_image("melspectrogram", image)

        melspectrogram_pred_for_plot = melspectrogram_pred[0].detach().cpu()
        image_pred = plot_melspectrogram(melspectrogram_pred_for_plot, sr=22050, hop_length=MelSpectrogramConfig.hop_length)
        self.writer.add_image("melspectrogram_pred", image_pred)

    def log_predictions(
        self, examples_to_log=10, **batch
    ):
        pass
