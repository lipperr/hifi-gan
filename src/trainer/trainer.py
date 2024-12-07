import torch
from pathlib import Path
import pandas as pd 
from src.logger.utils import plot_melspectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import wandb

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

        if self.is_train:
            self.log_audio(**batch)
            self.log_melspectrogram(**batch)
        else:
            self.log_predictions(**batch)
            self.log_melspectrogram(**batch)



    def log_audio(self, audio_pred, **batch):
        if "audio" in batch:
            self.writer.add_audio(f"audio", batch["audio"][0], 22050)
        self.writer.add_audio(f"audio_pred", audio_pred[0], 22050)

    def log_melspectrogram(self, melspectrogram, melspectrogram_pred, **batch):
        
        melspectrogram_for_plot = melspectrogram[0].detach().cpu().squeeze(0)
        image = plot_melspectrogram(melspectrogram_for_plot)
        self.writer.add_image("melspectrogram", image)

        melspectrogram_pred_for_plot = melspectrogram_pred[0].detach().cpu().squeeze(0)
        image_pred = plot_melspectrogram(melspectrogram_pred_for_plot)
        self.writer.add_image("melspectrogram_pred", image_pred)

    def log_predictions(
        self, text, audio_pred, utterance_id, examples_to_log=10, **batch
    ):

        rows = {}
        rows["text"] = [
            query for query in text[:examples_to_log]
        ]
        rows["audio_pred"] = [wandb.Audio(
                    wav.detach().cpu().numpy().T, sample_rate=22050
                ) for wav in audio_pred[:examples_to_log]]

        mos_metric = self.metrics['inference'][0]
        rows["mos"] = [mos_metric(wav) for wav in audio_pred[:examples_to_log]]

        df = pd.DataFrame.from_dict(rows)
        df.index = [id for id in utterance_id[:examples_to_log]]

        self.writer.add_table("predictions from text", df)
