import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import l1_loss


class GanLoss(nn.Module):

    def __init__(self, lambda_fm=2, lambda_mel=45):
        super(GanLoss, self).__init__()
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel

    def mse(self, input, target):
        loss = 0
        for inp in input:
            loss += torch.mean((inp - target)**2)
        return loss
    

    def fm_loss(self,rfmap_mpd, ffmap_mpd, rfmap_msd, ffmap_msd, **batch):
        loss = 0
        for real_maps, fake_maps in zip(rfmap_mpd, ffmap_mpd):
            for real, fake in zip(real_maps, fake_maps):
                loss += l1_loss(real, fake)

        for real_maps, fake_maps in zip(rfmap_msd, ffmap_msd):
            for real, fake in zip(real_maps, fake_maps):
                loss += l1_loss(real, fake)

        return self.lambda_fm * loss
    

    def mel_loss(self, melspectrogram, melspectrogram_pred, **batch):
        
        loss = l1_loss(melspectrogram, melspectrogram_pred)
        return self.lambda_mel * loss


    def gen(
        self, flog_mpd, flog_msd, **batch
    ) -> Tensor:
        loss_adv = self.mse(flog_mpd, 1) + self.mse(flog_msd, 1)
        loss_fm = self.fm_loss(**batch)
        loss_mel = self.mel_loss(**batch)

        loss = loss_adv + loss_fm + loss_mel
        return {"loss_gen": loss, "feature_matching_loss": loss_fm, "mel_loss": loss_mel, "loss_gen_adv": loss_adv}
    
    
    def disc(
        self, rlog_mpd, flog_mpd, rlog_msd, flog_msd, **batch
    ) -> Tensor:
        
        loss = self.mse(rlog_mpd, 1) + self.mse(flog_mpd, 0) 
        loss += self.mse(rlog_msd, 1) + self.mse(flog_msd, 0)

        return {"loss_disc": loss}

