
import sys
sys.path.append("../")

from torch import nn
from visionts import VisionTS

class Model(nn.Module):

    def __init__(self, config):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = config.task_name
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len

        self.vm = VisionTS(arch=config.vm_arch, finetune_type=config.ft_type, load_ckpt=config.vm_pretrained == 1, ckpt_dir=config.vm_ckpt)

        self.vm.update_config(context_len=config.seq_len, pred_len=config.pred_len, periodicity=config.periodicity, interpolation=config.interpolation, norm_const=config.norm_const, align_const=config.align_const)


    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):

        return self.vm.forward(x_enc)


    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raise NotImplementedError()


    def anomaly_detection(self, x_enc):
        raise NotImplementedError()

    def classification(self, x_enc, x_mark_enc):
        raise NotImplementedError()


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
