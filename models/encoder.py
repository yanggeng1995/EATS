import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from modules import Conv1d, Linear


class Encoder(nn.Module):
    def __init__(self,
           encoder_dim,
           z_channels,
           s_channels):
        super(Encoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.z_channels = z_channels
        self.s_channels = s_channels

        self.aligner = Aligner(encoder_dim, z_channels, s_channels, 10)
        self.expand_frame_model = ExpandFrame()

    def forward(self, encoder_inputs, z, speaker_inputs):
        encoder_outputs, duration = self.aligner(encoder_inputs, z, speaker_inputs)

        decoder_inputs = self.expand_frame_model(encoder_outputs, duration).transpose(1, 2)

        return decoder_inputs, duration

class Aligner(nn.Module):
    def __init__(self,
           embed_dim,
           z_channels,
           s_channels,
           num_dilation_layer=10):
        super(Aligner, self).__init__()

        self.embed_dim = embed_dim
        self.z_channels = z_channels
        self.s_channels = s_channels

        self.pre_process = Conv1d(embed_dim, 256, kernel_size=3)

        self.dilated_conv_layers = nn.ModuleList()
        for i in range(num_dilation_layer):
            dilation = 2**i
            self.dilated_conv_layers.append(DilatedConvBlock(256, 256,
                        z_channels, s_channels, dilation))

        self.post_process = nn.Sequential(
            Linear(256, 256),
            nn.ReLU(inplace=False),
            Linear(256, 1),
            nn.ReLU(inplace=False),
        )

    def forward(self, inputs, z, s):
        outputs = self.pre_process(inputs)
        for layer in self.dilated_conv_layers:
            outputs = layer(outputs, z, s)
        
        encoder_outputs = outputs.transpose(1, 2)
        duration = self.post_process(outputs.transpose(1, 2))

        return encoder_outputs, duration

class ExpandFrame(nn.Module):
    def __init__(self):
        super(ExpandFrame, self).__init__()

    def forward(self, encoder_outputs, duration):
        duration = (duration + 1.5).long()
        batch_size = encoder_outputs.size(0)

        lists = []
        for i in range(batch_size):
            lists.append(self.one_batch_expand(encoder_outputs[i], duration[i]))
        outputs = torch.stack(lists, dim=0)

        return outputs

    def one_batch_expand(self, encoder_outputs, duration):
        num_phoneme = encoder_outputs.size(0)
        sums = torch.sum(duration, dim=0, keepdim=True).repeat(num_phoneme, 1)
        center = (sums - (0.5 * duration)).float()
        lists = []
        for t in range(sums[0, 0]):
            for n in range(num_phoneme):
                x = torch.exp(-0.1 * (t - center[n : n + 1])**2)
                y = torch.sum(torch.exp(-0.1 * (t - center)**2), dim=0)
                w_n_t = x / y
            lists.append(torch.sum(w_n_t * encoder_outputs, dim=0))
        output = torch.stack(lists, dim=0)

        return output

class DilatedConvBlock(nn.Module):

    """A stack of dilated convolutions interspersed
      with batch normalisation and ReLU activations """

    def __init__(self,
           in_channels,
           out_channels,
           z_channels,
           s_channels,
           dilation):
        super(DilatedConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.s_channels = s_channels

        self.conv1d = Conv1d(in_channels, out_channels, kernel_size=3, dilation=dilation)
        self.batch_layer = BatchNorm1dLayer(out_channels, s_channels, z_channels)

    def forward(self, inputs, z, s):
        outputs = self.conv1d(inputs)
        outputs = self.batch_layer(outputs, z, s)
        return F.relu(outputs)

class BatchNorm1dLayer(nn.Module):

    """The latents z and speaker embedding s modulate the scale and
     shift parameters of the batch normalisation layers"""

    def __init__(self,
           num_features,
           s_channels=128,
           z_channels=128):
      super().__init__()

      self.num_features = num_features
      self.s_channels = s_channels
      self.z_channels = z_channels
      self.batch_nrom = nn.BatchNorm1d(num_features, affine=False)

      self.scale_layer = spectral_norm(nn.Linear(z_channels, num_features))
      self.scale_layer.weight.data.normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
      self.scale_layer.bias.data.zero_()        # Initialise bias at 0

      self.shift_layer = spectral_norm(nn.Linear(s_channels, num_features))
      self.shift_layer.weight.data.normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
      self.shift_layer.bias.data.zero_()        # Initialise bias at 0

    def forward(self, inputs, z, s):
      outputs = self.batch_nrom(inputs)
      scale = self.scale_layer(z)
      scale = scale.view(-1, self.num_features, 1)

      shift = self.shift_layer(s)
      shift = shift.view(-1, self.num_features, 1)

      outputs = scale * outputs + shift

      return outputs

if __name__ == "__main__":
    model = Encoder(256, 128, 128)
    phoneme_inputs = torch.LongTensor([0, 1]).view(2, 1).repeat(1, 10)
    z = torch.randn(2, 128)
    speaker = torch.LongTensor([0, 1]).view(2, 1)
    outputs, duration = model(phoneme_inputs, z, speaker)
    print(outputs.shape)