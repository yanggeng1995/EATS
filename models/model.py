import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class EETS(nn.Module):
    def __init__(self,
           encoder_dim,
           z_channels=128,
           s_channels=128):
        super(EETS, self).__init__()

        self.encoder_dim = encoder_dim
        self.z_channels = z_channels
        self.s_channels = s_channels

        self.phoneme_embedding = nn.Embedding(1000, encoder_dim)
        self.speaker_embedding = nn.Embedding(100, s_channels)

        self.encoder = Encoder(encoder_dim, z_channels, s_channels)
        self.decoder = Decoder(256, z_channels + s_channels)

    def forward(self, inputs, z, speakers):
        encoder_inputs = self.phoneme_embedding(inputs).transpose(1, 2)
        speaker_inputs = self.speaker_embedding(speakers).squeeze(1)

        decoder_inputs, duration = self.encoder(encoder_inputs, z, speaker_inputs)

        z_inputs = torch.cat([z, speaker_inputs], dim=1)
        print(decoder_inputs.shape, z_inputs.shape)
        outputs = self.decoder(decoder_inputs, z_inputs)

        return outputs, duration

if __name__ == "__main__":
    model = EETS(256, 128, 128)
    phoneme_inputs = torch.LongTensor([0, 1]).view(2, 1).repeat(1, 10)
    z = torch.randn(2, 128)
    speaker = torch.LongTensor([0, 1]).view(2, 1)
    outputs, duration = model(phoneme_inputs, z, speaker)
    print(outputs.shape)