import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class EETS(nn.Module):
    def __init__(self,
           phone_embedding_dim=128,
           tone_embedding_dim=64,
           prosody_embedding_dim=32,
           seg_embedding_dim=32,
           spk_embedding_dim=64,
           z_dim=64):
        super(EETS, self).__init__()

        self.phone_embedding_dim = phone_embedding_dim
        self.tone_embedding_dim = tone_embedding_dim
        self.prosody_embedding_dim = prosody_embedding_dim
        self.seg_embedding_dim = seg_embedding_dim

        self.spk_embedding_dim = spk_embedding_dim
        self.z_dim = z_dim

        self.encoder_dim = phone_embedding_dim + tone_embedding_dim + prosody_embedding_dim + seg_embedding_dim

        self.phone_embed = nn.Embedding(100, phone_embedding_dim)
        self.tone_embed = nn.Embedding(8, tone_embedding_dim)
        self.prosody_embed = nn.Embedding(4, prosody_embedding_dim)
        self.seg_embed = nn.Embedding(4, seg_embedding_dim)
        self.speaker_embed = nn.Embedding(100, spk_embedding_dim)

        self.encoder = Encoder(self.encoder_dim, spk_embedding_dim, z_dim)
        self.expand_model = ExpandFrame()
        self.decoder = Decoder(256, spk_embedding_dim + z_dim)

    def forward(self, phone, tone, prosody, segment, noise, speakers, duration):
        phone_inputs = self.phone_embed(phone).transpose(1, 2)
        tone_inputs = self.tone_embed(tone).transpose(1, 2)
        prosody_inputs = self.prosody_embed(prosody).transpose(1, 2)
        segment_inputs = self.seg_embed(segment).transpose(1, 2)

        encoder_inputs = torch.cat([phone_inputs, tone_inputs, prosody_inputs, segment_inputs], dim=1)

        speaker_inputs = self.speaker_embed(speakers).squeeze(1)

        encoder_outputs, predict_duration = self.encoder(encoder_inputs, noise, speaker_inputs)

        decoder_inputs = self.expand_model(encoder_outputs, duration)
        
        z_inputs = torch.cat([noise, speaker_inputs], dim=1)
        outputs = self.decoder(decoder_inputs.transpose(1, 2), z_inputs)

        return outputs, predict_duration

class ExpandFrame(nn.Module):
    def __init__(self):
        super(ExpandFrame, self).__init__()
        pass

    def forward(self, encoder_outputs, duration):
        t = torch.round(torch.sum(duration, dim=-1, keepdim=True)) #[B, 1]
        e = torch.cumsum(duration, dim=-1).float() #[B, L]
        c = e - 0.5 * t #[B, L]

        t = torch.range(0, torch.max(t)) 
        t = t.unsqueeze(0).unsqueeze(1) #[1, 1, T]
        c = c.unsqueeze(2)
        w_1 = torch.exp(-0.1 * (t - c) ** 2)  # [B, L, T]
        w_2 = torch.sum(torch.exp(-0.1 * (t - c) ** 2), dim=1, keepdim=True)  # [B, 1, T]
        
        w = w_1 / w_2

        out = torch.matmul(w.transpose(1, 2), encoder_outputs)

        return out

if __name__ == "__main__":
    l = [[1.5, 2.3, 3.4, 4.4, 5.1, 4.2, 3.5, 2.6, 1.8, 0, 0, 0, 0],
     [1.5, 2.3, 3.4, 4.4, 5.1, 4.2, 3.5, 2.6, 1.8, 4.5, 5.5, 2.3, 5.6]]
    model = EETS()
    outputs = torch.randn(2, 13, 512)
    phone = torch.LongTensor([1,0,2,1,0,1,1,1,1,1,1,1,1]).view(1, 13).repeat(2, 1)
    speaker = torch.LongTensor([0, 1]).view(2, 1)
    duration = torch.FloatTensor(l)
    noise = torch.randn(2, 64)
    print(duration.shape)
    out, pduration= model(phone, phone, phone, phone, noise, speaker, duration)
    print(out.shape, pduration.shape)
