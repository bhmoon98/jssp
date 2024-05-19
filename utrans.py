import torch.nn as nn
import torch
import numpy as np

class LearnedPositionEncoding1(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, input_size=2000*64):
        super().__init__(input_size, d_model)
        self.input_size = input_size
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.view(1, self.input_size, self.d_model).expand_as(x)
        x = x + weight
        return self.dropout(x)


class LearnedPositionEncoding2(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, input_size):
        super().__init__(input_size, d_model)
        self.input_size = input_size
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.view(self.input_size, self.d_model).unsqueeze(0).expand_as(x)
        x = x + weight
        return self.dropout(x)

class UTransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=[4, 4, 4], dim_feedforward=2048, dropout=0.1, patch_sizes=[20, 10, 10], pixel_sizes=[4, 4, 4]):
        super(UTransformerEncoder, self).__init__()
        self.zeros = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=False)
        self.d_model = d_model
        self.patch_sizes = patch_sizes
        self.pixel_sizes = pixel_sizes

        self.global_position_embedding = LearnedPositionEncoding1(d_model=d_model, dropout=dropout, input_size=np.prod(patch_sizes) * np.prod(pixel_sizes))
        self.position_embedding = nn.ModuleList([
            LearnedPositionEncoding2(d_model=d_model, dropout=dropout, input_size=pixel_sizes[i] * patch_sizes[i])
            for i in range(len(patch_sizes))
        ])

        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="relu"),
                num_layers[i], encoder_norm
            ) for i in range(len(patch_sizes))
        ])

        self.bottle_neck = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 4), 
                nn.LayerNorm(d_model // 4), 
                nn.SELU(),
                nn.Linear(d_model // 4, d_model), 
                nn.LayerNorm(d_model)
            ) for i in range(len(patch_sizes))
        ])

        self.final_compression = nn.Linear(d_model * len(patch_sizes), 256)

    def calculate_size(self, level):
        S = self.pixel_sizes[level]
        P = 1
        for i in range(level + 1, len(self.pixel_sizes), 1):
            P *= self.pixel_sizes[i]
        return P, S

    def forwardDOWN(self, x, encoder_block, position_embedding, level):
        _, BPSPS, C = x.size()
        P, S = self.calculate_size(level)
        B = BPSPS // (P * S * P * S)
        x = x.view(B, P, S, P, S, C).permute(2, 4, 0, 1, 3, 5).contiguous().view(S * S, B * P * P, C)
        pad = self.zeros.expand(-1, B * P * P, -1)
        x = encoder_block(src=torch.cat((pad.detach(), position_embedding(x)), dim=0))

        latent_patch = x[0, :, :].unsqueeze(0).contiguous()
        latent_pixel = x[1:, :, :].contiguous()

        return latent_patch, latent_pixel

    def forward(self, x):
        B, L, C = x.size()
        H = int(np.sqrt(L))
        W = H
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        
        x = self.global_position_embedding(x.view(B, -1)).view(B, -1, self.d_model)  # (B, 2000*64, d_model)
        
        latent_list = []
        for i in range(len(self.encoder)):
            x, l = self.forwardDOWN(x=x, encoder_block=self.encoder[i], position_embedding=self.position_embedding[i], level=i)
            latent_list.append(self.bottle_neck[i](l))

        x = torch.cat([latent.view(B, -1) for latent in latent_list], dim=-1)  # (B, combined_features)
        print(x.shape)
        x = self.final_compression(x)  # (B, 256)

        return x