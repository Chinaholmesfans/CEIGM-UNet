from __future__ import annotations
import torch
from torch import Tensor
from torch import nn
from typing import Any

from model.encoder import Encoder2

from model.best_decoder import EMCAD as EMCAD22n


class MSVMUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 9,
        *,
        enc_name: str = "gm_tiny"  # tiny_0230s, small_0229s
    ) -> None:
        super(MSVMUNet, self).__init__()

        self.encoder = Encoder2("gm_tiny",in_channels=in_channels)
        
        self.dims = self.encoder.dims
        
        if self.dims[0] == 96 or self.dims[0] == 64:
            self.dims = self.dims[::-1]
        
        self.decoder = EMCAD22nn(channels=self.dims,num_classes=num_classes)

        print('Model %s created, param count: %d' %
                     ('current decoder: ', sum([m.numel() for m in self.decoder.parameters()])))

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor]:
        # print(x.shape)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.decoder(self.encoder(x)[::-1])

    def save(self, save_mode_path) -> None:
        torch.save(self.state_dict(), save_mode_path)
        

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        self.encoder.freeze_params()

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        self.encoder.unfreeze_params()


class MSVMUNet2(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 9,
        *,
        enc_name: str = "gm_tiny"  # tiny_0230s, small_0229s
    ) -> None:
        super(MSVMUNet2, self).__init__()
        self.encoder = Encoder6("ev_tiny",in_channels=in_channels)
        self.dims = self.encoder.dims
        for i in range(len(self.dims)):
            self.dims[i] = self.dims[i] // 16
        self.dims = self.dims[::-1]

        self.decoders = nn.ModuleList()
        for i in range(16):
            self.decoders.append(EMCAD22_2(channels=self.dims,num_classes=num_classes))

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor]:
        # print(x.shape)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.encoder(x)[::-1]
        xs = []
        xss = []
        os = []
        for each in x:
            xs.append(torch.chunk(each,16,dim = 1))
        for i in range(16):
            xss.append([xs[0][i],xs[1][i],xs[2][i]])
            
        for i in range(16):
            os.append(self.decoders[i](xss[i]))

        combined_tensor = torch.cat([
    torch.cat(os[i*4:(i+1)*4], dim=3) for i in range(4)
], dim=2)
        
        return combined_tensor

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        self.encoder.freeze_params()

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        self.encoder.unfreeze_params()


class MSVMUNet3(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 9,
        *,
        enc_name: str = "gm_tiny"  # tiny_0230s, small_0229s
    ) -> None:
        super(MSVMUNet3, self).__init__()

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor]:
        return x

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        self.encoder.freeze_params()

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        self.encoder.unfreeze_params() 



def build_model(**kwargs: Any) -> MSVMUNet:
    return MSVMUNet(**kwargs)
