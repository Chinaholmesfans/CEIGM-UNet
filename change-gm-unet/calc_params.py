import torch
from thop import profile
from model import build_model
from utils import print_flops_params
from fvcore.nn import FlopCountAnalysis

torch.set_float32_matmul_precision("medium")
device: str = "cuda" if torch.cuda.is_available() else "cpu"

class Calc():
    def __init__(self) -> None:
        super(Calc, self).__init__()
        self.num_classes = 9

        self._model = build_model(
            in_channels=3,
            num_classes=self.num_classes,
        ).to(device)
        
    def lists(self) -> None:

        #input_tensor = torch.randn(1, 1, 256, 256).to(device)
        
        #flops, params = profile(self._model, inputs=(input_tensor,))

        # 将参数量转换为 M（百万）
        #params_m = params / 1e6  # 1e6 表示 1 百万
        
        # 将 FLOPS 转换为 G（十亿）
        #flops_g = flops / 1e9  # 1e9 表示 1 十亿

        input_tensor2 = torch.randn(1, 1, 224, 224).to(device)
        flops = FlopCountAnalysis(self._model, input_tensor2).total()

        total = sum([param.numel() for param in self._model.parameters()])
        print('Number of parameter:%.2fM' % (total / 1e6))
        
        flops2, params = profile(self._model, inputs=(input_tensor2,))

        params_m = params / 1e6  # 1e6 表示 1 百万

        flops_g2 = flops2 / 1e9  # 1e9 表示 1 十亿
        flops_g = flops / 1e9  # 1e9 表示 1 十亿
        
        print(f"Parameters: {params_m:.2f} M")
        print(f"Size: (224x224) FLOPS: {flops_g2:.2f} G")
        print(f"Size: (224x224) FLOPS: {flops_g:.2f} G")
        # print(f"Size: (256x256) FLOPS: {flops_g:.2f} G")

        # print_flops_params(self._model,(1,1,224,224))
        # print_flops_params(self._model,(1,3,256,256))
    

if __name__ == "__main__":
    c = Calc()
    c.lists()

