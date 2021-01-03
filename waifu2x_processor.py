# waifu2xのupconv7と同様のモデル構造のモデルです

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from MMD2illust_util import tensor_to_pil

MODEL_PATH = "latest_960_1920.pth"

class waifu2x_processor(object):
    def __init__(self):
        super().__init__()
        modules = [nn.ZeroPad2d(5),
        nn.Conv2d(3, 16, 3, 1, 0),
        nn.LeakyReLU(0.1, inplace=False),
        nn.Conv2d(16, 32, 3, 1, 0),
        nn.LeakyReLU(0.1, inplace=False),
        nn.Conv2d(32, 64, 3, 1, 0),
        nn.LeakyReLU(0.1, inplace=False),
        nn.Conv2d(64, 128, 3, 1, 0),
        nn.LeakyReLU(0.1, inplace=False),
        nn.Conv2d(128, 128, 3, 1, 0),
        nn.LeakyReLU(0.1, inplace=False),
        nn.Conv2d(128, 256, 3, 1, 0),
        nn.LeakyReLU(0.1, inplace=False),
        nn.ConvTranspose2d(256, 3, kernel_size=6, stride=2, padding=0, bias=False)
        ]
        self.model = nn.Sequential(*modules)
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model.eval()

        self.to_tensor = transforms.ToTensor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "model"):
            del self.model
        torch.cuda.empty_cache()

    def waifu2x_path(self, path):
        img = Image.open(path)
        return self.waifu2x_pil(img)

    def waifu2x_pil(self, img):
        with torch.no_grad():
            img = img.convert("RGB")
            tensor_img = self.to_tensor(img).unsqueeze(0)

            gpu_img = torch.tensor(0)
            if self.use_cuda:
                try:
                    gpu_img = tensor_img.cuda()
                    result_tensor_img = self.model(gpu_img)
                    result_pil_img = tensor_to_pil(result_tensor_img)
                    del gpu_img
                    del result_tensor_img
                    del tensor_img
                    return result_pil_img
                except:
                    print(gpu_img.shape)
                    del gpu_img
                    print("gpu error")

            result_tensor_img = self.model(tensor_img)
            result_pil_img = tensor_to_pil(result_tensor_img)
            del result_tensor_img
            del tensor_img
            return result_pil_img