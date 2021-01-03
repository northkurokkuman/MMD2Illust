# # # MIT License

# # # Copyright (c) 2018 Arnab Kumar Mondal

# # # Permission is hereby granted, free of charge, to any person obtaining a copy
# # # of this software and associated documentation files (the "Software"), to deal
# # # in the Software without restriction, including without limitation the rights
# # # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # # copies of the Software, and to permit persons to whom the Software is
# # # furnished to do so, subject to the following conditions:

# # # The above copyright notice and this permission notice shall be included in all
# # # copies or substantial portions of the Software.

# # # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # # SOFTWARE.


import torch
import torch.nn as nn
from torch.nn import init
import torchvision.transforms as transforms
from PIL import Image
from MMD2illust_util import tensor_to_pil, cuda, normalize_tensor, denormalize_tensor
import functools

MODEL_PATH = "model.pth"

def conv_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding=0,
                                 norm_layer = nn.BatchNorm2d, bias = False):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias = bias),
        norm_layer(out_dim), nn.ReLU(True))

def dconv_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding=0, output_padding=0,
                                 norm_layer = nn.BatchNorm2d, bias = False):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride,
                           padding, output_padding, bias = bias),
        norm_layer(out_dim), nn.ReLU(True))

class ResidualBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResidualBlock, self).__init__()
        res_block = [nn.ReflectionPad2d(1),
                     conv_norm_relu(dim, dim, kernel_size=3, 
                     norm_layer= norm_layer, bias=use_bias)]
        if use_dropout:
            res_block += [nn.Dropout(0.5)]
        res_block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                      norm_layer(dim)]

        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        return x + self.res_block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True, num_blocks=6):
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        res_model = [nn.ReflectionPad2d(3),
                    conv_norm_relu(input_nc, ngf * 1, 7, norm_layer=norm_layer, bias=use_bias),
                    conv_norm_relu(ngf * 1, ngf * 2, 3, 2, 1, norm_layer=norm_layer, bias=use_bias),
                    conv_norm_relu(ngf * 2, ngf * 4, 3, 2, 1, norm_layer=norm_layer, bias=use_bias)]

        for i in range(num_blocks):
            res_model += [ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)]

        res_model += [dconv_norm_relu(ngf * 4, ngf * 2, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias),
                      dconv_norm_relu(ngf * 2, ngf * 1, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias),
                      nn.ReflectionPad2d(3),
                      nn.Conv2d(ngf, output_nc, 7),
                      nn.Tanh()]
        self.res_model = nn.Sequential(*res_model)

    def forward(self, x):
        return self.res_model(x)

def define_Gen(input_nc, output_nc, ngf, use_dropout=False):
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    gen_net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_blocks=9)
    return gen_net

def define_upper():
    return nn.Sequential(nn.ZeroPad2d(6),
        nn.Conv2d(3, 16, 3, 1, 0),
        nn.LeakyReLU(0.1, inplace=False),
        nn.Conv2d(16, 64, 3, 1, 0),
        nn.LeakyReLU(0.1, inplace=False),
        nn.Conv2d(64, 256, 3, 1, 0),
        nn.LeakyReLU(0.1, inplace=False),
        nn.Conv2d(256, 64, 3, 1, 0),
        nn.LeakyReLU(0.1, inplace=False),
        nn.Conv2d(64, 16, 3, 1, 0),
        nn.LeakyReLU(0.1, inplace=False),
        nn.Conv2d(16, 3, 3, 1, 0),
        nn.BatchNorm2d(3)
        )

def define_lower():
    return nn.Sequential(nn.BatchNorm2d(3),
        nn.ZeroPad2d(6),
        nn.Conv2d(3, 16, 3, 1, 0),
        nn.LeakyReLU(0.1, inplace=False),
        nn.Conv2d(16, 64, 3, 1, 0),
        nn.LeakyReLU(0.1, inplace=False),
        nn.Conv2d(64, 256, 3, 1, 0),
        nn.LeakyReLU(0.1, inplace=False),
        nn.Conv2d(256, 64, 3, 1, 0),
        nn.LeakyReLU(0.1, inplace=False),
        nn.Conv2d(64, 16, 3, 1, 0),
        nn.LeakyReLU(0.1, inplace=False),
        nn.Conv2d(16, 3, 3, 1, 0)
        )

def get_model(model_path, ngf=32):
    main_model = define_Gen(input_nc=3, output_nc=3, ngf=ngf, use_dropout=True)
    upper_model = define_upper()
    lower_model = define_lower()
    model = nn.Sequential(upper_model, main_model, lower_model)
    model.load_state_dict(torch.load(model_path))
    return model

class style_converter(object):
    def __init__(self):
        self.model = get_model(MODEL_PATH)
        self.model.eval()
        self.model = self.model.cuda()

        self.to_tensor = transforms.ToTensor()

    def __enter__(self) :
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "model"):
            del self.model
        torch.cuda.empty_cache()

    def convert(self, pil_img):
        with torch.no_grad():
            pil_img = pil_img.convert("RGB")
            tensor_img = self.to_tensor(pil_img)
            cuda_img = cuda(tensor_img)
            cuda_img = normalize_tensor(cuda_img)
            cuda_img = torch.unsqueeze(cuda_img,0)
            fake_tensor = self.model(cuda_img)
            fake_tensor = denormalize_tensor(fake_tensor)
            fake_tensor = torch.clamp(fake_tensor, 0, 1)
            pil_fake = tensor_to_pil(fake_tensor)
            return pil_fake