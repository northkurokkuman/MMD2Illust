import numpy as np
import cv2
from PIL import Image
import torch
import torchvision

def tensor_to_pil(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                    normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    return img

def resize_pil_using_rate(img, rate):
    if rate == 1:
        return img
    width = int(img.width * rate) 
    height = int(img.height * rate)
    img_resize = img.resize((width, height))
    return img_resize

def resize_pils_using_rate(imgs, rate):
    out = []
    for img in imgs:
        out.append(resize_pil_using_rate(img, rate))
    return out

def read_img(img_path):
    img = Image.open(img_path).convert("RGB")
    return img

def read_imgs(img_paths):
    out = []
    for img_path in img_paths:
        out.append(read_img(img_path))
    return out

def hsv_tensor_to_rgb_tensor(img):
    h, s, v = img.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)

    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)

    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)

    return torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=img.dtype), a4)

def cv22pil(image):
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def pil2cv2(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def pil_add_margin(pil_img, top, right, bottom, left, color=(0, 0, 0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def pil_pad_to_size(pil_img, to_width = 1024, to_height = 576):
    is_pad = False
    w_rate = pil_img.width / to_width
    h_rate = pil_img.height / to_height
    can_pad = (w_rate >= 1) | (h_rate >= 1)
    if not can_pad:
        return pil_img, is_pad, None
    is_pad = True
    if w_rate > h_rate:
        pil_img = resize_pil_using_rate(pil_img, 1 / w_rate)
        real_size = pil_img.size
        top_padding = (to_height - pil_img.height) // 2
        bottom_padding = to_height - pil_img.height - top_padding
        pil_img = pil_add_margin(pil_img, top_padding, 0, bottom_padding, 0)
    else:
        pil_img = resize_pil_using_rate(pil_img, 1 / h_rate)
        real_size = pil_img.size
        left_padding = (to_width - pil_img.width) // 2
        right_padding = to_width - pil_img.width - left_padding
        pil_img = pil_add_margin(pil_img, 0, right_padding, 0, left_padding)
    return pil_img, is_pad, real_size

def get_pad_real_size(img_size, to_width = 1920, to_height = 1080):
    w_rate = img_size[0] / to_width
    h_rate = img_size[1] / to_height
    if w_rate > h_rate:
        real_size = (img_size[0] / w_rate, img_size[1] / w_rate)
    else:
        real_size = (img_size[0] / h_rate, img_size[1] / h_rate)
    return real_size

# img_sizeのどちらかがto_sizeのどちらかより大きいかどうか
def get_can_pad(img_size, to_width = 1920, to_height = 1080):
    w_rate = img_size[0] / to_width
    h_rate = img_size[1] / to_height
    can_pad = (w_rate >= 1) | (h_rate >= 1)
    return can_pad

def pil_remove_pad(pil_img, real_size):
    img_width, img_height = pil_img.size
    crop_width, crop_height = real_size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]

def get_over_size(width, height, to_width = 1024, to_height = 576):
    w_rate = to_width / width
    h_rate = to_height / height
    rate = np.min([w_rate, h_rate])
    return int(width * rate) * 2, int(height * rate) * 2

def get_video_info(video_path):
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return capture, fps, width, height, total_frame_count, fourcc

def normalize_tensor(img):
    img *= 2
    img -= torch.ones_like(img).to(img.device)
    return img 

def denormalize_tensor(img):
    img += torch.ones_like(img).to(img.device)
    img /= 2
    return img 