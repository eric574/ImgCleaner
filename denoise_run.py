import torch
from torch import nn
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as tvF
from PIL import Image
from io import BytesIO

def load_image(img_path):
    #Loads image as PIL 
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')
    
    return image

def _random_crop(img_list):
    #Performs random square crop of fixed size.
    w, h = img_list[0]._size
    w_crop = (w // 32) * 32
    h_crop = (h // 32) * 32
    crop_size = min(w_crop, h_crop)
    
    cropped_imgs = []
    i = np.random.randint(0, h - h_crop + 1)
    j = np.random.randint(0, w - w_crop + 1)
    
    
    for img in img_list:
        # Resize if dimensions are too small
        if min(w, h) < crop_size:
            img = tvF.resize(img, (crop_size, crop_size))
        
        
        # Random crop
        cropped_imgs.append(tvF.crop(img, i, j, h_crop, w_crop))
    
    return cropped_imgs

class UNet(nn.Module):
    """Custom U-Net architecture"""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        # print(x.size())
        pool1 = self._block1(x)
        # print(pool1.size())
        pool2 = self._block2(pool1)
        # print(pool2.size())
        pool3 = self._block2(pool2)
        # print(pool3.size())
        pool4 = self._block2(pool3)
        # print(pool4.size())
        pool5 = self._block2(pool4)
        # print(pool5.size())
        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        return self._block6(concat1)

def denoise(input_image_path):
    #Fixes grainy image
    model = UNet()
    model.load_state_dict(torch.load('ckpts/gaussian/n2n-gaussian.pt',map_location='cpu'))
    image = load_image(input_image_path)
    myimage = _random_crop([image])
    preprocess = transforms.ToTensor()
    images = preprocess(myimage[0])
    output = model(torch.stack([images]))[0]
    output_image = transforms.functional.to_pil_image(output, mode=None)
    return output_image

def detext(input_image_path):
    #Fixes image with text overlay
    model = UNet()
    model.load_state_dict(torch.load('ckpts/text/n2n-text.pt',map_location='cpu'))
    image = load_image(input_image_path)
    myimage = _random_crop([image])
    preprocess = transforms.ToTensor()
    images = preprocess(myimage[0])
    with torch.no_grad():
        output = model(torch.stack([images]))[0]
        output_image = transforms.functional.to_pil_image(output, mode=None)
    return output_image
