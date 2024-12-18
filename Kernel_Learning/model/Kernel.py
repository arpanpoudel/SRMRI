import torch
import torch.nn.functional as F
from torch import nn


# class Kernel(nn.Module):
#     def __init__(self, kernel_size, scaling_factors):
#         super(Kernel, self).__init__()
#         self.kernel_size = kernel_size
#         self.kernel = nn.Parameter(
#             torch.randn(1, 1, kernel_size, kernel_size)
#         )
    
#     def forward(self, image):
#         # Convolve the image with the kernel
#         convolved_image = F.conv2d(image, self.kernel, padding=self.kernel_size // 2)
    
#         return convolved_image

class Kernel(nn.Module):
    def __init__(self):
        super(Kernel, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3, padding=1)
        self.conv2=nn.Conv2d(in_channels=64,out_channels=1,kernel_size=3, padding=1)
        self.relu=nn.ReLU()
    
    def forward(self, image):
        # Convolve the image with the kernel
        convolved_image = self.conv1(image)
        convolved_image = self.relu(convolved_image)
        convolved_image = self.conv2(convolved_image)
    
        return convolved_image