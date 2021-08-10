import torch
import torch.nn as nn

class featmatch1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(featmatch1x1, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class featmatch2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super(featmatch2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size,
                              padding=self.padding)

    def forward(self, x):
        return self.conv(x)

''' This version for single gpu '''
class feathook():

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.feature_list = []
        self.device_list  = []

    def hook_fn(self, module, input, output):
                
        device = torch.cuda.current_device()

        self.device_list  = device
        self.feature_list = output
        
    def close(self):
        self.hook.remove()
        
