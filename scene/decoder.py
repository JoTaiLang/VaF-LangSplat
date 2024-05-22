import torch
import torch.nn as nn

# fc decoder

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        hidden_dims = [16, 32, 64, 128, 256, 512]

        decoder_layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(3, hidden_dims[i]))
            else:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        
        self.decoder = nn.ModuleList(decoder_layers)
    
    def forward(self, x):
        for m in self.decoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x


# conv decoder
'''
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        #hidden_dims = [16, 32, 64, 128, 256, 512]
        hidden_dims = [16, 64, 512]

        decoder_layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.ConvTranspose2d(3, hidden_dims[0], 
                               kernel_size=7, 
                               stride=1, 
                               padding=3))
            elif i == 1:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.ConvTranspose2d(hidden_dims[i-1], hidden_dims[i], 
                               kernel_size=5, 
                               stride=1, 
                               padding=2))
            else:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.ConvTranspose2d(hidden_dims[i-1], hidden_dims[i], 
                               kernel_size=3, 
                               stride=1, 
                               padding=1))
        
        self.decoder = nn.ModuleList(decoder_layers)
    
    def forward(self, x):
        for m in self.decoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x
'''

