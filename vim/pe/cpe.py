import torch
import torch.nn as nn

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, use_dilated=False):
        super(PosCNN, self).__init__()
        if not use_dilated:
            self.proj = nn.Sequential(
                # nn.Conv2d(in_chans, embed_dim, 3, 1, padding = "same", bias=True, groups=embed_dim), # per depthwise conv
                nn.Conv2d(in_chans, embed_dim, 5, 1, padding = "same", bias=True, groups=embed_dim), # normal conv
            )
        else:
            self.proj = nn.Sequential(
                                    nn.Conv2d(in_chans, embed_dim, 3, 1, padding = "same", bias=True, groups=embed_dim), # per depthwise conv
                                    # nn.Conv2d(in_chans, embed_dim, 3, 1, padding = "same", bias=True, groups=1), # normal conv
                                    nn.Conv2d(in_chans, embed_dim, 3, 1, padding = "same", bias=True, groups=embed_dim, dilation=3),
                                    nn.Conv2d(in_chans, embed_dim, 3, 1, padding = "same", bias=True, groups=embed_dim, dilation=5),
                                    nn.Conv2d(in_chans, embed_dim, 3, 1, padding = "same", bias=True, groups=embed_dim, dilation=7), # new
                                    # nn.Conv2d(in_chans, embed_dim, 3, 1, padding = "same", bias=True, groups=embed_dim, dilation=9), # new
                                    )

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


    
    
class AdaInPosCNN(nn.Module): # could use dilate conv for better consistency
    def __init__(self, in_chans, embed_dim=768, use_dilated=False):
        super(AdaInPosCNN, self).__init__()
        if not use_dilated:
            self.proj = nn.Sequential(
                # nn.Conv2d(in_chans, embed_dim, 3, 1, padding = "same", bias=True, groups=embed_dim), # per depthwise conv
                nn.Conv2d(in_chans, embed_dim, 5, 1, padding = "same", bias=True, groups=embed_dim), # normal conv
            )
        else:
            self.proj = nn.Sequential(
                                    nn.Conv2d(in_chans, embed_dim, 3, 1, padding = "same", bias=True, groups=embed_dim), # per depthwise conv
                                    # nn.Conv2d(in_chans, embed_dim, 3, 1, padding = "same", bias=True, groups=1), # normal conv
                                    nn.Conv2d(in_chans, embed_dim, 3, 1, padding = "same", bias=True, groups=embed_dim, dilation=3),
                                    nn.Conv2d(in_chans, embed_dim, 3, 1, padding = "same", bias=True, groups=embed_dim, dilation=5),
                                    nn.Conv2d(in_chans, embed_dim, 3, 1, padding = "same", bias=True, groups=embed_dim, dilation=7), # new
                                    # nn.Conv2d(in_chans, embed_dim, 3, 1, padding = "same", bias=True, groups=embed_dim, dilation=9), # new
                                    )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim, bias=True))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, c, H, W):
        B, N, C = x.shape
        feat_token = x
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat # add cnn pos to feat
        x = x.flatten(2).transpose(1, 2)
        x = modulate(self.norm(x), shift, scale) # add style to feat
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)] 
