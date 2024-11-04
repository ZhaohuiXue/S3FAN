global Dataset  # UP,IN,SV
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self,dim,dropout,inner_dim,step_size):
        super().__init__()
        self.LN = nn.LayerNorm(dim)
        #mlp
        self.mlp = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self,out):
        # FeedForward part
        out1 = self.LN(out)  # Norm
        out2 = self.mlp(out1)  # MLP
        out2 = out2 + out1
        return out2

class MSSA(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,step_size=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.FF = FeedForward(dim,dropout,inner_dim,step_size)

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h = self.heads)
        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)#softmax compute the attention score
        attn = self.dropout(attn)
        out = torch.matmul(attn, w)#attn is attention score using as weight
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = out + x
        #FeedForward part
        out = self.FF(out)
        return out

class Transformer(nn.Module):
    def __init__(self,dim,heads,dim_head,dropout,depth,step_size):
        super().__init__()
        self.attention = MSSA(dim,heads,dim_head,dropout,step_size)
        self.depth = depth


    def forward(self,x1_4):
        for _ in range(self.depth):
            x1_4 = self.attention(x1_4)

        return x1_4
class Feature_Fusion(nn.Module):
    def __init__(self):
        super(Feature_Fusion, self).__init__()
        self.r = nn.Parameter(torch.tensor([0.5,0.5]),requires_grad=True)

    def forward(self,x1,x2):
        x = (self.r[0] * x1+self.r[1] * x2)/(self.r[0]+self.r[1])
        return x

class S3FAN(nn.Module):#（batch，channels，image_size，image_size）
    def __init__(self, *, image_size, num_classes,KS, dim, depth, heads, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.,emb_dropout, model,ista=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        num_pixels = image_height * image_width
        pixel_dim = channels
        self.Pixel_Embedding = nn.Sequential(
            Rearrange('b c h w -> b (h w) c', h = image_height, w = image_width),
            nn.LayerNorm(pixel_dim),
            nn.Linear(pixel_dim, dim),
            nn.LayerNorm(dim),
        )

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.depth = depth
        self.pos_embedding = nn.Parameter(torch.randn(1, num_pixels + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.SpeRes = Spectral_ResNet(10,10,(1,1,7),(1,1,1),(0,0,3),channels,KS)#set padding (0,0,3) so keep output.shape[-1]=channels
        self.SpaRes = Spatial_ResNet(KS,KS,(3,3),(1,1),(1,1))
        self.LN = nn.LayerNorm(dim)
        self.transformer = Transformer(dim,heads,dim_head,dropout,depth,ista)
        self.STBlock1 = Spectral_Transition_Block(channels)
        self.STBlock2 = Spectral_Transition_Block(KS)
        self.GAP = nn.AdaptiveAvgPool1d(32)#GAP as the classifer
        self.sum = Feature_Fusion()
        self.SpeSFA = Spectral_Score_Fusion_Attention(channels)
        self.SpaSFA = Spatial_Score_Fusion_Attention(KS)
        self.classifer = nn.AdaptiveAvgPool1d(num_classes)
    def forward(self, x):
        #SWETM
        x = self.SpeSFA(x)
        x1, x2 = self.SpeRes(x)
          #Spectral Transition Block
        x1_1 = self.STBlock1(x1)
        x2_1 = self.STBlock2(x2)
        # x1_1 is the input of SSIM
        x1_2 = self.Pixel_Embedding(x1_1)
        b, n, _ = x1_2.shape
            #class embedding
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x1_2 = torch.cat((cls_tokens, x1_2), dim=1)
            #position embedding
        x1_2 += self.pos_embedding[:, :(n + 1)]
        x1_2 = self.dropout(x1_2)
            #LayerNorm
        x1_2 = self.LN(x1_2)
            #MSSA+FF
        x1_2 = self.transformer(x1_2)
        # x2_1 is the input of SWEM
        x2_2 = self.SpaSFA(x2_1)
        x2_2 = self.SpaRes(x2_2)
        ############################
        x1_3 = x1_2[:, 0]
        x1_3 = self.to_latent(x1_3)
        x1_3 = x1_3.unsqueeze(0)
        x1_3 = self.GAP(x1_3)
        x1_4 = x1_3.squeeze(0)
        x2_2 = x2_2.view(x2_2.shape[0],-1)
        x2_2 = x2_2.unsqueeze(0)
        x2_3 = self.GAP(x2_2)
        x2_4 = x2_3.squeeze(0)
        #Feature Fusion and Classification
        Y = self.sum(x1_4,x2_4)
        Y = Y.unsqueeze(0)
        Y = self.classifer(Y)
        Y = Y.squeeze(0)

        return Y
class Spectral_Score_Fusion_Attention(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.convSA1_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=3, kernel_size=(1, 1, 7), stride=(1, 1, 1), padding=(0, 0, 3)),
            nn.BatchNorm3d(3),
            nn.ReLU()
        )
        self.convBlock = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=channels//2, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(in_channels=channels // 2, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                      padding=(0, 0, 0)),
            nn.ReLU()
        )
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveMaxPool2d(1)
        self.sum = Spe_Score_Fusion()
        self.sigmoid = nn.Sigmoid()

    def forward(self,img):
        #torchcat raw img and the feature after conv
        x1 = self.convSA1_1(img)
        x2 = torch.cat([x1,img],dim=1)
        x2 = torch.sum(x2,dim=1).unsqueeze(1)
        #x2 = x1 + img
        x3 = x2.view(x2.size(0),-1,x2.size(2),x2.size(3))
        #two pooling strategy
        x3_1 = self.pool1(x3).unsqueeze(1)#avgpool
        x3_2 = self.pool2(x3).unsqueeze(1)#maxpool
        x3_1 = x3_1.view(x3_1.size(0), x3_1.size(1), x3_1.size(3), x3_1.size(4), x3_1.size(2))
        x3_2 = x3_2.view(x3_2.size(0), x3_2.size(1), x3_2.size(3), x3_2.size(4), x3_2.size(2))
        #convBlock extracts spectral feature
        x4_1 = self.convBlock(x3_1)
        x4_2 = self.convBlock(x3_2)
        sa1 = x4_1 + x4_2
        #compute attention score
        sa1_1 = self.sigmoid(sa1)
        sa1_2 = self.sigmoid(x4_1)
        sa1_3 = self.sigmoid(x4_2)
        sa1 = self.sum(sa1_1,sa1_2,sa1_3)
        sa1 = sa1 * img
        return sa1
class Spectral_ResNet(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,channels,KS):
        super().__init__()
        self.conv1D_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=10, kernel_size=(1, 1, 7), stride=(1, 1, 1), padding=(0, 0, 3)),
            nn.BatchNorm3d(10),
            nn.ReLU()
        )
        self.conv1D_2 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        self.conv1D_3 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        self.conv1D_4 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm3d(out_channels),
        )
        self.conv1D_5_1 = nn.Sequential(
            nn.Conv3d(in_channels=10, out_channels=channels, kernel_size=(1, 1, channels), stride=(1, 1, 1),
                      padding=(0, 0, 0)),
            nn.BatchNorm3d(channels),
            nn.ReLU()
        )
        self.conv1D_5_2 = nn.Sequential(
            nn.Conv3d(in_channels=10, out_channels=KS, kernel_size=(1, 1, channels), stride=(1, 1, 1),
                      padding=(0, 0, 0)),
            nn.BatchNorm3d(KS),
            nn.ReLU()
        )
    def forward(self,x):
        x1 = self.conv1D_1(x)
        x2 = self.conv1D_2(x1)
        x3 = self.conv1D_3(x2)
        x4 = self.conv1D_4(x3)
        x5 = F.relu(x1+x4)
        x5_1 = self.conv1D_5_1(x5)
        x5_2 = self.conv1D_5_2(x5)
        return x5_1, x5_2 #Residual connection

class Spatial_ResNet(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super().__init__()
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2D_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2D_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2D_4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self,x):
        x1 = self.conv2D_1(x)
        x2 = self.conv2D_2(x1)
        x3 = self.conv2D_3(x2)
        x4 = self.conv2D_4(x3)
        return F.relu(x1+x4)
class Spe_Score_Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.r = nn.Parameter(torch.tensor([1.0,1.0,1.0]),requires_grad=True)

    def forward(self,x1,x2,x3):
        x = (self.r[0] * x1 + self.r[1] * x2 + self.r[2] * x3) / (self.r[0] + self.r[1] + self.r[2])
        return x

class Spa_Score_Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.r = nn.Parameter(torch.tensor([1.0,1.0,1.0]),requires_grad=True)

    def forward(self,x1,x2,x3):
        x = (self.r[0] * x1 + self.r[1] * x2 + self.r[2] * x3) / (self.r[0] + self.r[1] + self.r[2])
        return x

class Spatial_Score_Fusion_Attention(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool1d(1)
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        self.convSA2_1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.convBlock = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=channels // 2, kernel_size=(3, 3, 1), stride=(1, 1, 1),
                      padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(in_channels=channels // 2, out_channels=1, kernel_size=(3, 3, 1), stride=(1, 1, 1),
                      padding=(1, 1, 0)),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()
        self.sum = Spa_Score_Fusion()

    def forward(self,x):
        x1 = self.convSA2_1(x)
        x2 = x + x1
        x2 = x2.view(x2.size(0),-1,x2.size(1))
        x3 = self.pool1(x2)
        x3_= self.pool2(x2)
        x3 = x3.view(x.size(0), x.size(2), x.size(3), -1).unsqueeze(1)
        x3_ = x3_.view(x.size(0), x.size(2), x.size(3), -1).unsqueeze(1)
        x4 = self.convBlock(x3).squeeze(1)
        x4_ = self.convBlock(x3_).squeeze(1)
        x4 = x4.view(x4.size(0), x4.size(3), x4.size(1), x4.size(2))
        x4_ = x4_.view(x4_.size(0), x4_.size(3), x4_.size(1), x4_.size(2))
        sa = x4 + x4_
        score1 = self.sigmoid(sa)
        score2 = self.sigmoid(x4)
        score3 = self.sigmoid(x4_)
        score = self.sum(score1, score2, score3)
        x5 = score * x
        return x5
class Spectral_Transition_Block(nn.Module):
    def __init__(self, channel):
        super(Spectral_Transition_Block, self).__init__()
        k = 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid()
        )
        self.sum = vector_fusion()

    def forward(self, x):
        x = x.view(x.size()[0], -1, x.size()[2], x.size()[3])
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        v1 = self.conv(y.squeeze(-1).transpose(-1,-2))
        v1 = v1.transpose(-1,-2).unsqueeze(-1)
        v1 = self.sigmoid(v1)
        v2 = self.linear(y.view(b, c))
        v2 = v2.view(b, c, 1, 1)
        map1 = x * v1.expand_as(x)
        map2 = x * v2.expand_as(x)
        v_sum = self.sum(v1, v2)
        map3 = x * v_sum.expand_as(x)
        map = map1 + map2 + map3
        return map
class vector_fusion(nn.Module):#Adaptive weight
    def __init__(self):
        super(vector_fusion, self).__init__()
        self.r = nn.Parameter(torch.tensor([1.0,1.0]),requires_grad=True)#

    def forward(self,x1,x2):
        x = (self.r[0] * x1+self.r[1] * x2)/(self.r[0]+self.r[1])
        return x
