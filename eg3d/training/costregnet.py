import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from inplace_abn import InPlaceABN

from ipdb import set_trace as st

#----------------------------------------
class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        self.act = nn.LeakyReLU()
        # self.bn = nn.ReLU()
        # self.conv.apply(conv3d_weights_init)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CostRegNet_Deeper(nn.Module): # 256^3 -> 8^3; 128^3 -> 4^3
    def __init__(self, in_channels, out_dim=8, norm_act=InPlaceABN):
        super(CostRegNet_Deeper, self).__init__()
        
        self.conv0 = ConvBnReLU3D(in_channels, out_dim, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(out_dim, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv51 = ConvBnReLU3D(64, 64, stride=2, norm_act=norm_act)
        self.conv61 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv52 = ConvBnReLU3D(64, 64, stride=2, norm_act=norm_act)
        self.conv62 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv27 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(64))
        
        self.conv17 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(64))

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        # self.conv11 = nn.Sequential(
        #     nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
        #                        stride=2, bias=False),
        #     norm_act(8))
        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, out_dim, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(out_dim))

        # self.conv12 = nn.Conv3d(8, 8, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        conv0 = self.conv0(x)

        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        # if self.conv3.bn.weight.grad != None:
        # st()

        # x = self.conv6(self.conv5(conv4))
        conv6 = self.conv6(self.conv5(conv4))

        conv61 = self.conv61(self.conv51(conv6))
        conv62 = self.conv62(self.conv52(conv61))
        # print("CostRegNetDeeper bottleneck:", conv62.shape): # 256^3 -> 8^3; 128^3 -> 4^3
        x = conv61 + self.conv27(conv62)
        x = conv6 + self.conv17(x)

        x = conv4 + self.conv7(x)
        # del conv4
        x = conv2 + self.conv9(x)
        # x = conv2 + self.conv9(conv4)
        del conv2, conv4
        x = conv0 + self.conv11(x)
        del conv0
        # x = self.conv12(x)
        return x


class PcWsUnet(nn.Module): # 256^3 -> 8^3; 128^3 -> 4^3
    def __init__(self, in_channels, in_resolution, block_resolutions, out_dim=8, norm_act=InPlaceABN):
        super(PcWsUnet, self).__init__()
        self.block_resolutions = block_resolutions
        self.conv0 = ConvBnReLU3D(in_channels, out_dim, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(out_dim, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv51 = ConvBnReLU3D(64, 64, stride=2, norm_act=norm_act)
        self.conv61 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv52 = ConvBnReLU3D(64, 64, stride=2, norm_act=norm_act)
        self.conv62 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv27 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(64))
        
        self.conv17 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(64))

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        # self.conv11 = nn.Sequential(
        #     nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
        #                        stride=2, bias=False),
        #     norm_act(8))
        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, out_dim, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(out_dim))

        # self.conv12 = nn.Conv3d(8, 8, 3, stride=1, padding=1, bias=True)

        ## construct FC layers
        max_res, min_res = max(self.block_resolutions), min(self.block_resolutions)
        ## in_res --> outdim
        ## inres//2 --> 16
        self.max_res = min(max_res, in_resolution, 32)
        res = self.max_res
        channels = {128:8, 64:16, 32:32, 16:64, 8:64, 4:64}
        while (res>= min_res):
            ch = channels.get(res)
            layer = nn.Linear((res**3)*ch, ch*3)
            setattr(self, f'fc{res}', layer)
            # print(res, ch)
            res = res//2


    def forward(self, x):
        res_feature = {}

        conv0 = self.conv0(x)
        # res_feature[conv0.shape[-1]]=conv0

        conv2 = self.conv2(self.conv1(conv0))
        # res_feature[conv2.shape[-1]]=conv2

        conv4 = self.conv4(self.conv3(conv2))
        # res_feature[conv4.shape[-1]]=conv4
        # if self.conv3.bn.weight.grad != None:

        # x = self.conv6(self.conv5(conv4))
        conv6 = self.conv6(self.conv5(conv4))
        # res_feature[conv6.shape[-1]]=conv6

        conv61 = self.conv61(self.conv51(conv6))
        # res_feature[conv61.shape[-1]]=conv61

        conv62 = self.conv62(self.conv52(conv61))
        # res_feature[conv62.shape[-1]]=conv62

        # print("CostRegNetDeeper bottleneck:", conv62.shape): # 256^3 -> 8^3; 128^3 -> 4^3
        x = conv61 + self.conv27(conv62)
        try:
            res = x.shape[-1]
            layer = getattr(self, f'fc{res}')
            res_feature[x.shape[-1]]=layer(x.flatten(1))
        except:
            pass

        x = conv6 + self.conv17(x)
        try:
            res = x.shape[-1]
            layer = getattr(self, f'fc{res}')
            res_feature[x.shape[-1]]=layer(x.flatten(1)) 
        except:
            pass
        # res_feature[x.shape[-1]]=x

        x = conv4 + self.conv7(x)
        # res_feature[x.shape[-1]]=x
        try:
            res = x.shape[-1]
            layer = getattr(self, f'fc{res}')
            res_feature[x.shape[-1]]=layer(x.flatten(1)) 
        except:
            pass

        # del conv4
        x = conv2 + self.conv9(x)
        # res_feature[x.shape[-1]]=x
        try:
            res = x.shape[-1]
            layer = getattr(self, f'fc{res}')
            res_feature[x.shape[-1]]=layer(x.flatten(1)) 
        except:
            pass

        # x = conv2 + self.conv9(conv4)
        del conv2, conv4
        x = conv0 + self.conv11(x)
        # if x.shape[-1] <= self.max_res:
        #     res_feature[x.shape[-1]]=x
        try:
            res = x.shape[-1]
            layer = getattr(self, f'fc{res}')
            res_feature[x.shape[-1]]=layer(x.flatten(1)) 
        except:
            pass
        
        del conv0
        # x = self.conv12(x)

        ## use FC layer to project 3D volumes to 1d ws feature
        # res_ws={}
        # for res, vol in res_feature.items():
        #     layer = getattr(self, f'fc{res}')
        #     ws = layer(vol.flatten(1))
        #     res_feature.update({res:ws})

        return res_feature


class Synthesis3DUnet(nn.Module): # 256^3 -> 8^3; 128^3 -> 4^3
    def __init__(self, 
            in_channels, 
            out_dim=8, 
            use_noise=False,
            noise_strength = 0.5,
            ws_channel=512,
            affine_act='relu', #### ???? FIXME: is this a good activation 
            norm_act=InPlaceABN):

        super(Synthesis3DUnet, self).__init__()

        self.use_noise = use_noise
        # noise_strength = 0.5
        self.noise_strength = noise_strength

        self.conv0 = ConvBnReLU3D(in_channels, out_dim, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(out_dim, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv51 = ConvBnReLU3D(64, 64, stride=2, norm_act=norm_act)
        self.conv61 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv52 = ConvBnReLU3D(64, 64, stride=2, norm_act=norm_act)
        self.conv62 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv27 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(64))
        self.affine27 = nn.Sequential(
                        nn.Linear(ws_channel, 64),
                        nn.ReLU()
                    )
        
        
        self.conv17 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(64))
        self.affine17 = nn.Sequential(
                        nn.Linear(ws_channel, 64),
                        nn.ReLU()
                    )

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))
        self.affine7 = nn.Sequential(
                        nn.Linear(ws_channel, 32),
                        nn.ReLU()
                    )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))
        self.affine9 = nn.Sequential(
                        nn.Linear(ws_channel, 16),
                        nn.ReLU()
                    )

        # self.conv11 = nn.Sequential(
        #     nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
        #                        stride=2, bias=False),
        #     norm_act(8))
        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, out_dim, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(out_dim))
        self.affine11 = nn.Sequential(
                        nn.Linear(ws_channel, out_dim),
                        nn.ReLU()
                    )
        

        # self.conv12 = nn.Conv3d(8, 8, 3, stride=1, padding=1, bias=True)

    def forward(self, x, ws):
        conv0 = self.conv0(x)
        

        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        # if self.conv3.bn.weight.grad != None:
        # st()

        # x = self.conv6(self.conv5(conv4))
        conv6 = self.conv6(self.conv5(conv4))

        conv61 = self.conv61(self.conv51(conv6))
        conv62 = self.conv62(self.conv52(conv61))
        # print("CostRegNetDeeper bottleneck:", conv62.shape) # 256^3 -> 8^3; 128^3 -> 4^3

        ### below is upconv process: add noises and latent
        w_idx=0


        x = conv61 + self.conv27(conv62)
        if self.use_noise:
            noise = torch.rand(x.shape, device=x.device, dtype=torch.float32)
            noise = noise*self.noise_strength
            
            x = x.add_(noise.to(x.dtype))
        style = self.affine27(ws.narrow(1, w_idx, 1)).permute(0,2,1)
        w_idx += 1
        # st()
        assert x.shape[:2] == style.shape[:2]
        B, C = x.shape[:2]
        style = style.reshape(B,C,1,1,1) # extend to 3D 
        x = x*style
        # st() # x.shape


        x = conv6 + self.conv17(x)
        if self.use_noise:
            noise = torch.rand(x.shape, device=x.device)
            noise = noise*self.noise_strength
            x = x.add_(noise.to(x.dtype))
        style = self.affine17(ws.narrow(1, w_idx, 1)).permute(0,2,1)
        w_idx += 1
        assert x.shape[:2] == style.shape[:2]
        B, C = x.shape[:2]
        style = style.reshape(B,C,1,1,1) # extend to 3D 
        x = x*style


        x = conv4 + self.conv7(x)
        if self.use_noise:
            noise = torch.rand(x.shape, device=x.device)
            noise = noise*self.noise_strength
            x = x.add_(noise.to(x.dtype))
        style = self.affine7(ws.narrow(1, w_idx, 1)).permute(0,2,1)
        w_idx += 1
        assert x.shape[:2] == style.shape[:2]
        B, C = x.shape[:2]
        style = style.reshape(B,C,1,1,1) # extend to 3D 
        x = x*style
        # del conv4


        x = conv2 + self.conv9(x)
        if self.use_noise:
            noise = torch.rand(x.shape, device=x.device)
            noise = noise*self.noise_strength
            x = x.add_(noise.to(x.dtype))
        style = self.affine9(ws.narrow(1, w_idx, 1)).permute(0,2,1)
        w_idx += 1
        assert x.shape[:2] == style.shape[:2]
        B, C = x.shape[:2]
        style = style.reshape(B,C,1,1,1) # extend to 3D 
        x = x*style

        del conv2, conv4


        x = conv0 + self.conv11(x)
        if self.use_noise:
            noise = torch.rand(x.shape, device=x.device)
            noise = noise*self.noise_strength
            x = x.add_(noise.to(x.dtype))
        style = self.affine11(ws.narrow(1, w_idx, 1)).permute(0,2,1)
        w_idx += 1
        assert x.shape[:2] == style.shape[:2]
        B, C = x.shape[:2]
        style = style.reshape(B,C,1,1,1) # extend to 3D 
        x = x*style

        del conv0
        
        # print(f"Totally used up to {w_idx} ws in synthesis3DUnet") ## currently used only 5
        
        return x