import sys
sys.path.append("../mmsegmentation/configs")
from mmseg.apis import init_model
from torch import nn
import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np

class SegFormer(nn.Module):
    def __init__(self, n_channels, n_classes, config_path='../mmsegmentation/configs/segformer/segformer_blackfed_camvid.py', device='cpu'):
        super(SegFormer, self).__init__()
        self.client = SegFormer_Client(n_channels, device=device)
        self.server = SegFormer_Server(n_channels, n_classes, config_path, device=device)

    def forward(self, x):
        out = self.client(x)
        out = self.server(out)
        return out

class SegFormer_Client(nn.Module):
    def __init__(self, n_channels, device='cpu'):
        super(SegFormer_Client, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(3, n_channels).to(device)

    def forward(self, x):
        out = self.inc(x)
        return out

class SegFormer_Server(nn.Module):
    def __init__(self, n_channels, n_classes, config_path='../mmsegmentation/configs/segformer/segformer_blackfed_camvid.py', device='cpu'):
        super(SegFormer_Server, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.config_path = config_path
        # self.network = init_model(self.config_path, checkpoint='/mnt/store/jparanj1/mmseg_segformer.b1.1024x1024.city.160k.pth')
        self.network = init_model(self.config_path).to(device)
        for p in self.network.parameters():
            # print(p.shape)
            try:
                torch.nn.init.xavier_uniform(p)
            except:
                torch.nn.init.uniform_(p)
        
        self.network.data_preprocessor = nn.Identity()
        self.network.backbone.layers[0][0].projection = nn.Identity()
        

    def forward(self, x):
        batch_img_metas = [{'img_shape':(x.shape[-2],x.shape[-1])}]*x.shape[0]
        out = self.network.encode_decode(x, batch_img_metas)
        out = torch.nn.functional.softmax(out, dim=1)

        return out
    
    def forward_backbone(self,x):
        return self.network.backbone(x)
    
    def forward_backbone1(self,x):
        for p in (self.network.backbone.layers[0][2]).parameters():
            print(p)
        1/0
        out = self.network.data_preprocessor(x)
        tmp = out.shape
        out, sh = self.network.backbone.layers[0][0](out)
        print("But here its bloody fine and shape of input was: ", out.shape)
        print(out)
        out2 = self.network.backbone.layers[0][1][0](out,sh)
        out2 = self.network.backbone.layers[0][1][1](out2,sh)
        print(out2)
        print(out2.shape)
        print("Now the fuck up")
        torch.save(out2, 'f_out.pt')
        print("Fucker: ",self.network.backbone.layers[0][2])
        print("Fucker state dict: ", self.network.backbone.layers[0][2].state_dict())
        torch.save(self.network.backbone.layers[0][2].state_dict(), 'f_layer_sd.pth')
        out = self.network.backbone.layers[0][2](out2)
        contradiction = torch.load('f_out.pt')
        print("Contradiction: ", self.network.backbone.layers[0][2](contradiction))
        print(out)
        return out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=7, padding=3, stride = 1),
        )

    def forward(self, x):
        return self.double_conv(x)

def convert_mit(ckpt):
    new_ckpt = OrderedDict()
    # Process the concat between q linear weights and kv linear weights
    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        # patch embedding conversion
        elif k.startswith('patch_embed'):
            stage_i = int(k.split('.')[0].replace('patch_embed', ''))
            new_k = k.replace(f'patch_embed{stage_i}', f'layers.{stage_i-1}.0')
            new_v = v
            if 'proj.' in new_k:
                new_k = new_k.replace('proj.', 'projection.')
        # transformer encoder layer conversion
        elif k.startswith('block'):
            stage_i = int(k.split('.')[0].replace('block', ''))
            new_k = k.replace(f'block{stage_i}', f'layers.{stage_i-1}.1')
            new_v = v
            if 'attn.q.' in new_k:
                sub_item_k = k.replace('q.', 'kv.')
                new_k = new_k.replace('q.', 'attn.in_proj_')
                new_v = torch.cat([v, ckpt[sub_item_k]], dim=0)
            elif 'attn.kv.' in new_k:
                continue
            elif 'attn.proj.' in new_k:
                new_k = new_k.replace('proj.', 'attn.out_proj.')
            elif 'attn.sr.' in new_k:
                new_k = new_k.replace('sr.', 'sr.')
            elif 'mlp.' in new_k:
                string = f'{new_k}-'
                new_k = new_k.replace('mlp.', 'ffn.layers.')
                if 'fc1.weight' in new_k or 'fc2.weight' in new_k:
                    new_v = v.reshape((*v.shape, 1, 1))
                new_k = new_k.replace('fc1.', '0.')
                new_k = new_k.replace('dwconv.dwconv.', '1.')
                new_k = new_k.replace('fc2.', '4.')
                string += f'{new_k} {v.shape}-{new_v.shape}'
        # norm layer conversion
        elif k.startswith('norm'):
            stage_i = int(k.split('.')[0].replace('norm', ''))
            new_k = k.replace(f'norm{stage_i}', f'layers.{stage_i-1}.2')
            new_v = v
        else:
            new_k = k
            new_v = v
        new_ckpt[new_k] = new_v
    return new_ckpt

def get_gflops(model, input_size):
    from ptflops import get_model_complexity_info
    import re
    macs, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=False, verbose=False)
    flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
    flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
    print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
    print('Number of parameters: {:<8}'.format(params))

# if __name__ == '__main__':    
#     client = SegFormer_Client(n_channels=64)
#     server = SegFormer_Server(n_channels=64, n_classes=32)
#     super_model = SegFormer(n_channels=64, n_classes=32)
#     dummy = torch.ones((1,3,256,256))
#     client_out = client(dummy)
#     print('Client GFLOPS: ')
#     print(tuple(dummy.shape))
#     get_gflops(client, tuple(dummy.shape[1:]))
#     # print('Server GFLOPS: ')
#     # get_gflops(server, tuple(client_out.shape[1:]))
#     print('Super Model GFLOPS: ')
#     get_gflops(super_model, tuple(dummy.shape[1:]))
#     server_out = server(client_out)

#     print(client_out.shape)
#     print(server_out.shape)
    # print(server)

if __name__=='__main__':
    from deepspeed.profiling.flops_profiler import FlopsProfiler
    # config_path = "../mmsegmentation/configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py"
    # model = init_model(config_path)
    # model.backbone.conv1 = nn.Identity()
    # print(model)

    client = SegFormer_Client(n_channels=64, device='cuda:3')
    server = SegFormer_Server(n_channels=64, n_classes=32, device='cuda:3')
    client_prof = FlopsProfiler(client, None)
    server_prof = FlopsProfiler(server, None)
    client.train()
    server.train()
    
    # super_model = SegFormer(n_channels=64, n_classes=32, config_path='../mmsegmentation/configs/segformer/segformer_blackfed_camvid.py')
    # super_model.train()
    # print(server.network.backbone)
    # dummy = torch.ones((8,3,256,256))
    dummy = torch.load('/mnt/store/jparanj1/blackfed/tmp3.pt')
    dummy = dummy.float().unsqueeze(0)
    dummy = dummy.to('cuda:3')
    
    print("dummy shape: ", dummy.shape)
    with torch.no_grad():
        client_prof.start_profile()
        client_out = client(dummy)
        client_prof.stop_profile()
        client_flops = client_prof.get_total_flops()
        client_prof.end_profile()
        print(client_out.shape)
        print("Client flops: ", client_flops/(10**9))

        # print(client_out[0,:,100,255])
        # print(client_out[0,:,50,355])
        # print(client_out[0,:,150,155])

        server = server.to('cuda:3')
        server_prof.start_profile()
        server_out = server(client_out)
        server_prof.stop_profile()
        server_flops = server_prof.get_total_flops()
        server_prof.end_profile()

        print(server_out.shape)
        print(("Server Flops: ", server_flops/(10**9)))
        print(server_out[0,:,100,255])
        print(server_out[0,:,50,355])
        print(server_out[0,:,150,155])
        
        
        # server_bck = server.forward_backbone(client_out)
        # print(server_bck[0][0,:,100,255])
        # print(server_bck[0][0,:,50,355])
        # print(server_bck[0][0,:,150,155])

        # for p in (server.network.backbone.layers[0][2]).parameters():
        #     print(p)
        # 1/0

        #save pred
        save_array = (server_out[0,0,:,:].cpu().numpy())
        plt.imshow(save_array>=0.5,cmap='gray')
        plt.savefig('../tmp5.png')


