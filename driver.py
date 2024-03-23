from train import train
import torch
from model import UNet_Server, UNet_Client
from data_utils import get_data
import yaml

datasets_list = ['refuge', 'isic2018', 'kvasir-seg', 'glas']
clients = []
datasets = []

for i in range(len(datasets_list)):
    clients.append(UNet_Client(3))
server = UNet_Server(1)

for i in range(len(datasets_list)):
    name = datasets_list[i]
    if name == 'refuge':
        with open('data_configs/refuge.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        datasets.append(get_data(config))
    
    elif name == 'isic2018':
        with open('data_configs/isic.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        datasets.append(get_data(config))

    elif name == 'kvasir-seg':
        with open('data_configs/kvasirseg.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        datasets.append(get_data(config))

    elif name == 'glas':
        with open('data_configs/glas.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        datasets.append(get_data(config))

    
num_meta_epochs = 10
for i in range(num_meta_epochs):
    for j in range(len(datasets_list)):
        try:
            server.load_state_dict(torch.load('./tmp_server.pth'))
        except:
            pass
            
        try:
            clients[j].load_state_dict(torch.load('./tmp_client_'+str(j)+'.pth'))
        except:
            pass

        print("Training for dataset ", datasets_list[j], " mega epoch ",i)
        server, clients[j] = train(server, clients[j], datasets[j], j, save_path='./save_models/'+datasets_list[j], loss_string='bce + dice', device='cuda:0' )
        torch.cuda.empty_cache()
        
