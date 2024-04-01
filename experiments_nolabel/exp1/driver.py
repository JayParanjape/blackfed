import sys
sys.path.append("../..")

from train import train, test
from train_combined import train as combined_train
import torch
from model import NoLabel_UNet_Client, NoLabel_UNet_Server
from data_utils import get_data
import yaml


datasets_list = [1, 2, 3]
clients = []
datasets = []
device = sys.argv[1]

for i in range(len(datasets_list)):
    clients.append(NoLabel_UNet_Client(start_channels=64, num_layers=1, num_classes=1))
server = NoLabel_UNet_Server(128,3)

with open('../../data_configs/isic.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

for i in range(len(datasets_list)):
    datasets.append(get_data(config, datasets_list[i]))

    
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
        server, clients[j] = no_label_train(server, clients[j], datasets[j], save_path='./saved_models/', loss_string='bce + dice', device=device)
        torch.cuda.empty_cache()
        

#testing
for j in range(len(datasets_list)):
    server.load_state_dict(torch.load('./tmp_server.pth'))
    clients[j].load_state_dict(torch.load('./saved_models'+'/client_'+str(j)+'_best_val.pth'))
    test(server, clients[j], datasets[j], device='cuda:0')