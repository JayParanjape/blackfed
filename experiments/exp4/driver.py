import sys
sys.path.append("../..")

from train_combined import test
from train_combined import train as combined_train
import torch
from model import UNet_Server, UNet_Client
from data_utils import get_data
import yaml


datasets_list = [1, 2, 3]
clients = []
datasets = []
device = sys.argv[1]

for i in range(len(datasets_list)):
    clients.append(UNet_Client(3))
server = UNet_Server(1)

with open('../../data_configs/isic.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

for i in range(len(datasets_list)):
    datasets.append(get_data(config, datasets_list[i]))

    
num_meta_epochs = 10
for i in range(num_meta_epochs):
    if i>0:
        server.load_state_dict(torch.load('./tmp_server.pth'))
        for j in range(len(datasets_list)):            
            clients[j].load_state_dict(torch.load('./tmp_client_'+str(j)+'.pth'))

    combined_train(server, clients, datasets, save_path='./saved_models/', loss_string='bce + dice', device=device)
    torch.cuda.empty_cache()
        

#testing
for j in range(len(datasets_list)):
    server.load_state_dict(torch.load('./tmp_server.pth'))
    clients[j].load_state_dict(torch.load('./saved_models'+'/client_'+str(j)+'_best_val.pth'))
    # clients[j].load_state_dict(torch.load('.'+'/tmp_client_'+str(j)+'.pth'))
    test(server, clients[j], datasets[j], device=device)