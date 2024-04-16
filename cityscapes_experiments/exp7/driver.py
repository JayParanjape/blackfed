import sys
sys.path.append("../..")

from train_white import train, test
import torch
from model import UNet_Server, UNet_Client
from data_utils import get_data
import yaml


datasets_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
clients = []
datasets = []
device = sys.argv[1]

for i in range(len(datasets_list)):
    clients.append(UNet_Client(3))
server = UNet_Server(n_classes=19)

with open('../../data_configs/cityscapes.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

for i in range(len(datasets_list)):
    datasets.append(get_data(config, datasets_list[i]))

    
num_meta_epochs = 1000
for i in range(num_meta_epochs):
    for j in range(len(datasets_list)):
        if i>0:
            server.load_state_dict(torch.load('./tmp_server.pth'))
                
            clients[j].load_state_dict(torch.load('./tmp_client_'+str(j)+'.pth'))

        print("Training for dataset ", datasets_list[j], " mega epoch ",i)
        server, clients[j] = train(server, clients[j], datasets[j], j, save_path='./saved_models3_dice/'+str(datasets_list[j]), loss_string='dice', device=device )
        torch.cuda.empty_cache()

#testing
for j in range(len(datasets_list)):
    server.load_state_dict(torch.load('./tmp_server.pth'))
    # clients[j].load_state_dict(torch.load('./saved_models/'+str(datasets_list[j])+'/client_best_val.pth'))
    clients[j].load_state_dict(torch.load('.'+'/tmp_client_'+str(j)+'.pth'))
    test(server, clients[j], datasets[j], device=device)
