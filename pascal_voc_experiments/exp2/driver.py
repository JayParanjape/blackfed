import sys
sys.path.append("../..")

from train import train, test
import torch
from models.unext import UNext_Client, UNext_Server
from data_utils import get_data
import yaml
import sys

datasets_list = [1, 2, 3, 4, 5, 6, 7,8, 9, 10]
clients = []
datasets = []
device = sys.argv[1]
lr_server = sys.argv[2]
lr_client = sys.argv[3]

for i in range(len(datasets_list)):
    clients.append(UNext_Client(64))
server = UNext_Server(20, 64)

with open('../../data_configs/pascal_voc.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

for i in range(len(datasets_list)):
    datasets.append(get_data(config, datasets_list[i]))

    
num_meta_epochs = 1000
for i in range(num_meta_epochs):
    for j in range(len(datasets_list)):
        if i>0 and j>0:
            try:
                server.load_state_dict(torch.load('./tmp_server.pth'))
            except:
                pass
                
            try:
                clients[j].load_state_dict(torch.load('./tmp_client_'+str(j)+'.pth'))
            except:
                pass

        print("Training for dataset ", datasets_list[j], " mega epoch ",i)
        server, clients[j] = train(server, clients[j], datasets[j], j, save_path='./saved_models3_dice_server'+str(lr_server)+'_client'+str(lr_client)+'/'+str(datasets_list[j]), loss_string='dice', device=device, bs=32, lr_server=float(lr_server), lr_client=float(lr_client))
        torch.cuda.empty_cache()

# #testing
# for j in range(len(datasets_list)):
#     server.load_state_dict(torch.load('./tmp_server.pth'))
#     clients[j].load_state_dict(torch.load('./saved_models3_dice/'+str(datasets_list[j])+'/client_best_val.pth'))
#     test(server, clients[j], datasets[j], device=device)

# testing nxn
# for j in range(len(datasets_list)):
#     server.load_state_dict(torch.load('./tmp_server.pth'))
#     clients[j].load_state_dict(torch.load('./saved_models3_dice/'+str(datasets_list[j])+'/client_best_val.pth'))
#     print('TESTING FOR MODEL FROM CENTER ', j+1)
#     for k in range(len(datasets_list)):
#         test(server, clients[j], datasets[k], device=device)