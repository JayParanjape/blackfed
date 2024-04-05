import sys
sys.path.append("../..")

from train import train, test
import torch
from models.unext import UNext_Client, UNext_Server
from data_utils import get_data
import yaml
import sys

datasets_list = [1, 2, 3, 4, 5, 6, 7,8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
clients = []
datasets = []
device = sys.argv[1]

for i in range(len(datasets_list)):
    clients.append(UNext_Client(64))
server = UNext_Server(19, 64)

with open('../../data_configs/cityscapes.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

for i in range(len(datasets_list)):
    datasets.append(get_data(config, datasets_list[i]))

    
# num_meta_epochs = 10
# for i in range(num_meta_epochs):
#     for j in range(len(datasets_list)):
#         try:
#             server.load_state_dict(torch.load('./tmp_server.pth'))
#         except:
#             pass
            
#         try:
#             clients[j].load_state_dict(torch.load('./tmp_client_'+str(j)+'.pth'))
#         except:
#             pass

#         print("Training for dataset ", datasets_list[j], " mega epoch ",i)
#         server, clients[j] = train(server, clients[j], datasets[j], j, save_path='./saved_models/'+str(datasets_list[j]), loss_string='bce + dice', device=device )
#         torch.cuda.empty_cache()

#testing
for j in range(len(datasets_list)):
    server.load_state_dict(torch.load('./tmp_server.pth'))
    clients[j].load_state_dict(torch.load('./saved_models/'+str(datasets_list[j])+'/client_best_val.pth'))
    test(server, clients[j], datasets[j], device=device)
