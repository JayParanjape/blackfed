import sys
sys.path.append("../..")

from train_2server1client import train, test
import torch
from models.deeplabv3 import DeepLabv3_Client, DeepLabv3_Server
from data_utils import get_data
import yaml
import sys

datasets_list = [1, 2, 3, 4, 5, 6]
clients = []
datasets = []
device = sys.argv[1]

for i in range(len(datasets_list)):
    clients.append(DeepLabv3_Client(64))
server = DeepLabv3_Server(64, 1, use_sigmoid=True)

with open('../../data_configs/polypgen.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

for i in range(len(datasets_list)):
    datasets.append(get_data(config, datasets_list[i]))

    
# num_meta_epochs = 1000
# for i in range(num_meta_epochs):
#     for j in range(len(datasets_list)):
#         if i==0 and j==0:
#             try:
#                 server.load_state_dict(torch.load('./saved_models6_2_dice/tmp_server_bg.pth'))
#             except:
#                 pass
                
#             try:
#                 clients[datasets_list[j]-1].load_state_dict(torch.load('./saved_models6_2_dice/tmp_client_bg'+str(datasets_list[j])+'.pth'))
#             except:
#                 pass

#         print("Training for dataset ", datasets_list[j], " mega epoch ",i)
#         server, clients[datasets_list[j]-1] = train(server, clients[datasets_list[j]-1], datasets[j], datasets_list[j]-1, save_path='./saved_models6_2_dice/'+str(datasets_list[j]), loss_string='dice', device=device, bs=9, lr_client=1e-4, ck=0.0001, tmp_save_path='./saved_models6_dice' )
#         torch.cuda.empty_cache()

#testing
for j in range(len(datasets_list)):
    # server.load_state_dict(torch.load('./saved_models6_dice/tmp_server_bg.pth'))
    server.load_state_dict(torch.load('./saved_models6_dice/'+'server_best_val2_'+str(datasets_list[j]-1)+'.pth'))
    server.eval()
    for k in range(len(datasets_list)):
        # try:
        clients[j].load_state_dict(torch.load('./saved_models6_dice/'+'client_current2_'+str(datasets_list[j]-1)+'.pth'))
        # except:
            # clients[j].load_state_dict(torch.load('./saved_models6_dice/tmp_client_bg'+str(datasets_list[j]-1)+'.pth'))
        clients[j].eval()
        print("client ", datasets_list[j], "dataset ", datasets_list[k])
        test(server, clients[j], datasets[k], device=device)
