import sys
sys.path.append("../..")

from train_rev import train, test
import torch
from models.deeplabv3 import DeepLabv3_Client, DeepLabv3_Server
from data_utils import get_data
import yaml
import sys

datasets_list = [1, 2, 3, 4]
clients = []
datasets = []
device = sys.argv[1]

for i in range(len(datasets_list)):
    clients.append(DeepLabv3_Client(64))
server = DeepLabv3_Server(64, 32, use_sigmoid=False)

with open('../../data_configs/camvid.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

for i in range(len(datasets_list)):
    datasets.append(get_data(config, datasets_list[i]))

    
# num_meta_epochs = 1000
# lr_client=1e-4
# ck=0.001
# for i in range(num_meta_epochs):
#     for j in range(len(datasets_list)):
#         if i==0 and j==0:
#             try:
#                 server.load_state_dict(torch.load('./saved_models_server31client10_rev/tmp_server_bg.pth'))
#             except:
#                 pass
                
#             try:
#                 clients[datasets_list[j]-1].load_state_dict(torch.load('./saved_models_server31client10_rev/tmp_client_bg'+str(datasets_list[j])+'.pth'))
#             except:
#                 pass

#         print("Training for dataset ", datasets_list[j], " mega epoch ",i)
#         # if i>0 and i%2==0:
#         #     ck = ck/2
#         server, clients[datasets_list[j]-1] = train(server, clients[datasets_list[j]-1], datasets[j], datasets_list[j]-1, save_path='./saved_models_server31client10_rev/'+str(datasets_list[j]), loss_string='dice', device=device, bs=9, lr_client=lr_client, ck=ck, tmp_save_path='./saved_models_server31client10_rev', num_epochs_client=10, num_epochs_server=31)

#testing
for j in range(len(datasets_list)):
    # server.load_state_dict(torch.load('./safe/saved_models_server31client10_rev/tmp_server_bg.pth'))
    server.load_state_dict(torch.load('./safe/saved_models_server31client10_rev/'+str(datasets_list[j])+'/server_best_val.pth'))
    server.eval()
    for k in range(len(datasets_list)):
        # try:
        #     clients[j].load_state_dict(torch.load('./safe_5/saved_models_server31client10/'+str(datasets_list[j])+'/client_best_val.pth'))
        # except:
        clients[j].load_state_dict(torch.load('./safe/saved_models_server31client10_rev/tmp_client_bg'+str(datasets_list[j]-1)+'.pth'))
        clients[j].eval()
        print("client ", datasets_list[j], "dataset ", datasets_list[k])
        test(server, clients[j], datasets[k], device=device, vis_dir="client_"+str(j+1)+"_dataset_"+str(k+1))
