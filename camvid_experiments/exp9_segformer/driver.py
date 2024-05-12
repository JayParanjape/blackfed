import sys
sys.path.append("../..")

from train_white import train, test
import torch
from models.segformer import SegFormer_Client, SegFormer_Server
from data_utils import get_data
import yaml


datasets_list = [1, 2, 3, 4]
clients = []
datasets = []
device = sys.argv[1]

for i in range(len(datasets_list)):
    clients.append(SegFormer_Client(n_channels=64))
server = SegFormer_Server(n_channels=64, n_classes=32, config_path='/mnt/store/jparanj1/blackfed/mmsegmentation/configs/segformer/segformer_blackfed_camvid.py')

with open('../../data_configs/camvid.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

for i in range(len(datasets_list)):
    datasets.append(get_data(config, datasets_list[i]))

    
# num_meta_epochs = 1000
# for i in range(num_meta_epochs):
#     for j in range(len(datasets_list)):
#         if i==0 and j==0:
#             try:
#                 server.load_state_dict(torch.load('./saved_models/tmp_server.pth'))
                    
#                 clients[j].load_state_dict(torch.load('./saved_models/tmp_client_'+str(j)+'.pth'))
#             except:
#                 pass
#         print("Training for dataset ", datasets_list[j], " mega epoch ",i)
#         server, clients[j] = train(server, clients[j], datasets[j], j, save_path='./saved_models/'+str(datasets_list[j]), loss_string='dice', device=device, tmp_save_path='./saved_models', bs=4)

#testing
for j in range(len(datasets_list)):
    server.load_state_dict(torch.load('./saved_models/tmp_server.pth'))
    server.eval()
    for k in range(len(datasets_list)):
        try:
            clients[j].load_state_dict(torch.load('./saved_models/'+str(datasets_list[j])+'/client_best_val_dice.pth'))
        except:
            clients[j].load_state_dict(torch.load('./saved_models/tmp_client_'+str(datasets_list[j]-1)+'.pth'))
        clients[j].eval()
        print("client ", datasets_list[j], "dataset ", datasets_list[k])
        test(server, clients[j], datasets[k], device=device)

