import sys
sys.path.append("../..")

from train import train, test
import torch
from models.mask2former import Mask2Former_Client, Mask2Former_Server
from data_utils import get_data
import yaml
import sys

datasets_list = [1, 2, 3, 4, 5, 6, 7,8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
clients = []
datasets = []
device = sys.argv[1]

for i in range(len(datasets_list)):
    clients.append(Mask2Former_Client(64))
server = Mask2Former_Server(19, 64, '../../mmsegmentation/configs/mask2former/cityscapes_blackfed.py')

with open('../../data_configs/cityscapes.yml', 'r') as f:
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
        server, clients[j] = train(server, clients[j], datasets[j], j, save_path='./saved_models_dice/'+str(datasets_list[j]), loss_string='dice', device=device, bs=8 )
        torch.cuda.empty_cache()

#testing
for j in range(len(datasets_list)):
    server.load_state_dict(torch.load('./tmp_server.pth'))
    try:
        clients[j].load_state_dict(torch.load('./saved_models_dice/'+str(datasets_list[j])+'/client_best_val.pth'))
    except:
        clients[j].load_state_dict(torch.load('./tmp_client_'+str(datasets_list[j]-1)+'.pth'))
    # test(server, clients[j], datasets[j], device=device)
    print("Loaded Model ", datasets_list[j])
    for k in range(len(datasets_list)):
        test(server, clients[j], datasets[k], device=device)
