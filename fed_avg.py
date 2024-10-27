import os
import torch
import sys

#set up command line parameters
mode = sys.argv[1]  #['fedavg']

#set up paths for individual models
center_model_root = './saved_models'
center_metadata = './center_metadata.txt'

with open(center_metadata,'r') as f:
    metadata = f.readlines()
f.close()

nk = [int(line[line.strip().find(' '):]) for line in metadata]
frac_nk = [i/sum(nk) for i in nk]
fed_wt = torch.load(os.path.join(center_model_root,'client_'+str(0)+'_best_val.pth'),map_location='cpu')
for k,v in fed_wt.items():
    fed_wt[k] = v*frac_nk[0]

if mode=='fedavg':
    #load saved model weights and perform weighted averaging
    for i in range(1, len(nk)):
        wt = torch.load(os.path.join(center_model_root,'client_'+str(i)+'_best_val.pth'), map_location='cpu')
        for k,v in wt.items():
            fed_wt[k] += v*frac_nk[i]
    
    #save the federated model at path expected by the clients
    torch.save(fed_wt, './fed_learning_model.pth')

elif mode=='fedper':
    for i in range(1, len(nk)):
        wt = torch.load(os.path.join(center_model_root,'client_'+str(i)+'_best_val.pth'), map_location='cpu')
        for k,v in wt.items():
            if 'classifier' not in k:
                fed_wt[k] += v*frac_nk[i]
    torch.save(fed_wt, './fed_learning_model.pth')
    