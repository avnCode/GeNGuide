import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import FinetuneIncrementalNet
from torchvision import transforms
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.metrics.pairwise import cosine_similarity  
from torch.nn.functional import kl_div, softmax  
import random
from utils.toolkit import tensor2numpy, accuracy
import copy
import os
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from sklearn.neighbors import NearestNeighbors
torch.manual_seed(2024)
np.random.seed(2024)

epochs = 20
lrate = 0.01 
milestones = [60,100,140]
lrate_decay = 0.1
batch_size = 128
split_ratio = 0.1
T = 2
weight_decay = 5e-4
num_workers = 8
ca_epochs = 5
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


class GeNGuide(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = FinetuneIncrementalNet(args['convnet_type'], pretrained=True)
        self.log_path = "logs/{}_{}".format(args['model_name'], args['model_postfix'])
        self.model_prefix = args['prefix']
        if 'epochs' in args.keys():
            global epochs
            epochs = args['epochs'] 
        if 'milestones' in args.keys():
            global milestones
            milestones = args['milestones']
        if 'lr' in args.keys():
            global lrate
            lrate = args['lr']
            print('set lr to ', lrate)
        if 'bcb_lrscale' in args.keys():
            self.bcb_lrscale = args['bcb_lrscale']
        else:
            self.bcb_lrscale = 1.0/100
        if self.bcb_lrscale == 0:
            self.fix_bcb = True
        else:
            self.fix_bcb = False
        print('fic_bcb', self.fix_bcb)


        
        if 'save_before_ca' in args.keys() and args['save_before_ca']:
            self.save_before_ca = True
        else:
            self.save_before_ca = False

        if 'ca_epochs' in args.keys():
            global ca_epochs
            ca_epochs = args['ca_epochs'] 

        if 'ca_with_logit_norm' in args.keys() and args['ca_with_logit_norm']>0:
            self.logit_norm = args['ca_with_logit_norm']
        else:
            self.logit_norm = None

        self.run_id = args['run_id']
        self.seed = args['seed']
        self.task_sizes = []
        self.modify = args['modify']
        self.neighbors_to_check = args['neighbors_to_check']
        self.biased_cov = args["biased"]
        self.mean_only = args["mean_only"]
        self.N2 = args["N2"]
        self.loss_modification = False

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        # self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}'.format(self.seed), head_only=self.fix_bcb)
        self._network.fc.recall()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.topk = self._total_classes if self._total_classes<5 else 5
        self._network.update_fc(data_manager.get_task_size(self._cur_task))                                                 # Extending classifier for stage 1 training
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))
        self._network.to(self._device)

        train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),                          # Getting current task dataset
                                                  source='train', mode='train',
                                                  appendent=[], with_raw=False)
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')                 # Test: All test data in current task and all previous tasks
        # source: test
        dset_name = data_manager.dataset_name.lower()

        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self._stage1_training(self.train_loader, self.test_loader, data_manager)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


        # CA
        self._network.fc.backup()
        print("Task Number")
        print(self._cur_task)
        if self._cur_task == 0:
            if self.loss_modification:
                self.save_checkpoint(self.log_path+'/'+self.model_prefix + 'withloss'+ '_seed{}_before_ca'.format(self.seed), head_only=self.fix_bcb)
            else:
                self.save_checkpoint(self.log_path+'/'+self.model_prefix + '_seed{}_before_ca'.format(self.seed), head_only=self.fix_bcb)
        
                
        # if self._cur_task == 1:
        #     self.save_network(self._cur_task)
        #     print("Network saved......")
            

        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        if self._cur_task>-1 and ca_epochs>0:
            self._stage2_compact_classifier(task_size)
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
    
    def save_network(self, task_id):
        """Save network state after a task"""
        if not isinstance(self._network, nn.DataParallel):
            state = {
                'model_state_dict': self._network.state_dict(),
                'task_id': task_id,
                'known_classes': self._known_classes,
                'total_classes': self._total_classes
            }
        else:
            state = {
                'model_state_dict': self._network.module.state_dict(),
                'task_id': task_id,
                'known_classes': self._known_classes,
                'total_classes': self._total_classes
            }
            
        save_path = os.path.join(self.save_dir, f'task_{task_id}_checkpoint.pth')
        torch.save(state, save_path)
        logging.info(f"Saved network state after task {task_id} to {save_path}")

    def _run(self, train_loader, test_loader, optimizer, scheduler, data_manager):
        run_epochs = epochs        
        for epoch in range(1, run_epochs+1):
            train_features = []                                                                                                 # Extract features from the training data
            train_labels = []
            train_idxs = []
            self._network.eval()
            if self.modify:
                for _, inputs, targets, _, train_ids in train_loader:
                    inputs, targets, train_ids = inputs.to(self._device), targets.to(self._device), train_ids.to(self._device)
                    with torch.no_grad():
                        train_features.append(self._network.module.extract_vector(inputs).cpu().numpy())
                    train_labels.append(targets.cpu().numpy())
                    train_idxs.append(train_ids.cpu().numpy())
                    torch.cuda.empty_cache()
                train_idxs = np.concatenate(train_idxs)
                train_features = np.vstack(train_features)
                train_labels = np.concatenate(train_labels)
                neighbors_to_check = self.neighbors_to_check
                if epoch==1:
                    print(f" Checking {neighbors_to_check} neighbors of a points other than itself ")
                knn = NearestNeighbors(n_neighbors=neighbors_to_check+1)                                                                               # Create and fit NearestNeighbors model
                knn.fit(train_features)
                distances, ind = knn.kneighbors(train_features)
                neighbor_labels = train_labels[ind]
                wt = []
                for i, target in enumerate(train_labels):
                    neighbor_labels_i = neighbor_labels[i]
                    neighbor_distances_i = distances[i, 1:]  
                    neighbor_weights = 1.0 / (neighbor_distances_i + 1e-8)  
                    neighbor_weights /= np.sum(neighbor_weights)
                    wts = 0
                    for j, neigh in enumerate(neighbor_labels_i):
                        if neighbor_labels_i[j] == target and j!=0:
                            wts+=neighbor_weights[j-1]*neighbors_to_check
                    wt.append(wts)
                wt = torch.tensor(wt, dtype=torch.float64, device='cpu')  
                data_manager._update_rank(wt, train_idxs)
                del wt, neighbor_labels, ind, train_labels, train_features, train_idxs
                torch.cuda.set_per_process_memory_fraction(0.9, device=0)
                torch.cuda.empty_cache()

            self._network.train()
            losses = 0.
            for i, (_, inputs, targets, _, idx) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                output = self._network(inputs, bcb_no_grad=self.fix_bcb, fc_only=False)
                features = output['features']
                logits = output['logits']
                cur_targets = torch.where(targets-self._known_classes>=0,targets-self._known_classes,-100)
                loss = 0
                if self.modify:
                    batch_wt = data_manager._get_weights(idx)
                    batch_wt = torch.tensor(batch_wt, dtype=torch.float64, device='cuda')
                    loss = (F.cross_entropy(logits[:, self._known_classes:], cur_targets, reduction='none') * batch_wt/neighbors_to_check).mean()
                else:
                    loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                torch.cuda.empty_cache()
            scheduler.step()

            if epoch%5==0:
                train_acc = self._compute_accuracy(self._network, train_loader)
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader))
            logging.info(info)


    def _stage1_training(self, train_loader, test_loader, data_manager):
        '''
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        '''
        base_params = self._network.convnet.parameters()
        base_fc_params = [p for p in self._network.fc.parameters() if p.requires_grad==True]
        head_scale = 1. if 'moco' in self.log_path else 1.
        if not self.fix_bcb:
            base_params = {'params': base_params, 'lr': lrate*self.bcb_lrscale, 'weight_decay': weight_decay}
            base_fc_params = {'params': base_fc_params, 'lr': lrate*head_scale, 'weight_decay': weight_decay}
            network_params = [base_params, base_fc_params]
        else:
            for p in base_params:
                p.requires_grad = False
            network_params = [{'params': base_fc_params, 'lr': lrate*head_scale, 'weight_decay': weight_decay}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._run(train_loader, test_loader, optimizer, scheduler, data_manager)

    def _stage2_compact_classifier(self, task_size):
            for p in self._network.fc.parameters():
                p.requires_grad=True
            torch.manual_seed(2024)
            run_epochs = ca_epochs
            crct_num = self._total_classes    
            param_list = [p for p in self._network.fc.parameters() if p.requires_grad]
            network_params = [{'params': param_list, 'lr': lrate,
                            'weight_decay': weight_decay}]
            optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

            self._network.to(self._device)
            if len(self._multiple_gpus) > 1:
                self._network = nn.DataParallel(self._network, self._multiple_gpus)

            self._network.eval()
            for epoch in range(run_epochs):
                losses = 0.

                sampled_data = []
                sampled_label = []
                
            
                for c_id in range(crct_num):
                    t_id = c_id//task_size
                    decay = (t_id+1)/(self._cur_task+1)*0.1
                    cls_mean = torch.tensor(self._class_means[c_id], dtype=torch.float64).to(self._device)*(0.9+decay) # torch.from_numpy(self._class_means[c_id]).to(self._device)
                    if self.mean_only:
                        num_sampled_pcls = 1
                        sampled_data.append(cls_mean.unsqueeze(dim=0))   
                        sampled_label.extend([c_id]*num_sampled_pcls)
                    else:
                        if self.modify:
                            num_sampled_pcls = 64
                        else:
                            num_sampled_pcls = 256
                        cls_cov = self._class_covs[c_id].to(self._device)
                        m = MultivariateNormal(cls_mean.float(), cls_cov.float())
                        sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                        sampled_data.append(sampled_data_single)    
                        sampled_label.extend([c_id]*num_sampled_pcls)            


                sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
                sampled_label = torch.tensor(sampled_label).long().to(self._device)

                inputs = sampled_data
                targets= sampled_label

                sf_indexes = torch.randperm(inputs.size(0))
                inputs = inputs[sf_indexes]
                targets = targets[sf_indexes]

                
                for _iter in range(crct_num):
                    if self.mean_only:
                        outputs = self._network(inputs, bcb_no_grad=True, fc_only=True)
                        logits = outputs['logits']
                        if self.logit_norm is not None:
                            per_task_norm = []
                            prev_t_size = 0
                            cur_t_size = 0
                            for _ti in range(self._cur_task+1):
                                cur_t_size += self.task_sizes[_ti]
                                temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                                per_task_norm.append(temp_norm)
                                prev_t_size += self.task_sizes[_ti]
                            per_task_norm = torch.cat(per_task_norm, dim=-1)
                            norms = per_task_norm.mean(dim=-1, keepdim=True)
                                
                            norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                            decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                            loss = F.cross_entropy(decoupled_logits, targets)
                        else:
                            loss = F.cross_entropy(logits[:, :crct_num], targets)                        
                    else:
                        inp = inputs[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                        tgt = targets[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                        outputs = self._network(inp, bcb_no_grad=True, fc_only=True)
                        logits = outputs['logits']
                        if self.logit_norm is not None:
                            per_task_norm = []
                            prev_t_size = 0
                            cur_t_size = 0
                            for _ti in range(self._cur_task+1):
                                cur_t_size += self.task_sizes[_ti]
                                temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                                per_task_norm.append(temp_norm)
                                prev_t_size += self.task_sizes[_ti]
                            per_task_norm = torch.cat(per_task_norm, dim=-1)
                            norms = per_task_norm.mean(dim=-1, keepdim=True)
                                
                            norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                            decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                            loss = F.cross_entropy(decoupled_logits, tgt)
                        else:
                            loss = F.cross_entropy(logits[:, :crct_num], tgt)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                scheduler.step()
                test_acc = self._compute_accuracy(self._network, self.test_loader)
                info = 'CA Task {} => Loss {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, losses/self._total_classes, test_acc)
                logging.info(info)



    def calculate_ncr_loss(self, logits, features, temperature=2.0, k=5):  
        m = logits.shape[0]  
        normalized_logits = logits / temperature  
        cosine_sim = cosine_similarity(features.cpu().detach().numpy())  
        cosine_sim = torch.tensor(cosine_sim).to(features.device)  
        cosine_sim.fill_diagonal_(0)  
        topk_sim, topk_indices = torch.topk(cosine_sim, k=k, dim=-1)  
        loss_tmp = 0.0  
        for i in range(m):  
            # Get the logits for the current sample and its neighbors  
            zi = normalized_logits[i]  
            zj = normalized_logits[topk_indices[i]]  
    
            # Calculate the softmax of the logits  
            soft_zi = softmax(zi, dim=-1)  
            soft_zj = softmax(zj / temperature, dim=-1)  
    
            # Calculate the weighted sum of neighbors' logits  
            weights = topk_sim[i] / topk_sim[i].sum()  
            weighted_logits = torch.sum(weights[:, None] * soft_zj, dim=0)  

            loss_tmp += kl_div(soft_zi.log(), weighted_logits, reduction='batchmean')  
    
        return loss_tmp / m  