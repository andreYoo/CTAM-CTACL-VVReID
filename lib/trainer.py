import time
import numpy as np
import torch
from .utils.meters import AverageMeter
from .utils.plot_figures import utils_for_fig3
from .utils.mlp_statistics import precision_recall
from .loss import CTAM_SSCL_Loss
from .onlinesamplemining import DSM,KNN,SS
import pdb
device_0 = torch.device('cuda:0')
device_1 = torch.device('cuda:1')
device_cpu = torch.device('cpu')

class Trainer(object):
    def __init__(self, cfg, model, memory,use_dram=False):
        super(Trainer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.memory = memory
        self.use_dram = use_dram
        self.eval_mlp = True
        self.ldb = cfg.SSCL.L
        self.thr = cfg.SSCL.T

        self.criterion = CTAM_SSCL_Loss(temperature=cfg.SSCL.TEMP, base_temperature=cfg.SSCL.BTEMP,contrast_mode=cfg.SSCL.MODE).to(self.device)



    def train(self, epoch, data_loader, optimizer,writer,gi=True, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()



        if gi==True and epoch==0:
            print('Memory Re-initisliation')
            with torch.no_grad():
                for i, inputs in enumerate(data_loader):
                    inputs,camid,tid,pids = self._parse_data(inputs)
                    outputs = self.model(inputs, 'l2feat')
                    self.memory.store(outputs,camid,tid,pids)
                    #self.graph.global_normalisation()

            print('Done!')
        

        if epoch % 5 == 0 and epoch != 0:
            print('Look-up table Overhaul - [reinitialising]')
            with torch.no_grad():
                for i, inputs in enumerate(data_loader):
                    inputs,camid,tid, pids = self._parse_data(inputs)
                    outputs = self.model(inputs, 'l2feat')
                    self.memory.store(outputs,camid,tid,pids)
                    print('[Reinitilisaing] Overhaul (%.3f %%) is finished'%(i/len(data_loader)*100.0))
                    #self.graph.global_normalisation()
                print('Dictionary overhaul is finished. - Overhaul (100%%) is finished')

        
        precision = 0.0
        recall = 0.0
        _num_positive = 0



        for i, inputs in enumerate(data_loader):
            if i!=0 and i%500==0:
                print('VGA Cooling for 120 secs')
                time.sleep(120)

            data_time.update(time.time() - end)
            inputs,camid,tid,pids  = self._parse_data_v2(inputs)
            camid = camid.to(device_1)
            tid = tid.to(device_1)
            outputs = self.model(inputs, 'l2feat') #output = feature batch of input images
            #Batch output - so start on this section
            logits = self.memory(outputs, pids,epoch=epoch)
            if epoch > 5:
                #Local + global contrastive learnings
                #local_loss,_hard_pos = self.criterion(self.memory,logits,camid,hard_pos=None,trackids=tid,type='local')
                #global_loss,_tmp_pos = self.criterion(self.memory,logits,camid,hard_pos=_hard_pos,trackids=tid,type='cl',thr=self.thr)
                cl_loss,_ttt = self.criterion(self.memory,logits,camid,hard_pos=None,trackids=tid,type='cl')
                cam_kl_loss = self.criterion(self.memory,outputs.to(device_1),camid,hard_pos=None,type='cam')
                loss = cl_loss +self.ldb*cam_kl_loss
                #loss = local_loss  + global_loss

            else:
                loss,_ = self.criterion(self.memory,logits,camid,trackids=tid,type='local')


            losses.update(loss.item(), outputs.size(0))
            writer.add_scalar("Loss/train", loss.item(), epoch * len(data_loader)+i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                log = "Epoch: [{}][{}/{}], Time {:.3f} ({:.3f}), Data {:.3f} ({:.3f}), Loss {:.3f} ({:.3f}) " \
                    .format(epoch, i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            losses.val, losses.avg)
                print(log)
            torch.cuda.empty_cache()
        if epoch > 5:
            plog = "[Epoch {}]Average # of positive {} Prediction {:.3f} Recall {:.3f}".format(epoch,    _num_positive/len(data_loader),precision/len(data_loader), recall/len(data_loader))
            print(plog)


    def _parse_data(self, inputs):
        imgs, _t1, camid,tid,pids = inputs #img, fname,camid, pid, idx
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        #print(_t1)
        #print(_t2)
        return inputs,camid,tid, pids

    def _parse_data_v2(self, inputs):
        imgs, _t1,camid,tid, pids = inputs #img, fname,camid, pid, idx
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        #print(_t1)
        #print(_t2)
        return inputs,camid,tid,pids
