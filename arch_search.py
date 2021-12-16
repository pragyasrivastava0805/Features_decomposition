import os
import matplotlib.pyplot as plt
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
from torch.utils.data import Dataset,DataLoader
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from Generalized_features_Extractor import *
from dataset import *
from architect import Architect
from Classification_Head import *
from sklearn.manifold import t_sne
from Mnist_dataset import *
from utils import *
from dalib.vision.datasets import Office31, OfficeHome, VisDA2017, ImageCLEF


parser = argparse.ArgumentParser()
parser.add_argument('--data_source', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--data_target',type=str, default='../data',help='location')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float,
                    default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=5, help='report frequency')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--epochs', type=int, default=50,
                    help='num of training epochs')
parser.add_argument('--init_channels', type=int,
                    default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=6,
                    help='total number of layers')
parser.add_argument('--model_path', type=str,
                    default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true',
                    default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int,
                    default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float,
                    default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP-ab1-', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float,
                    default=0.70, help='portion of training data')
parser.add_argument('--unrolled', action='store_true',
                    default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float,
                    default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float,
                    default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--num_classes',type=int,default=31,help='number of categories in which the images fall')
parser.add_argument('--learning_rate_feature_extractor', type=float, default=0.025)
parser.add_argument('--learning_rate_head_g', type=float, default=0.025)
parser.add_argument('--weight_decay_fe', type=float, default=3e-4)
parser.add_argument('--weight_decay_hg', type=float, default=3e-4)
parser.add_argument('--is_parallel', type=int, default=0)
parser.add_argument('--source_domain',type=str,default='A',help='Source Domain')
parser.add_argument('--target_domain',type=str,default='D',help='Target Domain')
parser.add_argument('--data_dir',type=str,metavar='PATH',default=os.path.join('root_dir', 'data'))
parser.add_argument('--dataset',type=str, choices=['office31', 'visda', 'officehome', 'imageclef'],help='Dataset',default='office31')
parser.add_argument('--baseline1',action='store_true',default=False,help="train on tgt")
parser.add_argument('--baseline2',action='store_true',default=False, help="train on src and tgt")
parser.add_argument('--ours1',action='store_true',default=False,help="train on reweighted src and tgt")
parser.add_argument('--baseline3',action='store_true',default=False,help="train on tgt with regularization from src")
parser.add_argument('--baseline4',action='store_true',default=False,help="train on src and tgt with regularization from src")
parser.add_argument('--ours2',action='store_true',default=False,help='train on tgt with regularization from reweighted src')
parser.add_argument('--ours3',action='store_true',default=False,help='train on reweighted src and tgt with regularization from  src')
parser.add_argument('--ours4',action='store_true',default=True,help='train on src and tgt with regularization from reweighted src')
parser.add_argument('--ours5',action='store_true',default=False,help='train on reweighted src and tgt with regularization from reweighted src')



args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  if not args.is_parallel:
    torch.cuda.set_device(int(args.gpu))
    logging.info('gpu device = %d' % int(args.gpu))
  else:
    logging.info('gpu device = %s' % args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled = True
  torch.cuda.manual_seed(args.seed)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  num_domains=2
  model = Network(args.init_channels, num_domains, args.layers, criterion)
  output_dim=args.init_channels*16
  feature_extractor= ResNet(criterion,output_dim)
  head_g = Classification_Head(output_dim,args.num_classes)
  
  print(model.parameters())
  if args.is_parallel:
    gpus = [int(i) for i in args.gpu.split(',')]
    model = nn.parallel.DataParallel(
        model, device_ids=gpus, output_device=gpus[0]).cuda()
    feature_extractor= nn.parallel.DataParallel(
        feature_extractor, device_ids=gpus, output_device=gpus[0]).cuda()
    head_g = nn.parallel.DataParallel(
        head_g, device_ids=gpus, output_device=gpus[0]).cuda()
   
    model = model.module
    feature_extractor = feature_extractor.module
    head_g = head_g.module
  
  else:
    model = model.cuda()
    feature_extractor=feature_extractor.cuda()
    head_g = head_g.cuda()


 

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  logging.info("param size = %fMB", utils.count_parameters_in_MB(feature_extractor))
  logging.info("param size = %fMB", utils.count_parameters_in_MB(head_g))



  optimizer = torch.optim.SGD(model.parameters(),args.learning_rate,momentum=args.momentum, weight_decay=args.weight_decay)
  optimizer_fe = torch.optim.SGD(
      feature_extractor.parameters(),
      args.learning_rate_feature_extractor,
      momentum=args.momentum,
      weight_decay=args.weight_decay_fe)
  optimizer_hg = torch.optim.SGD(head_g.parameters(),
                                    lr=args.learning_rate_head_g,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay_hg)
  train_transform = transform(train=True)
  test_transform = transform(train=False)
  data_root = os.path.join(args.data_dir, args.dataset)
  if args.dataset == 'office31':
    getDataset = Office31
  elif args.dataset == 'officehome':
    getDataset = OfficeHome
  elif args.dataset == 'visda':
    getDataset = VisDA2017
  elif args.dataset == 'imageclef':
    getDataset = ImageCLEF
  train_target_dataset =  get_dataset(args.data_target + '/train',args.target_domain)
  train_source_dataset = get_dataset(args.data_source + '/train',args.source_domain)
  valid_target_dataset = get_dataset(args.data_target + '/val',args.target_domain)
  valid_source_dataset = get_dataset(args.data_source + '/val',args.source_domain)
  test_target_dataset = get_dataset(args.data_target + '/test',args.target_domain)
  valid_dataset = valid_target_dataset
  if args.baseline4 or args.ours3 or args.ours4 or args.ours5 or args.baseline2 or args.ours1:
    val_target_source_dataset = torch.utils.data.ConcatDataset(
            [valid_target_dataset, valid_source_dataset])
    valid_dataset = val_target_source_dataset

  train_source_loader = torch.utils.data.DataLoader(train_source_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=6,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last=True)
  train_target_loader = torch.utils.data.DataLoader(train_target_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=6,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last=True)
    

  valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=args.batch_size ,
                                               num_workers=6,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last=False)
  test_loader = torch.utils.data.DataLoader(test_target_dataset,
                                              batch_size=args.batch_size ,
                                              num_workers=6,
                                              shuffle=True,
                                              pin_memory=True,
                                              drop_last=False)
 
  




  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, float(args.epochs), eta_min=args.learning_rate_min)




  architect = Architect(model,feature_extractor,head_g,args)

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)


    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))
    
    train_acc_s, train_acc_t = train(train_source_loader,valid_loader,train_target_loader, criterion,optimizer,optimizer_fe, optimizer_hg,lr,feature_extractor,head_g,model,architect,args.batch_size)
    logging.info('train_acc %f %f', train_acc_s,train_acc_t)

   
    # validation
    valid_acc_t,mean_acc = infer(valid_loader,model,feature_extractor,head_g,criterion,args.batch_size)
    logging.info('valid_acc %f %f', valid_acc_t,mean_acc)
    utils.save(model, os.path.join(args.save, 'weights.pt'))
  genotype = model.genotype()
  logging.info('genotype = %s', genotype)


  
def train(train_source_loader,val_loader,train_target_loader,criterion,optimizer,optimizer_fe,optimizer_hg,lr,feature_extractor,head_g,model,architect,batch_size):
  len_dataloader=min(len(train_source_loader),len(train_target_loader))
  data_source_iter=iter(train_source_loader)
  data_target_iter=iter(train_target_loader)
  i=0
  
  n_total=0
  n_correct_source=0
  n_correct_target=0
  while i<len_dataloader:
    model.train()
    data_source=data_source_iter.next()
    input_img_source,class_label_source=data_source
    print(i)    
    input_img_source = input_img_source.cuda()
    generalized_features_source=feature_extractor(input_img_source)
    class_label_source = class_label_source.cuda(non_blocking=True)
    domain_label_source=torch.zeros(batch_size,dtype=torch.long)
    domain_label_source=domain_label_source.cuda(non_blocking=True)
    _,domain_logits_source=model(input_img_source)
    domain_loss_source=criterion(domain_logits_source,domain_label_source)
 

    data_target=data_target_iter.next()
    input_img_target,class_label_target=data_target
  

    input_img_target = input_img_target.cuda()
    generalized_features_target=feature_extractor(input_img_target)
    class_label_target = class_label_target.cuda(non_blocking=True)
    domain_label_target=torch.ones(batch_size,dtype=torch.long)
    
    domain_label_target=domain_label_target.cuda(non_blocking=True)
    _,domain_logits_target=model(input_img_target)
    domain_loss_target=criterion(domain_logits_target,domain_label_target)

    
    input_search, target_search = next(iter(val_loader))

    input_search= input_search.cuda()
    target_search= target_search.cuda(non_blocking=True)
    

    domain_loss_total=domain_loss_target+domain_loss_source
    optimizer.zero_grad()
    domain_loss_total.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    domain_features_source,_=model(input_img_source)
    domain_features_target,_=model(input_img_target)



    class_features_source=generalized_features_source-domain_features_source
    class_features_target=generalized_features_target-domain_features_target
    if i==0:
      allFeatures_Source=class_features_source
      allFeatures_Target=class_features_target
    else:
      allFeatures_Source=torch.cat((allFeatures_Source,class_features_source),0)
      allFeatures_Target=torch.cat((allFeatures_Target,class_features_target),0)
    
    class_logits_source=head_g(class_features_source)
    class_logits_target=head_g(class_features_target)
    class_loss_source=criterion(class_logits_source,class_label_source)
    class_loss_target=criterion(class_logits_target,class_label_target)

    class_loss_total=class_loss_source+class_loss_target
    optimizer_fe.zero_grad()
    optimizer_hg.zero_grad()
    class_loss_total.backward()
    optimizer_fe.step()
    optimizer_hg.step()
    architect.step(input_img_source,class_label_source,domain_label_source,input_img_target,class_label_target,domain_label_target,input_search, target_search, lr, optimizer,optimizer_fe,optimizer_hg,args.init_channels)
    architect.step1(input_img_source,class_label_source,domain_label_source,input_img_target,class_label_target,domain_label_target,input_search, target_search, lr, optimizer,optimizer_fe,optimizer_hg,args.init_channels)
    architect.step2(input_img_source,class_label_source,domain_label_source,input_img_target,class_label_target,domain_label_target,input_search, target_search, lr, optimizer,optimizer_fe,optimizer_hg,args.init_channels)
    pred_s = class_logits_source.data.max(1, keepdim=True)[1] 
    n_correct_source += pred_s.eq(class_label_source.data.view_as(pred_s)).sum().item()
    n_total += batch_size
    pred_t = class_logits_target.data.max(1, keepdim=True)[1]
    n_correct_target += pred_t.eq(class_label_target.data.view_as(pred_t)).sum().item()

    if i % args.report_freq == 0:
      logging.info('train %03d %e %f', i,class_loss_source,class_loss_target)
    i+=1
  acc_source = n_correct_source * 1.0 / n_total
  acc_target = n_correct_target * 1.0 / n_total
  return acc_source, acc_target


def infer(val_loader,model,feature_extractor,head_g,criterion,batch_size):
  len_dataloader=len(val_loader)
  data_target_iter=iter(val_loader)
  i=0
  res={}
  res['probs'],res['gt']=[],[]
  model.eval()
  with torch.no_grad():
    n_total=0
    n_correct_target=0
    n_correct_source=0
    while i<len_dataloader:
 
      data_target=data_target_iter.next()
      input_img_target,class_label_target=data_target
     
      class_label_target=class_label_target.cuda()
      input_img_target = input_img_target.cuda()
      domain_features_target,_=model(input_img_target)
      generalized_features_target=feature_extractor(input_img_target)
      class_features_target=generalized_features_target-domain_features_target
      logits_target=head_g(class_features_target)
      res['probs']+=[logits_target]
      res['gt']+=[class_label_target]
      i+=1
      n_total += batch_size
      pred_t = logits_target.data.max(1, keepdim=True)[1]
      n_correct_target += pred_t.eq(class_label_target.data.view_as(pred_t)).sum().item()
      
  probs=torch.cat(res['probs'],dim=0)
  gt=torch.cat(res['gt'],dim=0)
  eval_res=utils.mean_accuracy(probs,gt)
  acc_target = n_correct_target * 1.0 / n_total
    

  return acc_target,eval_res
 


      
  
    

if __name__ == '__main__':
  main()
