import os
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




parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float,
                    default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--epochs', type=int, default=50,
                    help='num of training epochs')
parser.add_argument('--init_channels', type=int,
                    default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8,
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
                    default=0.5, help='portion of training data')
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
parser.add_argument('--source',type=str,default='amazon',help='Source Domain')
parser.add_argument('--target',type=str,default='dslr',help='Target Domain')

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
  model = model.cuda()
  output_dim=args.init_channels*16
  
  feature_extractor= ResNet50(output_dim).cuda()
  head_g = Classification_Head(output_dim,args.num_classes).cuda()
  

  if args.is_parallel:
    gpus = [int(i) for i in args.gpu.split(',')]
    model = nn.parallel.DataParallel(
        model, device_ids=gpus, output_device=gpus[0])
    feature_extractor= nn.parallel.DataParallel(
        feature_extractor, device_ids=gpus, output_device=gpus[0])
    head_g = nn.parallel.DataParallel(
        head_g, device_ids=gpus, output_device=gpus[0])
   
    model = model.module
    feature_extractor = feature_extractor.module
    head_g = head_g.module
 

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(model.parameters(),args.learning_rate,momentum=args.momentum, weight_decay=args.weight_decay)
  optimizer_fe = torch.optim.SGD(
      feature_extractor.parameters(),
      args.learning_rate_feature_extractor,
      momentum=args.momentum,
      weight_decay=args.weight_decay_fe)
  optimizer_hg = torch.optim.Adam(head_g.parameters(),
                                    lr=args.learning_rate_feature_extractor,
                                    betas=(0.5, 0.999),
                                    weight_decay=args.weight_decay_hg)
  source_train_loader,source_val_loader=get_dataloader(get_office_dataset(args.source,args.data),args.batch_size,num_workers=0,train_ratio=args.train_portion)
  target_train_loader,target_val_loader=get_dataloader(get_office_dataset(args.target,args.data),args.batch_size,num_workers=0,train_ratio=args.train_portion)
 

  




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
    
    train_acc, train_obj = train(source_train_loader,source_val_loader,target_train_loader,target_val_loader, criterion,optimizer,optimizer_fe, optimizer_hg,lr,feature_extractor,head_g,model,architect)
    logging.info('train_acc %f', train_acc)

   
    # validation
    valid_acc, valid_obj = infer(source_val_loader,target_val_loader,model,feature_extractor,head_g,criterion)
    logging.info('valid_acc %f', valid_acc)
  


def train(source_train_loader,source_val_loader,target_train_loader,target_val_loader,criterion,optimizer,optimizer_fe,optimizer_hg,lr,feature_extractor,head_g,model,architect):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  len_dataloader=min(len(source_train_loader),len(target_train_loader))
  data_source_iter=iter(source_train_loader)
  data_target_iter=iter(target_train_loader)
  i=0
  image_size=256

  while i<len_dataloader:
    model.train()
    data_source=data_source_iter.next()
    s_img,s_label=data_source
    batch_size=len(s_img)
    
    input_img_source=torch.FloatTensor(batch_size,3,image_size,image_size)
    class_label_source=torch.LongTensor(batch_size)
    input_img_source.resize_as_(s_img).copy_(s_img)
    class_label_source.resize_as(s_label).copy_(s_label)
    input_img_source = input_img_source.cuda()
    generalized_features_source=feature_extractor(input_img_source)
    class_label_source = class_label_source.cuda(non_blocking=True)
    domain_label_source=torch.zeros(batch_size,dtype=torch.long)
    domain_label_source=domain_label_source.cuda(non_blocking=True)
    _,domain_logits=model(input_img_source)
    domain_loss_source=model._loss(input_img_source,domain_label_source)
 

    data_target=data_target_iter.next()
    t_img,t_label=data_target
    batch_size=len(t_img)
    input_img_target=torch.FloatTensor(batch_size,3,image_size,image_size)
    class_label_target=torch.LongTensor(batch_size)
    input_img_target.resize_as_(t_img).copy_(t_img)
    class_label_target.resize_as(t_label).copy_(t_label)
    input_img_target = input_img_target.cuda()
    generalized_features_target=feature_extractor(input_img_target)
    class_label_target = class_label_target.cuda(non_blocking=True)
    domain_label_target=torch.ones(batch_size,dtype=torch.long)
    
    domain_label_target=domain_label_target.cuda(non_blocking=True)
    _,domain_logits=model(input_img_target)
    domain_loss_target=model._loss(input_img_source,domain_label_target)

    domain_loss_total=domain_loss_target+domain_loss_source
    optimizer.zero_grad()
    domain_loss_total.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    domain_features_source,_=model(input_img_source)
    domain_features_target,_=model(input_img_target)



    class_features_source=generalized_features_source-domain_features_source
    class_features_target=generalized_features_target-domain_features_target
    
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


    input_search_source, target_search_source = next(iter(source_val_loader))

    input_search_source= input_search_source.cuda()
    target_search_source= target_search_source.cuda(non_blocking=True)

    input_search_target, target_search_target = next(iter(target_val_loader))
    input_search_target = input_search_target.cuda()
    target_search_target = target_search_target.cuda(non_blocking=True)

    architect.step(input_img_source,class_label_source,domain_label_source,input_img_target,class_label_target,domain_label_target,input_search_source, target_search_source, input_search_target, target_search_target,
                   lr, optimizer,optimizer_fe,optimizer_hg,args.init_channels)
    architect.step1(input_img_source,class_label_source,domain_label_source,input_img_target,class_label_target,domain_label_target,input_search_source, target_search_source, input_search_target, target_search_target,
                   lr, optimizer,optimizer_fe,optimizer_hg,args.init_channels)
    architect.step2(input_img_source,class_label_source,domain_label_source,input_img_target,class_label_target,domain_label_target,input_search_source, target_search_source, input_search_target, target_search_target,
                   lr, optimizer,optimizer_fe,optimizer_hg,args.init_channels)
    
    logits=torch.cat((class_logits_source,class_logits_target),0)
    target=torch.cat((class_label_source,class_label_target),0)
    loss=class_loss_total

 
    

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input_img_source.size(0)+input_img_target.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(source_val_loader,target_val_loader,model,feature_extractor,head_g,criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  len_dataloader=min(len(source_val_loader),len(target_val_loader))
  data_source_iter=iter(source_val_loader)
  data_target_iter=iter(target_val_loader)
  i=0
  image_size=256
  model.eval()
 
  while i<len_dataloader:
    data_source=data_source_iter.next()
    s_img,s_label=data_source
    batch_size=len(s_label)
    
    input_img_source=torch.FloatTensor(batch_size,3,image_size,image_size)
    class_label_source=torch.LongTensor(batch_size)
    input_img_source.resize_as_(s_img).copy_(s_img)
    class_label_source.resize_as(s_label).copy_(s_label)
    class_label_source=class_label_source.cuda()
    input_img_source = input_img_source.cuda()
    domain_features_source,_=model(input_img_source)
    data_target=data_target_iter.next()
    t_img,t_label=data_target
    batch_size=len(t_label)
    input_img_target=torch.FloatTensor(batch_size,3,image_size,image_size)
    class_label_target=torch.LongTensor(batch_size)
    input_img_target.resize_as_(t_img).copy_(t_img)
    class_label_target.resize_as(t_label).copy_(t_label)
    class_label_target=class_label_target.cuda()
    input_img_target = input_img_target.cuda()
    domain_features_target,_=model(input_img_target)
    generalized_features_source=feature_extractor(input_img_source)
    generalized_features_target=feature_extractor(input_img_target)
        
       

    class_features_source=generalized_features_source-domain_features_source
    class_features_target=generalized_features_target-domain_features_target
    logits_source=head_g(class_features_source)
    logits_target=head_g(class_features_target)
    loss_source=criterion(logits_source,class_label_source)
    loss_target=criterion(logits_target,class_label_target)
    loss=loss_target+loss_source
    logits=torch.cat((logits_source,logits_target),0)
    target=torch.cat((class_label_source,class_label_target),0)
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input_img_source.size(0)+input_img_target.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()
