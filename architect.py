import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from Generalized_features_Extractor import *


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

def cal_loss(unrolled_model,unrolled_feature_extractor,unrolled_head_g,input_valid_source,input_valid_target,target_valid_source,target_valid_target):
  domain_features_source,domain_logits_s=unrolled_model(input_valid_source)
  domain_features_target,domain_logits_t=unrolled_model(input_valid_target)
  generalized_features_source = unrolled_feature_extractor(input_valid_source)
  generalized_features_target = unrolled_feature_extractor(input_valid_target)
  logits1=unrolled_head_g(generalized_features_source-domain_features_source)
  logits2=unrolled_head_g(generalized_features_target-domain_features_target)
  crit = nn.CrossEntropyLoss()
  loss1 = crit(logits1, target_valid_source)
  loss2 = crit(logits2, target_valid_target)
  loss=loss1+loss2
  return loss



class Architect(object):

  def __init__(self, model,feature_extractor,head_g,args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.feature_extractor=feature_extractor
    self.head_g=head_g
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input_source,input_target, domain_label_source,domain_label_target, eta, network_optimizer):
    loss_source = self.model._loss(input_source, domain_label_source)
    loss_target = self.model._loss(input_target,domain_label_target)
    loss=loss_source+loss_target
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters(),retain_graph=True)).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def _compute_unrolled_model1(self,input_train_source,input_train_target,class_label_source_train,class_label_target_train,unrolled_model,eta,feature_extractor_optimizer,head_g_optimizer,init_channels):
    domain_features_source,domain_logits_s=unrolled_model(input_train_source)
    domain_features_target,domain_logits_t=unrolled_model(input_train_target)
    generalized_features_source = self.feature_extractor(input_train_source)
    generalized_features_target = self.feature_extractor(input_train_target)
    logits1=self.head_g(generalized_features_source-domain_features_source)
    logits2=self.head_g(generalized_features_target-domain_features_target)
    crit = nn.CrossEntropyLoss()
    loss1 = crit(logits1, class_label_source_train)
    loss2 = crit(logits2, class_label_target_train)
    loss=loss1+loss2
    theta1= _concat(self.feature_extractor.parameters()).data
    theta2= _concat(self.head_g.parameters()).data

    try:
      moment = _concat(feature_extractor_optimizer.state[v]['momentum_buffer'] for v in self.feature_extractor.parameters()).mul_(self.network_momentum)
      moment1= _concat(head_g_optimizer.state[v]['momentum_buffer'] for v in self.head_g.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta1)
      moment1= torch.zeros_like(theta2)
    dtheta1 = _concat(torch.autograd.grad(loss, self.feature_extractor.parameters(),retain_graph=True)).data + self.network_weight_decay*theta1
    dtheta2 = _concat(torch.autograd.grad(loss, self.head_g.parameters(),retain_graph=True)).data + self.network_weight_decay*theta2
    unrolled_feature_extractor = self._construct_model_from_theta1(theta1.sub(eta, moment+dtheta1),init_channels)
    unrolled_head_g = self._construct_model_from_theta2(theta2.sub(eta,moment1+dtheta2))
    return unrolled_feature_extractor,unrolled_head_g

  def step(self, input_train_source,class_label_source_train,domain_label_source_train,input_train_target,class_label_target_train,domain_label_target_train,input_valid_source, target_valid_source, input_valid_target,target_valid_target, eta, network_optimizer,feature_extractor_optimizer,head_g_optimizer,init_channels):
    self.optimizer.zero_grad()
    
    self._backward_step_unrolled(input_train_source, class_label_source_train,domain_label_source_train, input_train_target,class_label_target_train,domain_label_target_train,input_valid_source,target_valid_source,input_valid_target,target_valid_target, eta, network_optimizer,feature_extractor_optimizer,head_g_optimizer,init_channels)
   
    self.optimizer.step()
    
  def step1(self,input_train_source, class_label_source_train,domain_label_source_train, input_train_target,class_label_target_train,domain_label_target_train,input_valid_source,target_valid_source,input_valid_target,target_valid_target,eta, network_optimizer, feature_extractor_optimizer,head_g_optimizer,init_channels):
    self.optimizer.zero_grad()
    
    unrolled_model = self._compute_unrolled_model(input_train_source,input_train_target, domain_label_source_train,domain_label_target_train, eta, network_optimizer)
    unrolled_feature_extractor,unrolled_head_g=self._compute_unrolled_model1(input_train_source,input_train_target,class_label_source_train,class_label_target_train,unrolled_model,eta,feature_extractor_optimizer,head_g_optimizer,init_channels)
    loss=cal_loss(unrolled_model,unrolled_feature_extractor,unrolled_head_g,input_valid_source,input_valid_target,target_valid_source,target_valid_target)
  
    loss.backward()
    
    vector_v_dash = [v.grad.data for v in unrolled_feature_extractor.parameters()]
    
    implicit_grads = self._outer1(vector_v_dash, input_train_source,class_label_source_train,input_train_target,class_label_target_train,domain_label_source_train,domain_label_target_train,unrolled_model,unrolled_feature_extractor,unrolled_head_g, eta)
    
    for v, g in zip(self.model.arch_parameters(), implicit_grads):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)
    
    self.optimizer.step()
  
  def step2(self,input_train_source, class_label_source_train,domain_label_source_train, input_train_target,class_label_target_train,domain_label_target_train,input_valid_source,target_valid_source,input_valid_target,target_valid_target, eta, network_optimizer, feature_extractor_optimizer,head_g_optimizer,init_channels):
    self.optimizer.zero_grad()
    
    unrolled_model = self._compute_unrolled_model(input_train_source,input_train_target, domain_label_source_train,domain_label_target_train, eta, network_optimizer)
    unrolled_feature_extractor,unrolled_head_g=self._compute_unrolled_model1(input_train_source,input_train_target,class_label_source_train,class_label_target_train,unrolled_model,eta,feature_extractor_optimizer,head_g_optimizer,init_channels)
    loss=cal_loss(unrolled_model,unrolled_feature_extractor,unrolled_head_g,input_valid_source,input_valid_target,target_valid_source,target_valid_target)
  
    loss.backward()
    
    vector_g_dash = [v.grad.data for v in unrolled_head_g.parameters()]
    
    implicit_grads = self._outer2(vector_g_dash, input_train_source,class_label_source_train,input_train_target,class_label_target_train,domain_label_source_train,domain_label_target_train,unrolled_model,unrolled_feature_extractor,unrolled_head_g, eta)
    
    for v, g in zip(self.model.arch_parameters(), implicit_grads):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)
    
    self.optimizer.step()
    



  def _backward_step_unrolled(self,input_train_source, class_label_source_train,domain_label_source_train, input_train_target,class_label_target_train,domain_label_target_train,input_valid_source,target_valid_source,input_valid_target,target_valid_target, eta, network_optimizer,feature_extractor_optimizer,head_g_optimizer,init_channels):
    unrolled_model = self._compute_unrolled_model(input_train_source,input_train_target, domain_label_source_train,domain_label_target_train, eta, network_optimizer)
    unrolled_feature_extractor,unrolled_head_g=self._compute_unrolled_model1(input_train_source,input_train_target,class_label_source_train,class_label_target_train,unrolled_model,eta,feature_extractor_optimizer,head_g_optimizer,init_channels)
    
    unrolled_loss=cal_loss(unrolled_model,unrolled_feature_extractor,unrolled_head_g,input_valid_source,input_valid_target,target_valid_source,target_valid_target)

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train_source, input_train_target,domain_label_source_train,domain_label_target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _construct_model_from_theta1(self, theta,init_channels):
    model_new = ResNet50(init_channels*16)
    model_dict = self.feature_extractor.state_dict()

    params, offset = {}, 0
    for k, v in self.feature_extractor.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _construct_model_from_theta2(self, theta):
    model_new = self.head_g.new()
    model_dict = self.head_g.state_dict()

    params, offset = {}, 0
    for k, v in self.head_g.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  

  

  def _hessian_vector_product(self, vector, input_train_source,input_train_target,domain_label_source_train,domain_label_target_train, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    
    loss_s = self.model._loss(input_train_source, domain_label_source_train)
    loss_t=  self.model._loss(input_train_target, domain_label_target_train)
    loss=loss_s+loss_p
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss_s = self.model._loss(input_train_source, domain_label_source_train)
    loss_t=  self.model._loss(input_train_target, domain_label_target_train)
    loss=loss_s+loss_p
  
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

  def _outer1(self,vector_v_dash, input_train_source,class_label_source_train,input_train_target,class_label_target_train,domain_label_source_train,domain_label_target_train,unrolled_model,unrolled_feature_extractor,unrolled_head_g, eta, r=1e-2):
    R1 = r / _concat(vector_v_dash).norm()
    for p, v in zip(self.feature_extractor.parameters(), vector_v_dash):
      p.data.add_(R1, v)
    domain_features_source,domain_logits_s=unrolled_model(input_train_source)
    domain_features_target,domain_logits_t=unrolled_model(input_train_target)
    generalized_features_source = self.feature_extractor(input_train_source)
    generalized_features_target = self.feature_extractor(input_train_target)
    logits1=self.head_g(generalized_features_source-domain_features_source)
    logits2=self.head_g(generalized_features_target-domain_features_target)
    crit = nn.CrossEntropyLoss()
    loss1 = crit(logits1,class_label_source_train)
    loss2 = crit(logits2, class_label_target_train)
    loss_dash_1=loss1+loss2
  
    vector_w_dash = torch.autograd.grad(loss_dash_1, unrolled_model.parameters())
    grad_part1 = self._hessian_vector_product(vector_w_dash, input_train_source,input_train_target,domain_label_source_train,domain_label_target_train)
    
    for p, v in zip(self.feature_extractor.parameters(), vector_v_dash):
      p.data.sub_(2*R1, v)
    
    domain_features_source,domain_logits_s=unrolled_model(input_train_source)
    domain_features_target,domain_logits_t=unrolled_model(input_train_target)
    generalized_features_source = self.feature_extractor(input_train_source)
    generalized_features_target = self.feature_extractor(input_train_target)
    logits1=self.head_g(generalized_features_source-domain_features_source)
    logits2=self.head_g(generalized_features_target-domain_features_target)
    crit = nn.CrossEntropyLoss()
    loss1 = crit(logits1, class_label_source_train)
    loss2 = crit(logits2, class_label_target_train)
    loss_dash_2=loss1+loss2
    
    
    vector_w_dash = torch.autograd.grad(loss_dash_2, unrolled_model.parameters())
    grad_part2 = self._hessian_vector_product(vector_w_dash, input_train_source,input_train_target,domain_label_source_train,domain_label_target_train)

    for p, v in zip(self.feature_extractor.parameters(), vector_v_dash):
      p.data.add_(R1, v)

    return [(x-y).div_((2*R1)/(eta*eta)) for x, y in zip(grad_part1, grad_part2)]
  
  def _outer2(self,vector_g_dash,input_train_source,class_label_source_train,input_train_target,class_label_target_train,domain_label_source_train,domain_label_target_train,unrolled_model,unrolled_feature_extractor,unrolled_head_g, eta, r=1e-2):
    R1 = r / _concat(vector_g_dash).norm()
    for p, v in zip(self.head_g.parameters(), vector_g_dash):
      p.data.add_(R1, v)
    domain_features_source,domain_logits_s=unrolled_model(input_train_source)
    domain_features_target,domain_logits_t=unrolled_model(input_train_target)
    generalized_features_source = self.feature_extractor(input_train_source)
    generalized_features_target = self.feature_extractor(input_train_target)
    logits1=self.head_g(generalized_features_source-domain_features_source)
    logits2=self.head_g(generalized_features_target-domain_features_target)
    crit = nn.CrossEntropyLoss()
    loss1 = crit(logits1, class_label_source_train)
    loss2 = crit(logits2, class_label_target_train)
    loss_dash_1=loss1+loss2
  
    vector_w_dash = torch.autograd.grad(loss_dash_1, unrolled_model.parameters())
    grad_part1 = self._hessian_vector_product(vector_w_dash, input_train_source,input_train_target,domain_label_source_train,domain_label_target_train)
    
    for p, v in zip(self.head_g.parameters(), vector_g_dash):
      p.data.sub_(2*R1, v)
    
    domain_features_source,domain_logits_s=unrolled_model(input_train_source)
    domain_features_target,domain_logits_t=unrolled_model(input_train_target)
    generalized_features_source = self.feature_extractor(input_train_source)
    generalized_features_target = self.feature_extractor(input_train_target)
    logits1=self.head_g(generalized_features_source-domain_features_source)
    logits2=self.head_g(generalized_features_target-domain_features_target)
    crit = nn.CrossEntropyLoss()
    loss1 = crit(logits1, class_label_source_train)
    loss2 = crit(logits2, class_label_target_train)
    loss_dash_2=loss1+loss2
    
    
    vector_w_dash = torch.autograd.grad(loss_dash_2, unrolled_model.parameters())
    grad_part2 = self._hessian_vector_product(vector_w_dash, input_train_source,input_train_target,domain_label_source_train,domain_label_target_train)

    for p, v in zip(self.head_g.parameters(), vector_g_dash):
      p.data.add_(R1, v)

    return [(x-y).div_((2*R1)/(eta*eta)) for x, y in zip(grad_part1, grad_part2)]


