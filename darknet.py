from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np  


def parse_cfg(cfgfile):
    """"
    This Function takes Darknet Cfg file and returns 
    CONVNEt Blocks, each block is represented as dicttionary
    """
    file=open(cfgfile,'r')
    lines=file.read().split('\n')
    lines=[x for x in lines if len(x)>0 and x[0]!='#']
    lines = [x.rstrip().lstrip() for x in lines]  
    file.close()
    del(file)
    file=open('cfg/cfgout.txt','w')
    file.write(str(lines))
    file.close()


    block={}
    blocks=[]
    
    for line in lines:
        if line[0]=='[':
            if len(block)!=0:
                blocks.append(block)
                block={}
            block["type"]=line[1:-1].rstrip()
        else:
            key,value=line.split('=')
            block[key.rstrip()]=value.lstrip()
    
    blocks.append(block)
    return blocks


def create_modules(blocks):
    net_info=blocks[0]
