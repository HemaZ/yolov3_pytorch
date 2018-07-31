from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np  

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer,self).__init__()
       


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
    prev_filters=3
    module_list=nn.ModuleList()
    output_filters=[]

    for index,x in enumerate(blocks[1:]):
        module=nn.Sequential()

        if x["type"]=="convolutional":
            kernel_size=int(x['size'])
            stride=int(x['stride'])
            pad=int(x['pad'])
            activation=x['activation']
            filters=int(x['filters'])
            try:
                batch_normalize=int(x['batch_normalize'])
                bias=False
            except:
                batch_normalize=0
                bias=True
            if pad:
                padding=(kernel_size-1)//2
            else:
                padding=0
            conv=nn.Conv2d(prev_filters,filters,kernel_size,stride,padding,bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            if batch_normalize:
                bn=nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
            if activation=='leaky':
                actv=nn.LeakyReLU(0.1,inplace=True)
                module.add_module("leaky_{0}".format(index), actv)
        elif x['type']=='upsample':
            stride=int(x['stride'])
            upsample=nn.Upsample(scale_factor=2,mode='bilinear')
            module.add_module("upsample_{}".format(index), upsample)
        elif x['type']=='route':
            x['layers']=x['layers'].split(',')
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            if start > 0: 
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
        elif x['type']=='yolo':
            mask=x['mask'].split(',')
            mask=[int(x) for x in mask]
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
        
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    
    return (net_info, module_list)



