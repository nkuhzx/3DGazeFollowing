import torch
import torch.nn as nn
from gazefollowmodel.utils.resnet import resnet50,Bottleneck

class GazeEncoder(nn.Module):


    def __init__(self,pretrained=False):

        super(GazeEncoder, self).__init__()

        org_resnet=resnet50(pretrained)

        self.conv1=nn.Conv2d(7,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=org_resnet.bn1
        self.relu=org_resnet.relu
        self.maxpool=org_resnet.maxpool
        self.layer1=org_resnet.layer1
        self.layer2=org_resnet.layer2
        self.layer3=org_resnet.layer3
        self.layer4=org_resnet.layer4
        # add
        self.layer5=self._make_layer(Bottleneck,org_resnet.inplanes,256,2,stride=1)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample =  None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self,x):

        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)

        return x


class CoarseDecoder(nn.Module):

    def __init__(self):
        super(CoarseDecoder, self).__init__()

        self.depth_compress=nn.Sequential(
            nn.Conv2d(1024,512,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,1,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        self.dirxyz_fc=nn.Linear(49, 3)


    def forward(self,feats):

        bs=feats.size(0)

        feats=self.depth_compress(feats)

        directionxyz=feats.view(bs,-1)
        directionxyz=self.dirxyz_fc(directionxyz)

        norm = torch.norm(directionxyz, 2, dim=1)
        norm_=norm.clone()
        norm_[norm_<=0]=1.0
        normalized_directionxyz=directionxyz/norm_.view([-1,1])


        return normalized_directionxyz #,depth_value

class FineDecoder(nn.Module):

    def __init__(self):
        super(FineDecoder,self).__init__()

        self.depth_compress=nn.Sequential(
            nn.Conv2d(3072,512,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        self.dirxyz_fc=nn.Linear(49, 3)

    def forward(self,feats):

        bs=feats.size(0)

        feats=self.depth_compress(feats)

        directionxyz=feats.view(bs,-1)


        directionxyz=self.dirxyz_fc(directionxyz)
        norm = torch.norm(directionxyz, 2, dim=1)
        norm_=norm.clone()
        norm_[norm_<=0]=1.0
        normalized_directionxyz=directionxyz/norm_.view([-1,1])


        return normalized_directionxyz #,depth_value

