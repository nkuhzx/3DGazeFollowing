import torch.nn as nn
import torch
from gazefollowmodel.utils.resnet import resnet50,Bottleneck

class SceneEncoder(nn.Module):


    def __init__(self,pretrained=False):

        super(SceneEncoder,self).__init__()

        org_resnet=resnet50(pretrained)

        self.conv1=nn.Conv2d(5,64,kernel_size=7,stride=2,padding=3,bias=False)
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


class SceneDecoder(nn.Module):

    def __init__(self):
        super(SceneDecoder,self).__init__()


        self.relu=nn.ReLU(inplace=True)

        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)



    def forward(self,scene_face_feat):



        encoding = self.compress_conv1(scene_face_feat)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)

        x = self.deconv1(encoding)
        x = self.deconv_bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.deconv_bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.relu(x)
        x = self.conv4(x)

        return x


