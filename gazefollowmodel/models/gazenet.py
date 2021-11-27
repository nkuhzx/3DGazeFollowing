import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from gazefollowmodel.models.headpath import HeadEncoder
from gazefollowmodel.models.gazepath import GazeEncoder,CoarseDecoder,FineDecoder
from gazefollowmodel.models.saliencypath import SceneEncoder,SceneDecoder

MODEL_PARA="/root/Desktop/attention_model_new/gazefollowmodel/model_para"


class GazeNet(nn.Module):

    def __init__(self,pretrained=False):

        super(GazeNet,self).__init__()

        self.avgpool=nn.AvgPool2d(kernel_size=7,stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.saliency_ec=SceneEncoder()

        self.gaze_ec=GazeEncoder(pretrained=True)

        self.head_ec=HeadEncoder()

        self.attFace=nn.Linear(1024+784,1*7*7)

        self.heatmap_dc=SceneDecoder()

        self.coarse_dc=CoarseDecoder()

        self.fine_dc=FineDecoder()

        self.relu=nn.ReLU()


    def forward(self,simg,gaze_vector_space,face,headloc):

        # batch_size
        bs=simg.size(0)
        wei,hei=simg.size(2),simg.size(3)

        ## head pathway
        # extract the head feature
        _,head_feats=self.head_ec(face)

        # obtain the attention map
        headfeats_reduced = self.avgpool(head_feats).view(bs, -1)
        headloc_reduced = self.maxpool(self.maxpool(self.maxpool(headloc))).view(bs, -1)
        headfeats_with_headloc_feats=torch.cat([headfeats_reduced,headloc_reduced],dim=1)
        alphaFace=F.softmax(self.attFace(headfeats_with_headloc_feats).view(bs,1,49),dim=2)
        alphaFace=alphaFace.view(bs,1,7,7)

        ## 3d gaze pathway
        # extract the 3d gaze feature
        gaze_feats=self.gaze_ec(torch.cat([gaze_vector_space.permute(0, 3, 1, 2), face, headloc.detach()], dim=1))

        # obtain the coarsed 3d gaze vector
        coarse_dir_xyz=self.coarse_dc(gaze_feats)

        # form the gaze_heatmap
        gaze_vector_space=gaze_vector_space.reshape([bs,-1,3])
        gaze_vector_space_heatmap=torch.matmul(gaze_vector_space,coarse_dir_xyz.unsqueeze(2))
        gaze_vector_heatmap=gaze_vector_space_heatmap.reshape(bs,1,wei,hei)
        # gaze_vector_heatmap=torch.pow(gaze_vector_heatmap,2)
        # gaze_vector_heatmap=torch.pow
        ## saliency pathway
        # extract the saliency feature
        saliency_feats = self.saliency_ec(torch.cat((simg, gaze_vector_heatmap, headloc), dim=1))

        # obtain the saliency feature with attention map
        saliency_att_fus = torch.mul(alphaFace, saliency_feats)


        ## two prediction brach
        # gaze heatmap prediction branch
        concat_feats_one = torch.cat([saliency_att_fus, head_feats], dim=1)
        gaze_heatmap = self.heatmap_dc(concat_feats_one)

        # 3d gaze vector prediction branch
        concat_feats_two=torch.cat([saliency_feats, gaze_feats, head_feats], dim=1)
        fine_dir_xyz=self.fine_dc(concat_feats_two)

        return gaze_heatmap, fine_dir_xyz.squeeze()


