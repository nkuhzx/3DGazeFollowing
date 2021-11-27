import os

import numpy as np
from PIL import Image

import pandas as pd

import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from utils import img_utils

from gazefollowmodel.models.gazenet import GazeNet


class Infer_engine(object):

    def __init__(self,rgb_transform=None,depth_transform=None,input_size=224,device="cuda",checkpoint=""):
        super(Infer_engine, self).__init__()

        # define torch transform
        if rgb_transform is None:

            transform_list = []
            transform_list.append(transforms.Resize((input_size, input_size)))
            transform_list.append(transforms.ToTensor())
            transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

            self.rgb_transform=transforms.Compose(transform_list)
        else:
            self.rgb_transform=rgb_transform

        if depth_transform is None:

            self.depth_transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
            ])
        else:
            self.depth_transform=depth_transform

        # define input size
        self.input_size=input_size


        self.device=device
        self.checkpoint=checkpoint

        self.format_input=dict()
        self.model_output=dict()
        self.infer_output=dict()

        self.eval=dict()

        self.ground_truth=dict()

        self.model=self.model_init()

        # init the model

    def model_init(self):
        cudnn.deterministic = True

        model=GazeNet(pretrained=False)
        model=model.to(self.device)
        model.eval()

        checkpoint = torch.load(self.checkpoint)
        model.load_state_dict(checkpoint)

        return model


    def format_model_input(self,rgb_path,depth_path,head_bbox,campara,eye_coord):


        # load the rgbimg and depthimg
        rgbimg=Image.open(rgb_path)
        rgbimg = rgbimg.convert('RGB')

        width, height = rgbimg.size

        depthimg = Image.open(depth_path)

        depthvalue=np.array(depthimg.copy())
        depthvalue.flags.writeable=True
        depthvalue=depthvalue/1000.0


        # expand the head bounding box (in pixel coordinate )
        x_min, y_min, x_max, y_max=map(float,img_utils.expand_head_box(head_bbox
                                                             ,[width,height]))


        self.img_para = [width, height]

        head = rgbimg.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        self.head_loc=[]
        head_channel = img_utils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                    resolution=self.input_size, coordconv=False).unsqueeze(0)


        rgbimg=self.rgb_transform(rgbimg)
        headimg=self.rgb_transform(head)
        depthimg=self.depth_transform(depthimg)

        # obtain the trasformation matrix from camera coordinate system to eye coordinate system
        T_c2e=self.getTransmatrix(eye_coord)

        # obatain the gazevector in eye coordinate system
        depthmap=depthimg.numpy()/1000.0
        depthmap=depthmap[0]

        Gaze_Vector_Space=self.getGazevectorspace(depthmap,campara,T_c2e)


        self.format_input['rgbimg']=rgbimg
        self.format_input['headimg']=headimg
        self.format_input['headchannel']=head_channel
        self.format_input['gvs']=Gaze_Vector_Space
        self.format_input['depthmap']=depthvalue
        # self.format_input=[rgbimg,headimg,head_channel,Gaze_Vector_Space]

    def run_3DGazeFollowModel(self):

        rgb_img= self.format_input['rgbimg']
        h_img=self.format_input['headimg']
        h_channel=self.format_input['headchannel']
        gvs=self.format_input['gvs']

        x_rgbimg=rgb_img.to(self.device).unsqueeze(0)
        x_himg=h_img.to(self.device).unsqueeze(0)
        x_hc=h_channel.to(self.device).unsqueeze(0)
        x_gvs=gvs.to(self.device).unsqueeze(0)


        pred_heatmap, pred_gazevector = self.model(x_rgbimg, x_gvs, x_himg, x_hc)

        pred_heatmap=pred_heatmap.squeeze()
        pred_heatmap = pred_heatmap.data.cpu().numpy()

        pred_gazevector=pred_gazevector.data.cpu().numpy()

        self.model_output["pred_heatmap"]=pred_heatmap
        self.model_output["pred_gazevector"]=pred_gazevector


    def inference(self,eye_3dim,campara,ratio=0.1):

        #
        img_W,img_H=self.img_para
        #

        pred_heatmap=self.model_output["pred_heatmap"]
        pred_gazevector=self.model_output["pred_gazevector"]

        depthmap=self.format_input["depthmap"]

        img_H,img_W=depthmap.shape

        # get the center of 2d proposal area
        output_h,output_w=pred_heatmap.shape

        pred_center = list(img_utils.argmax_pts(pred_heatmap))

        pred_center[0]=pred_center[0]*img_W/output_w
        pred_center[1]=pred_center[1]*img_H/output_h


        pu_min=pred_center[0]-img_W*ratio/2
        pu_max=pred_center[0]+img_W*ratio/2

        pv_min=pred_center[1]-img_H*ratio/2
        pv_max=pred_center[1]+img_H*ratio/2

        if pu_min < 0:
            pu_min, pu_max = 0, img_W * ratio
        elif pu_max > img_W:
            pu_max, pu_min = img_W, img_W - img_W * ratio

        if pv_min < 0:
            pv_min, pv_max = 0, img_H * ratio
        elif pv_max > img_H:
            pv_max, pv_min = img_H, img_H - img_H * ratio

        pu_min,pu_max,pv_min,pv_max=map(int,[pu_min,pu_max,pv_min,pv_max])

        self.rectangle=[pu_min,pu_max,pv_min,pv_max]

        # unproject to 3d proposal area
        range_depthmap=depthmap[pv_min:pv_max,pu_min:pu_max]
        cx,cy,fx,fy,rm=campara

        range_space_DW = np.linspace(pu_min, pu_max - 1, pu_max - pu_min)
        range_space_DH = np.linspace(pv_min, pv_max - 1, pv_max - pv_min)
        [range_space_xx, range_space_yy] = np.meshgrid(range_space_DW, range_space_DH)

        range_space_X = (range_space_xx - cx) * range_depthmap / fx
        range_space_Y = (range_space_yy - cy) * range_depthmap / fy
        range_space_Z = range_depthmap

        proposal_3d=np.dstack([range_space_X,range_space_Y,range_space_Z])
        proposal_3d=proposal_3d.reshape([-1,3])
        proposal_3d=np.dot(rm,proposal_3d.T)
        proposal_3d=proposal_3d.T
        proposal_3d=proposal_3d.reshape([pv_max-pv_min,pu_max-pu_min,3])

        # trans from camera coordinate system to eye system
        T_c2e=self.getTransmatrix(eye_3dim)
        ones_np=np.ones((pv_max-pv_min,pu_max-pu_min,1))
        proposal_3d=np.concatenate([proposal_3d,ones_np],axis=2)
        proposal_3d=proposal_3d.reshape(-1,4)
        proposal_3d=proposal_3d.T
        proposal_3d=np.dot(T_c2e,proposal_3d)
        proposal_3d=proposal_3d.T
        proposal_3d=proposal_3d.reshape(pv_max-pv_min,pu_max-pu_min,4)

        gaze_vector_set=proposal_3d[:,:,:3]-0

        norm_value = np.linalg.norm(gaze_vector_set, axis=2, keepdims=True)
        norm_value[norm_value <= 0] = 1
        gaze_vector_set=gaze_vector_set/norm_value

        gaze_vector_set[range_depthmap==0]=0

        gaze_vector_similar_set=np.dot(gaze_vector_set,pred_gazevector)
        max_index_u,max_index_v=img_utils.argmax_pts(gaze_vector_similar_set)

        pred_gazetarget_eye=proposal_3d[int(max_index_v),int(max_index_u)]


        # in eye coordinate system
        self.infer_output["pred_gavetarget_e"]=pred_gazetarget_eye[:3]
        self.infer_output["pred_gazevector_e"]=gaze_vector_set[int(max_index_v),int(max_index_u)]-0

        # in camera coordinate system
        # obtain the inverse transformation matrix

        T_e2c=T_c2e.copy()
        T_e2c[:3,:3]=T_e2c[:3,:3].T
        T_e2c[:,3]=np.append(eye_3dim,1)

        pred_gazetarget_camera=np.dot(T_e2c,pred_gazetarget_eye)[:3]
        pred_gazevector_camera=pred_gazetarget_camera-eye_3dim
        pred_gazevector_camera/(np.linalg.norm(pred_gazevector_camera)+1e-6)

        self.infer_output["pred_gavetarget_c"]=pred_gazetarget_camera
        self.infer_output["pred_gazevector_c"]=pred_gazevector_camera


    def evaluation(self):

        #
        gt_gazevector_eye=self.ground_truth["gaze_vector_e"]
        gt_gazetarget_eye=self.ground_truth["gaze_target3d_e"]

        pred_gazevector_eye=self.infer_output["pred_gazevector_e"]
        pred_gazetarget_eye=self.infer_output["pred_gavetarget_e"]

        # angle error
        pred_cosine_similarity=np.sum(gt_gazevector_eye*pred_gazevector_eye)
        angle_error=np.arccos(pred_cosine_similarity)
        angle_error=np.rad2deg(angle_error)

        # dist error
        dist=np.linalg.norm(pred_gazetarget_eye-gt_gazetarget_eye)

        self.eval["angle_error"]=angle_error
        self.eval["dist_error"]=dist

    def getGroundTruth(self,aux_info):

        width,height=self.img_para

        # in camera coordinate system
        eye_3dim, gaze_3dim,gaze_2dim = aux_info
        gaze_vector=gaze_3dim-eye_3dim

        gaze_u,gaze_v=gaze_2dim

        gaze_2dim=np.array([gaze_u,gaze_v])

        T_c2e=self.getTransmatrix(eye_3dim)

        gaze_vector_eye=np.dot(T_c2e[:3,:3],gaze_vector)
        gaze_3dim_eye=np.append(gaze_3dim,1)
        gaze_3dim_eye=np.dot(T_c2e,gaze_3dim_eye)[0:3]

        norm_value =1e-6 if np.linalg.norm(gaze_vector)<=0 else np.linalg.norm(gaze_vector)
        gaze_vector=gaze_vector/norm_value

        norm_value =1e-6 if np.linalg.norm(gaze_vector_eye)<=0 else np.linalg.norm(gaze_vector_eye)
        gaze_vector_eye=gaze_vector_eye/norm_value

        self.ground_truth['gaze_vector_e']=gaze_vector_eye
        self.ground_truth['gaze_target3d_e']=gaze_3dim_eye

        self.ground_truth["gaze_vector"]=gaze_vector
        self.ground_truth["gaze_target3d"]=gaze_3dim

        self.ground_truth['gaze_target2d']=gaze_2dim

    def getTransmatrix(self,eye_coord):
        upVector = np.array([0, -1, 0], np.float32)
        zAxis=eye_coord.flatten()
        xAxis = np.cross(upVector, zAxis)
        xAxis /= np.linalg.norm(xAxis)
        yAxis = np.cross(zAxis, xAxis)
        yAxis /= np.linalg.norm(yAxis)
        # the transform from camera coordinate to eye coordinate
        Rotation_c2e = np.stack([xAxis, yAxis, zAxis], axis=0)

        Trans_c2e=np.zeros((4,4))
        Trans_c2e[:3,:3]=Rotation_c2e
        Trans_c2e[:3,3]=-np.dot(Rotation_c2e,eye_coord)
        Trans_c2e[3,3]=1

        return Trans_c2e

    def getGazevectorspace(self,dmap,camera_p,Trans_T):

        img_W,img_H=self.img_para

        cx,cy,fx,fy,R_r2d=camera_p

        gaze_space_DW = np.linspace(0, self.input_size - 1, self.input_size)
        gaze_space_DH = np.linspace(0, self.input_size - 1, self.input_size)
        [gaze_space_xx, gaze_space_yy] = np.meshgrid(gaze_space_DW, gaze_space_DH)

        scale_width, scale_height = img_W / self.input_size, img_H / self.input_size

        gaze_vector_space_X = (gaze_space_xx*scale_width - cx) * dmap /fx

        gaze_vector_space_Y = (gaze_space_yy*scale_height  - cy) * dmap /fy
        gaze_vector_space_Z = dmap

        gaze_vector_space = np.dstack((gaze_vector_space_X, gaze_vector_space_Y, gaze_vector_space_Z))

        gaze_vector_space = gaze_vector_space.reshape([-1, 3])

        gaze_vector_space = np.dot(R_r2d, gaze_vector_space.T)
        gaze_vector_space = gaze_vector_space.T
        gaze_vector_space = gaze_vector_space.reshape([self.input_size, self.input_size, 3])

        ones_np=np.ones((self.input_size,self.input_size,1))
        gaze_vector_space=np.concatenate([gaze_vector_space,ones_np],axis=2)
        gaze_vector_space=gaze_vector_space.reshape(-1,4)
        gaze_vector_space=gaze_vector_space.T
        gaze_vector_space=np.dot(Trans_T,gaze_vector_space)
        gaze_vector_space=gaze_vector_space.T
        gaze_vector_space=gaze_vector_space.reshape(self.input_size,self.input_size,4)

        # same as gaze_vector_space=gaze_vector_space[:,:,:3]-0
        gaze_vector_space=gaze_vector_space[:,:,:3]

        norm_value = np.linalg.norm(gaze_vector_space, axis=2, keepdims=True)
        norm_value[norm_value <= 0] = 1

        gaze_vector_space=gaze_vector_space/norm_value

        gaze_vector_space=torch.from_numpy(gaze_vector_space)
        gaze_vector_space=gaze_vector_space.float()
        # gaze_vector_space=gaze_vector_space.permute((2,0,1))

        return gaze_vector_space
