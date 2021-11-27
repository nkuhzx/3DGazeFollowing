import os
import argparse

import numpy as np
import torch

import pandas as pd

from tqdm import tqdm

from utils.infer_eigen import Infer_engine
from utils.visual_tools import VisualTool
from utils.dataconst import RGBDAttention

class AverageMeter():

    def __init__(self):

        self.reset()

    def reset(self):

        self.count=0
        self.newval=0
        self.sum=0
        self.avg=0

    def update(self,newval,n=1):

        self.newval=newval
        self.sum+=newval*n
        self.count+=n
        self.avg=self.sum/self.count


def overview(checkpoint,device,visualize=True):

    rgbdatt_const=RGBDAttention()
    infer_instance=Infer_engine(checkpoint=checkpoint,device=device)

    if visualize:
        visual_instance=VisualTool()

    rgbdatt_pd=pd.read_csv(rgbdatt_const.annofile)

    avg_dist_counter=AverageMeter()
    avg_angle_counter=AverageMeter()

    avg_angle_counter.reset()
    avg_dist_counter.reset()

    pbar=tqdm(total=len(rgbdatt_pd))
    for index, row in rgbdatt_pd.iterrows():

        dataset_id=row["dataset_id"]
        frame_index=row["frame_index"]

        # the path of rgbimg and depthimg
        rgbimg_path=os.path.join(rgbdatt_const.rgb_dir,"{}/RGB_{}.png".format(dataset_id,frame_index))

        depthimg_path=os.path.join(rgbdatt_const.depth_dir,"{}/Depth_{}.png".format(dataset_id,frame_index))

        # the head bounding box
        head_bbox=np.array(row["x_initial":"h"].tolist())
        # the eye coordinate in camera coordinate system
        eye_3d=np.array(row["eye_X":"eye_Z"].tolist())

        # gaze target in camera coordinate system
        gaze_2d=np.array(row["gaze_x":"gaze_y"].tolist())
        gaze_3d=np.array(row["gaze_X":"gaze_Z"].tolist())

        infer_instance.format_model_input(rgbimg_path,depthimg_path,head_bbox,rgbdatt_const.camera_para,eye_3d)

        infer_instance.run_3DGazeFollowModel()

        infer_instance.inference(eye_3d,rgbdatt_const.camera_para)

        infer_instance.getGroundTruth([eye_3d,gaze_3d,gaze_2d])
        # infer_instance.visualization()
        infer_instance.evaluation()

        if visualize:

            visual_instance.constructPcd(rgbimg_path,depthimg_path,rgbdatt_const.camera_para)
            visual_instance.constructGazeline(eye_3d,
                                              infer_instance.ground_truth["gaze_target3d"],
                                              infer_instance.infer_output["pred_gavetarget_c"]
                                              )
            visual_instance.visualization()

        avg_dist_counter.update(infer_instance.eval['dist_error'])
        avg_angle_counter.update(infer_instance.eval['angle_error'])

        pbar.set_postfix(eval_avg_dist=avg_dist_counter.avg,
                         eval_angle_dist=avg_angle_counter.avg,
                         dataset_id=dataset_id,
                         frame_index=frame_index)

        pbar.update(1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="3D GazeFollow Evaluation"
    )
    parser.add_argument(
        "--modelweight",
        "-m",
        default="modelpara/3DGazefollowNetPara.pt",
        metavar="FILE",
        help="path to file of model weight",
        type=str,
    )
    parser.add_argument(
        "--cpu",
        "-c",
        action="store_true",
        default=False,
        help="choose whether to use cpu"
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        default=False,
        help="choose whether to visualize"
    )

    args = parser.parse_args()

    device="cuda" if not args.cpu and torch.cuda.is_available() else "cpu"

    print("The evaluation run on [{}]".format(device))

    if not os.path.exists(args.modelweight):

        raise NotImplementedError("The model weight is not exist!")

    overview(args.modelweight,device,visualize=args.visualize)
