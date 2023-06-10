import click
import cv2
from mmpose.apis import (init_pose_model, process_mmdet_results,
                         inference_top_down_pose_model, vis_pose_result)
from mmdet.apis import init_detector, inference_detector

from mmpose.core.evaluation.top_down_eval import (keypoint_nme,
                                                  keypoint_pck_accuracy)


def load_mmcv_models():
    pose_config = 'mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/hrnet_w32_mpii_256x256_dark.py'
    pose_checkpoint = 'checkpoints/mmcv/hrnet_w32_mpii_256x256_dark-f1601c5b_20200927.pth'
    det_config = 'mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
    det_checkpoint = 'checkpoints/mmcv/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    # initialize pose model
    pose_model = init_pose_model(pose_config, pose_checkpoint)
    # initialize detector
    det_model = init_detector(det_config, det_checkpoint)

    return pose_model, det_model

@click.command()
@click.pass_context
@click.option('--video_path', help='Path to rendered video', metavar='PATH', required=True)
@click.option('--smpl_path', help='Path to SMPL params', metavar='PATH', required=True)


def calc_poseconst_metrics(ctx, video_path, smpl_path):
    pose_model, det_model = load_mmcv_models()

    img = 'mmpose/tests/data/coco/000000196141.jpg'

    mmdet_results = inference_detector(det_model, img)
    person_results = process_mmdet_results(mmdet_results, cat_id=1)

    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        img,
        person_results,
        bbox_thr=0.3,
        format='xyxy',
        dataset=pose_model.cfg.data.test.type)

    vis_result = vis_pose_result(
        pose_model,
        img,
        pose_results,
        dataset=pose_model.cfg.data.test.type,
        show=False)
    vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)
    cv2.imwrite('test.png', vis_result)

    breakpoint()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    calc_poseconst_metrics() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------