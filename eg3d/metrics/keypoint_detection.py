import numpy as np
import scipy.linalg
from . import metric_utils

from mmpose.apis import (init_pose_model, process_mmdet_results,
                         inference_top_down_pose_model, vis_pose_result)
from mmdet.apis import init_detector, inference_detector

from mmpose.core.evaluation.top_down_eval import (keypoint_nme,
                                                  keypoint_pck_accuracy)
import copy
import torch
import cv2
from tqdm.autonotebook import tqdm
from SPIN import process_EG3D_image
import consts

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (PerspectiveCameras,
                                RasterizationSettings, PointLights,
                                MeshRasterizer, MeshRenderer,
                                HardPhongShader, TexturesVertex)

#----------------------------------------------------------------------------

def compute_pck(opts, max_real, num_gen):
    pose_model, det_model = load_mmcv_models()

    pck = compute_pck_for_dataset(opts=opts, max_items=num_gen, pose_model=pose_model, det_model=det_model)
    if opts.rank != 0:
        return float('nan')
    return float(pck)

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

def compute_kpts(images, pose_model, det_model, mode):
    kps = []
    failed = 0
    for i in range(images.shape[0]):
        img = images[i].permute(1,2,0).cpu().numpy()

        mmdet_results = inference_detector(det_model, img)
        person_results = process_mmdet_results(mmdet_results, cat_id=1)

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=0.3,
            format='xyxy',
            dataset=pose_model.cfg.data.test.type)

        # debug test to see how accurate kpt estimation isS
        if False:
            vis_result = vis_pose_result(
                pose_model,
                img,
                pose_results,
                dataset=pose_model.cfg.data.test.type,
                show=False)
            vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)
            cv2.imwrite(f'test_{mode}.png', vis_result)

        # no human even detected:
        if len(pose_results) == 0:
            kps.append(-1e8 * np.ones((16, 3)))
            failed = 1
        else:
            kps.append(pose_results[0]['keypoints'])

    return np.stack(kps, 0), failed

def get_mesh_img_gt(c, G):
    ishape = 128 if G.rendering_kwargs['cfg_name'] == 'surreal' else 256
    smpl_params = c[:, 25:107]
    smpl_out_mvc_canon = G.renderer.smpl_reduced.forward(
        betas=smpl_params[:,72:],
        body_pose=smpl_params[:,3:72],
        global_orient=smpl_params[:,:3],
        transl=G.renderer.smpl_avg_transl)
    smpl_out_mvc_canon.vertices *= (G.renderer.smpl_avg_scale * .9)

    verts_pose = copy.deepcopy(smpl_out_mvc_canon.vertices)
    cur_meshes_pose = Meshes(
        verts_pose,
        faces=torch.from_numpy(
            G.renderer.smpl_reduced.faces.astype('int64')).view(
            1, -1, 3).expand(verts_pose.shape[0], -1, -1).to(verts_pose.device)
    )

    pose = c[:,:16].view(-1,4,4)
    intrinsics = c[:,16:25].view(-1,3,3)

    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
    world2cam = pose.clone()
    world2cam[:, [0, 1]] *= -1
    world2cam = world2cam.inverse()
    cameras = PerspectiveCameras(focal_length=2 * intrinsics[:, 0, 0],
                                 principal_point=[[0, 0]],
                                 device=c.device,
                                 R=world2cam[:, :3, :3],
                                 T=world2cam[:, :3, 3])

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
    # -z direction.
    lights = PointLights(device=c.device, location=pose[:, :3, 3])

    raster_settings = RasterizationSettings(
        image_size=ishape,
        blur_radius=0.0,
        faces_per_pixel=1,
        )
    mesh_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras,
                                  raster_settings=raster_settings),
        shader=HardPhongShader(device=c.device,
                               cameras=cameras,
                               lights=lights))

    # Initialize each vertex to be rand in color.
    # breakpoint()
    verts_rgb = torch.rand_like(cur_meshes_pose.verts_padded())  # (1, V, 3)
    # verts_rgb[0, :100, :] = torch.Tensor([0., 0., 1.]).cuda()
    # verts_rgb[0, 100:200, :] = torch.Tensor([0., 1., 0.]).cuda()
    # verts_rgb[0, 200:300, :] = torch.Tensor([1., 0., 0.]).cuda()
    # verts_rgb[0, 300:400, :] = torch.Tensor([1., 0., 1.]).cuda()
    # verts_rgb[0, 400:500, :] = torch.Tensor([1., 1., 0.]).cuda()
    # verts_rgb[0, 500:600, :] = torch.Tensor([0., 1., 1.]).cuda()
    # verts_rgb[0, 600:, :] = torch.Tensor([1., 0., 1.]).cuda()

    textures = TexturesVertex(verts_features=verts_rgb.to(c.device))
    cur_meshes_pose.textures = textures
    mesh_imgs = mesh_renderer(cur_meshes_pose,
                              lights=lights,
                              cameras=cameras)[:, :, :, :3]
    mesh_imgs = (mesh_imgs * 255).to(torch.uint8)
    mesh_imgs[mesh_imgs == 255] = 0

    return mesh_imgs.permute(0,3,1,2)

# Returns intrinsic camera matrix
# Parameters are hard-coded since all SURREAL images use the same.
def get_intrinsic():
    # These are set in Blender (datageneration/main_part1.py)
    res_x_px = 320  # *scn.render.resolution_x
    res_y_px = 240  # *scn.render.resolution_y
    f_mm = 60  # *cam_ob.data.lens
    sensor_w_mm = 32  # *cam_ob.data.sensor_width
    sensor_h_mm = sensor_w_mm * res_y_px / res_x_px  # *cam_ob.data.sensor_height (function of others)
    scale = 1  # *scn.render.resolution_percentage/100
    skew = 0  # only use rectangular pixels
    pixel_aspect_ratio = 1    # From similar triangles:
    # sensor_width_in_mm / resolution_x_inx_pix = focal_length_x_in_mm / focal_length_x_in_pix
    fx_px = f_mm * res_x_px * scale / sensor_w_mm
    fy_px = f_mm * res_y_px * scale * pixel_aspect_ratio / sensor_h_mm    # Center of the image
    u = res_x_px * scale / 2
    v = res_y_px * scale / 2    # Intrinsic camera matrix
    K = np.array([[fx_px, skew, u], [0, fy_px, v], [0, 0, 1]])
    return K# Returns extrinsic camera matrix
#   T : translation vector from Blender (*cam_ob.location)
#   RT: extrinsic computer vision camera matrix
#   Script based on https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
def get_extrinsic(T):
    # Take the first 3 columns of the matrix_world in Blender and transpose.
    # This is hard-coded since all images in SURREAL use the same.
    R_world2bcam = np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]).transpose()
    # *cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']),
    #                               (0., -1, 0., -1.0),
    #                               (-1., 0., 0., 0.),
    #                               (0.0, 0.0, 0.0, 1.0)))    # Convert camera location to translation vector used in coordinate changes
    T_world2bcam = -1 * np.dot(R_world2bcam, T)    # Following is needed to convert Blender camera to computer vision camera
    R_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = np.dot(R_bcam2cv, R_world2bcam)
    T_world2cv = np.dot(R_bcam2cv, T_world2bcam)    # Put into 3x4 matrix
    RT = np.concatenate([R_world2cv, T_world2cv], axis=1)
    return RT, R_world2cv, T_world2cv

import torchvision.ops
import PIL

def bbox_from_segm(seg):
    # bbox is (xmin, ymin, xmax, ymax)
    bin_seg = (seg > 0)
    bbox = torchvision.ops.masks_to_boxes(torch.from_numpy(bin_seg).unsqueeze(0))
    return bbox[0]

def cropresize(frame, seg):
    bbox = bbox_from_segm(seg)
    h = int(bbox[3]) - int(bbox[1])
    w = int(bbox[2]) - int(bbox[0])

    # get a square crop of the image
    if w < h:
        ls = max(0, int(bbox[0]-np.floor((h-w)/2)))
        if ls + h < frame.shape[1]:
            rs = ls + h
        else:
            rs = frame.shape[1]
            ls = rs - h

        image_crop_sq = frame[int(bbox[1]):int(bbox[3]), ls:rs, :]
        seg_crop_sq = seg[int(bbox[1]):int(bbox[3]), ls:rs]

        new_crop = int(bbox[1]), int(bbox[3]), int(ls), int(rs)
    else:
        ls = max(0, int(bbox[1]-np.floor((h-w)/2)))
        if ls + w < frame.shape[0]:
            rs = ls + w
        else:
            rs = frame.shape[0]
            ls = rs - w

        image_crop_sq = frame[ls:rs, int(bbox[0]):int(bbox[2]), :]
        seg_crop_sq = seg[ls:rs, int(bbox[0]):int(bbox[2])]

        new_crop = int(ls), int(rs), int(bbox[0]), int(bbox[2])

    new_crop = np.array(new_crop)
    orig_size = image_crop_sq.shape[0]
    img_resize = np.array(PIL.Image.fromarray(image_crop_sq).resize((128, 128)))
    seg_resize = np.array(PIL.Image.fromarray(seg_crop_sq).resize((128, 128)))
    # img_resize = image_crop_sq
    # seg_resize = None

    return img_resize, seg_resize, orig_size, new_crop

def get_surreal_data(G):
    from pathlib import Path
    import skvideo.io
    import scipy.io
    import imageio
    import math
    import transforms3d

    def get_frame_info(video_info, frame_no):
        frame_info = {}
        for key in video_info:
            if key in ['camDist', 'camLoc', 'clipNo', 'sequence', 'source', 'stride']:
                frame_info[key] = video_info[key]
            elif key in ['bg', 'cloth', 'joints2D', 'joints3D', 'light', 'pose', 'shape']:
                frame_info[key] = video_info[key][..., frame_no]
            elif key in ['gender', 'zrot']:
                frame_info[key] = video_info[key][frame_no, ...]
        return frame_info

    def rotateBody(RzBody, pelvisRotVec):
        angle = np.linalg.norm(pelvisRotVec)
        Rpelvis = transforms3d.axangles.axangle2mat(pelvisRotVec / angle, angle)
        globRotMat = np.dot(RzBody, Rpelvis)
        R90 = transforms3d.euler.euler2mat(np.pi / 2, 0, 0)
        globRotAx, globRotAngle = transforms3d.axangles.mat2axangle(np.dot(R90, globRotMat))
        globRotVec = globRotAx * globRotAngle
        return globRotVec

    def project_vertices(pts, intrinsic, extrinsic):
        homo_coords = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1).transpose()
        proj_coords = np.dot(intrinsic, np.dot(extrinsic, homo_coords))
        proj_coords = proj_coords / proj_coords[2]
        proj_coords = proj_coords[:2].transpose()
        return proj_coords

    def project_eg3d(pts, intrinsic, extrinsic, output_resolution):
        """
        It assumes that the output pixels are in range [0,1]. Pretty weird but if this is what it takes.
        """
        pts_2d_mine = intrinsic[None, ...] @ ((extrinsic[None, :3, :3] @ pts[..., None]) + extrinsic[None, :3, 3:])
        pts_2d_mine = pts_2d_mine[..., :2, 0] / pts_2d_mine[..., 2:, 0]  # Should be -1,1
        pts_2d_mine = pts_2d_mine * output_resolution[None]
        return pts_2d_mine

    _COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
               [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
               [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
               [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
               [255, 0, 170], [255, 0, 85]]

    def plot_kpt(keypoint, canvas, color=None):
        for i, (x, y) in enumerate(keypoint[:, 0:2]):
            if np.isnan(x) or np.isnan(y) or x < 0 or y < 0:
                continue
            cv2.circle(canvas, (int(x), int(y)),
                       1,
                       color if color is not None else _COLORS[i % len(_COLORS)],
                       thickness=-1)
        return canvas

    # GET INFO
    # fname = Path('/home/awb/Data/hgan/surreal/SURREAL/data/cmu/test/run0/104_24/104_24_c0001.mp4')
    # fname = Path('/home/awb/Data/hgan/surreal/SURREAL/data/cmu/test/run0/ung_127_11/ung_127_11_c0001.mp4')
    fname = Path('/home/awb/Data/hgan/surreal/SURREAL/data/cmu/test/run1/75_19/75_19_c0002.mp4')
    frame = skvideo.io.vread(fname)[0]
    seg = scipy.io.loadmat(fname.parent / (fname.stem + '_segm.mat'))['segm_1']
    info = get_frame_info(scipy.io.loadmat(fname.parent / (fname.stem + '_info.mat')), 0)
    forig_shape = frame.shape[:2]

    crop = True
    if crop:
        frame, _, orig_size, new_crop = cropresize(frame, seg)

    # GET INRTINSICS AND EXTRINSICS
    zrot = info['zrot']
    RzBody = np.array(((math.cos(zrot), -math.sin(zrot), 0), (math.sin(zrot), math.cos(zrot), 0), (0, 0, 1)))
    intrinsic = get_intrinsic()
    extrinsic, R, T = get_extrinsic(info['camLoc'])

    # GET SMPL FROM INFO??
    go = rotateBody(RzBody, info['pose'][:3])

    betas = torch.from_numpy(info['shape'])[None].cuda()
    body_pose = torch.from_numpy(info['pose'])[None, 3:].cuda()
    global_orient = torch.from_numpy(go)[None, :3].cuda().float()
    joints3d = torch.from_numpy(info['joints3D']).T.cuda()
    root_pose = G.renderer.smpl_reduced.forward().joints[0,0]
    smpl_out = G.renderer.smpl_reduced.forward(betas=betas, body_pose=body_pose,
                                               global_orient=global_orient, transl=(joints3d[:1]-root_pose[None]))
    smpl_vertices = smpl_out.vertices[0].cpu().numpy()
    pts = smpl_vertices[:, :]

    # LOG RESULTS
    breakpoint()
    if crop:
        intrinsic[0,2] = (intrinsic[0,2] - new_crop[2]) * (128 / orig_size)
        intrinsic[1,2] = (intrinsic[1,2] - new_crop[0]) * (128 / orig_size)
        intrinsic[0,0] = intrinsic[0,0] * (128 / orig_size)
        intrinsic[1,1] = intrinsic[1,1] * (128 / orig_size)

    eg3d = True
    if eg3d:
        intrinsic[0,2] /= 128
        intrinsic[1,2] /= 128
        intrinsic[0,0] /= 128
        intrinsic[1,1] /= 128
        output_resolution = np.array([128, 128])

    if eg3d:
        pts_2d = project_eg3d(pts, intrinsic, extrinsic, output_resolution)
    else:
        pts_2d = project_vertices(pts, intrinsic, extrinsic)
    im_viz = frame.copy()
    plot_kpt(pts_2d, im_viz)
    imageio.imwrite('mesh_proj.png', im_viz)
    imageio.imwrite('gt.png', frame)

    breakpoint()

def compute_pck_for_dataset(opts, batch_size=64, batch_gen=None, pose_model=None, det_model=None, max_items=0):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0
    batch_gen = 8

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    imgc_iter = metric_utils.iterate_random_imgs_labels(opts=opts, batch_size=batch_gen)

    SPIN_processor = process_EG3D_image.EG3D_ImageProcessor(G.rendering_kwargs['cfg_name'], )

    # Initialize.
    teval_imgs = 0
    # Main loop.
    hits = 0
    total = 0

    with tqdm(total=max_items) as pbar:
        while teval_imgs < max_items:
            for _i in range(batch_size // batch_gen):
                gt_img, c = next(imgc_iter)
                # gt_img, c = get_surreal_data(G)
                z = torch.randn([batch_gen, G.z_dim], device=opts.device)

                override_warp = False
                if override_warp:
                    # G.rendering_kwargs['projector'] = 'none'
                    # G.rendering_kwargs['project_inside_only'] = False
                    img = G(z=z, c=c, **opts.G_kwargs)['image']
                    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                    # import imageio
                    # imageio.imwrite('test.png', img[-1].permute(1, 2, 0).cpu().numpy())
                    # breakpoint()

                    G.rendering_kwargs['projector'] = 'surface_field'
                    G.rendering_kwargs['project_inside_only'] = True
                    G.rendering_kwargs['cam2world'] = c[:, :16].view(c.shape[0], 4, 4)
                    G.rendering_kwargs['intrinsics'] = c[:, 16:25].view(c.shape[0], 3, 3)

                    params = [SPIN_processor.forward(img[i].permute(1, 2, 0)) for i in range(img.shape[0])]
                    betas = torch.cat([param['pred_betas'] for param in params], dim=0)
                    orients = torch.cat([param['global_orient'] for param in params], dim=0)
                    bodyposes = torch.cat([param['pred_rotmat'] for param in params], dim=0)[:, 3:]
                    if G.rendering_kwargs['cfg_name'] == 'surreal':
                        transl = torch.from_numpy(np.array(consts.SURREAL_TRANSL)).expand(betas.shape[0], -1).to(
                            betas.device)
                    elif G.rendering_kwargs['cfg_name'] == 'aist' or G.rendering_kwargs['cfg_name'] == 'aist_rescaled':
                        transl = torch.from_numpy(np.array(consts.AIST_TRANSL)).expand(betas.shape[0], -1).to(
                            betas.device)
                    else:
                        print(f"Error, unsupported dataset: {G.rendering_kwargs['cfg_name']}")
                        exit()

                    G.renderer.smpl_avg_body_pose = bodyposes
                    G.renderer.smpl_avg_orient = orients
                    G.renderer.smpl_avg_transl = transl
                    G.renderer.smpl_avg_betas = betas

                    pred_img = G(z=z, c=c, **opts.G_kwargs)['image']
                    pred_img = (pred_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                else:
                    pred_img = G(z=z, c=c, **opts.G_kwargs)['image']
                    pred_img = (pred_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                    # breakpoint()
                    # import imageio
                    # imageio.imwrite('test.png', pred_img[-1].permute(1, 2, 0).cpu().numpy())

                # if G.rendering_kwargs['cfg_name'] == 'surreal':
                #     gt_img = get_mesh_img_gt(c, G)

                gt_kpts, gt_failed = compute_kpts(gt_img, pose_model, det_model, 'gt')
                pred_kpts, pred_failed = compute_kpts(pred_img, pose_model, det_model, 'pred')
                gt_scores = gt_kpts[..., -1]
                gt_kpts = gt_kpts[..., :2]
                pred_scores = pred_kpts[..., -1]
                pred_kpts = pred_kpts[..., :2]

                # breakpoint()
                # if gt_failed or pred_failed:
                #     continue
                # mask = np.ones_like(gt_kpts)[..., 0].astype(bool)
                mask = np.logical_and((gt_scores > 0.8), (pred_scores > 0.8))
                # mask = (gt_scores > 0.8)
                thr = 0.5
                interocular = np.linalg.norm(gt_kpts[:, 0, :] - gt_kpts[:, 1, :], axis=1, keepdims=True)
                normalize = np.tile(interocular, [1, 2])

                oe = keypoint_pck_accuracy(pred_kpts, gt_kpts, mask, thr, normalize)
                hits += oe[1] * oe[2] * pred_kpts.shape[0]
                total += oe[2] * pred_kpts.shape[0]

                teval_imgs += batch_gen
                pbar.update(batch_gen)

    print(f'Total: {total}')
    return float(hits) / float(total)