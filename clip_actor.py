import os
import clip
from tqdm import tqdm
import kaolin.ops.mesh
import kaolin as kal
import torch
import numpy as np
import random
import copy
import torchvision
import argparse
from pathlib import Path
import torch.nn.functional as F
import warnings
warnings.simplefilter("ignore", UserWarning)
from torchvision import transforms
from smplx import SMPLX, SMPLH, SMPL
from pytorch3d.structures.meshes import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing
import time
import wandb

import motion_retrieval.retrieval_ as rtr
from motion_retrieval.sent2vec import Sent2Vec
import cfg as cfg
from models import NeuralStyleField
from render import Renderer
from mesh import HumanMesh
from utils import device, clip_model, create_video


def parse_args():
    parser = argparse.ArgumentParser()

    # body mesh
    parser.add_argument('--body_model', type=str, default='smplx', choices=['smpl', 'smplx', 'smplh'])
    parser.add_argument('--symmetry', type=eval, default=True, choices=[True, False])
    parser.add_argument('--standardize', type=eval, default=True, choices=[True, False])
    parser.add_argument('--mesh_subdivide', type=eval, default=True, choices=[True, False])

    # model
    parser.add_argument('--sigma', type=float, default=8.0)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--colordepth', type=int, default=2)
    parser.add_argument('--normdepth', type=int, default=2)
    parser.add_argument('--normwidth', type=int, default=256)
    parser.add_argument('--decay', type=float, default=0)  # weight decay
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--lr_plateau', action='store_true')
    parser.add_argument('--no_pe', dest='pe', default=False, action='store_false')
    parser.add_argument('--decay_step', type=int, default=100)
    parser.add_argument('--clamp', type=str, default="tanh")
    parser.add_argument('--normclamp', type=str, default="tanh")
    parser.add_argument('--normratio', type=float, default=0.1)
    parser.add_argument('--exclude', type=int, default=0)
    parser.add_argument('--input_verts', type=str, default='canonical', choices=['posed', 'canonical'])

    # training
    parser.add_argument('--prompt', nargs="+", default="a 3D rendering of the walking Steve Jobs in unreal engine")
    parser.add_argument('--anchor_mesh', type=str, default='top3', choices=['center', 'top1', 'top3'])
    parser.add_argument('--topk', type=int, default=3, choices=[1, 3, 5])
    parser.add_argument('--score_views', type=str, default='front', choices=['front', 'uniform'])
    parser.add_argument('--weighted_clip_mean', type=eval, default=True, choices=[True, False])
    parser.add_argument('--weight_th', type=float, default=0.05)
    parser.add_argument('--max_grad_norm', type=float, default=0.0)
    parser.add_argument('--n_iter', type=int, default=1500)  # can be increased
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--w_laplacian', type=float, default=250.0)
    parser.add_argument('--normweight', type=float, default=1.0)
    parser.add_argument('--clipavg', type=str, default='view')
    parser.add_argument('--geoloss', default=True, action="store_true")
    parser.add_argument('--mincrop', type=float, default=1)
    parser.add_argument('--maxcrop', type=float, default=1)
    parser.add_argument('--normmincrop', type=float, default=0.1)
    parser.add_argument('--normmaxcrop', type=float, default=0.4)
    parser.add_argument('--cropforward', action='store_true')
    parser.add_argument('--use_2d_aug', type=eval, default=True, choices=[True, False])
    parser.add_argument('--use_3d_aug', type=eval, default=True, choices=[True, False])

    # render
    parser.add_argument('--limited_elev', type=eval, default=True, choices=[True, False])
    parser.add_argument('--rand_cam_distance', type=eval, default=False, choices=[True, False])
    parser.add_argument('--elev_div', type=int, default=6)
    parser.add_argument('--n_azim', type=int, default=8)
    parser.add_argument('--n_elev', type=int, default=1)
    parser.add_argument('--n_augs', type=int, default=3)
    parser.add_argument('--n_normaugs', type=int, default=4)
    parser.add_argument('--frontview_std', type=float, default=4.0)
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[0., 0.])
    parser.add_argument('--background', nargs=3, type=float, default=[1.0, 1.0, 1.0])

    # logging
    parser.add_argument('--wandb_logging', type=eval, default=False, choices=[True, False])
    parser.add_argument('--exp_name', type=str, default='debug', required=True)
    parser.add_argument('--save_render', default=True, action="store_true")

    args = parser.parse_args()

    return args


def seed_all(args):
    # Constrain all sources of randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def retrieve_raw_label(encoded_text, encoded_raw_label):
    sim_raw = torch.cosine_similarity(encoded_text, encoded_raw_label, dim=1)
    idx = torch.topk(sim_raw, k=3)[1].cpu().numpy()
    return idx


def get_body_meshes(body_pose, right_hand_pose, left_hand_pose, args, device):
    smpl_vertices, smpl_faces = None, None
    batch_size = body_pose.shape[0]  # L

    # Rectify the randomly oriented SMPL mesh (head towards +y direction in world coordinate)
    global_orient = torch.zeros((batch_size, 3), device=device)
    global_orient[:, 1] = np.pi / 2  # Rectify
    betas = torch.zeros((batch_size, 16), device=device)  # Shape parameter

    # Get body mesh: vertices and faces
    if args.body_model == 'smplx':
        print("=> Loading SMPL-X model ...")
        smplx = SMPLX(model_path=cfg.smplx_path, ext='pkl', use_pca=False, num_betas=16, batch_size=batch_size).to(device)
        smplx_mesh = smplx(betas=betas, global_orient=global_orient, body_pose=body_pose, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
        smpl_vertices = smplx_mesh.vertices.detach()
        smpl_vertices -= smplx_mesh.joints.detach()[:, 0, :][:, None, :]
        smpl_faces = torch.tensor((smplx.faces.astype(np.int64)), device=device).unsqueeze(0)  # [1, 20908, 3]

    elif args.body_model == 'smpl':
        print("=> Loading SMPL model ...")
        smpl = SMPL(model_path=cfg.smpl_path, num_betas=16, batch_size=batch_size).to(device)
        body_pose = torch.cat([body_pose, torch.zeros((batch_size, 2 * 3), device=device)], dim=1)
        smpl_mesh = smpl(betas=betas, global_orient=global_orient, body_pose=body_pose)
        smpl_vertices = smpl_mesh.vertices.detach()  # [L, 6890, 3]
        smpl_vertices -= smpl_mesh.joints.detach()[:, 0, :][:, None, :]
        smpl_faces = smpl.faces_tensor.unsqueeze(0)  # [1, 13776, 3]

    return smpl_vertices, smpl_faces


def get_body_mesh(args, device):
    ########### Action 2 SMPLH param ###########
    '''
    input: args.prompt[0] (dtype = str)
    output: SMPL-H parameters
    '''
    smpl_vertices, smpl_faces = None, None

    # Define pose and shape parameters
    betas = torch.zeros((1, 16), device=device)  # Shape parameter
    global_orient = torch.zeros((1, 3), device=device)
    global_orient[:, 1] = np.pi / 2  # Rectify
    body_pose = torch.zeros((1, 21 * 3), device=device)  # Pose parameter (global_orient + body_pose): [1 x 3 + 22 x 3]
    left_hand_pose = torch.zeros((45,), device=device)  # Left hand parameter (15 x 3)
    right_hand_pose = torch.zeros((45,), device=device)  # Right hand parameter (15 x 3)

    # Rectify the randomly oriented SMPL mesh (head towards +y direction in world coordinate)
    global_orient = torch.zeros((1, 3), device=device)
    global_orient[:, 1] = np.pi / 2  # Rectify

    # Get body mesh: vertices and faces
    if args.body_model == 'smplx':
        smplx = SMPLX(model_path=cfg.smplx_path, ext='pkl', use_pca=False, num_betas=16).to(device)
        smplx_mesh = smplx(betas=betas, global_orient=global_orient, body_pose=body_pose, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
        smpl_vertices = smplx_mesh.vertices.detach()
        smpl_vertices -= smplx_mesh.joints.detach()[:, 0, :]
        smpl_faces = torch.tensor((smplx.faces.astype(np.int64)), device=device).unsqueeze(0)  # [1, 20908, 3]

    elif args.body_model == 'smpl':
        smpl = SMPL(model_path=cfg.smpl_path, num_betas=16).to(device)
        body_pose = torch.cat([body_pose, torch.zeros((1, 2 * 3), device=device)], dim=1)
        smpl_mesh = smpl(betas=betas, global_orient=global_orient, body_pose=body_pose)
        smpl_vertices = smpl_mesh.vertices.detach()  # [L, 6890, 3]
        smpl_vertices -= smpl_mesh.joints.detach()[:, 0, :]
        smpl_faces = torch.tensor((smpl.faces.astype(np.int64)), device=device).unsqueeze(0)  # [1, 13776, 3]

    return smpl_vertices, smpl_faces


def get_augmentation(args):
    clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    # CLIP Transform
    clip_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        clip_normalizer
    ])

    # Augmentation settings
    full_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(1, 1)),
        transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
        clip_normalizer
    ])
    # Augmentations for normal network
    if args.cropforward:
        curcrop = args.normmincrop
    else:
        curcrop = args.normmaxcrop

    # Local crop augmentations
    local_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(curcrop, curcrop)),
        transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
        clip_normalizer
    ])

    if args.weighted_clip_mean:
        # Displacement-only augmentations
        dispaug_transform = transforms.Compose([
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            clip_normalizer
        ])

    else:
        # Displacement-only augmentations
        dispaug_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(args.normmincrop, args.normmincrop)),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            clip_normalizer
        ])

    return clip_transform, full_transform, local_transform, dispaug_transform


def get_model(args):
    # MLP Settings
    input_dim = 3
    mlp = NeuralStyleField(args.sigma, args.depth, args.width, 'gaussian', args.colordepth, args.normdepth,
                           args.normratio, args.clamp, args.normclamp, niter=args.n_iter,
                           progressive_encoding=args.pe, input_dim=input_dim, exclude=args.exclude).to(device)
    mlp.reset_weights()

    optim = torch.optim.Adam(mlp.parameters(), args.learning_rate, weight_decay=args.decay)
    activate_scheduler = args.lr_decay < 1 and args.decay_step > 0 and not args.lr_plateau
    if activate_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.decay_step, gamma=args.lr_decay)

    return mlp, optim, activate_scheduler, lr_scheduler


def report_process(args, dir, i, loss, rendered_images, exp_name, clip_score, pose_label):
    full_loss = loss['global_loss'] + loss['local_loss'] + loss['geo_loss']
    if i == args.n_iter:
        print('[{}] Final CLIP score: {}'.format(exp_name, clip_score))
    else:
        print('[{}] iter: {} loss: {} CLIP score: {}'.format(exp_name, i, full_loss, clip_score))
    # WandB support
    if args.wandb_logging:
        render_img_log = wandb.Image(rendered_images, caption='/'.join(args.prompt) + ' [' + pose_label + ']')
        logs = dict(
            # loss=full_loss,
            # global_loss=loss['global_loss'],
            # local_loss=loss['local_loss'],
            # geo_loss=loss['geo_loss'],
            # clip_score=clip_score,
            rendered_images=render_img_log
        )
        wandb.log(logs, step=i)
    else:
        torchvision.utils.save_image(rendered_images, os.path.join(dir, 'iter_{}.jpg'.format(i)))


def export_medium_results(args, dir, mesh, mlp, network_input, vertices, n_iter):
    with torch.no_grad():
        pred_rgb, pred_normal = mlp(network_input)
        pred_rgb = pred_rgb.detach().cpu()
        pred_normal = pred_normal.detach().cpu()

        base_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5)
        final_color = torch.clamp(pred_rgb + base_color, 0, 1)

        mesh.vertices = vertices.detach().cpu() + mesh.vertex_normals.detach().cpu() * pred_normal

        save_rendered_results(args, dir, final_color, mesh, str(n_iter))


def export_final_results(args, dir, losses, mesh, pred_rgb, pred_normal, vertices, exp_name='YYYYMMDD'):
    with torch.no_grad():
        pred_rgb = pred_rgb.detach().cpu()
        pred_normal = pred_normal.detach().cpu()

        torch.save(pred_rgb, os.path.join(dir, f"colors_final.pt"))
        torch.save(pred_normal, os.path.join(dir, f"normals_final.pt"))

        base_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5)
        final_color = torch.clamp(pred_rgb + base_color, 0, 1)

        mesh.vertices = vertices.detach().cpu() + mesh.vertex_normals.detach().cpu() * pred_normal

        mesh.export(os.path.join(dir, f"{exp_name}_final.obj"), color=final_color)

        # Run renders
        if args.save_render:
            save_rendered_results(args, dir, final_color, mesh)


def export_motion_results(args, dir, mesh_t, pred_rgb, pred_normal, meshes, i=None):
    pred_rgb = pred_rgb.detach().cpu()
    pred_normal = pred_normal.detach().cpu()

    base_color = torch.full(size=(mesh_t.vertices.shape[0], 3), fill_value=0.5)
    final_color = torch.clamp(pred_rgb + base_color, 0, 1)
    if i is None:
        os.mkdir(os.path.join(dir, 'motion_res'))
    else:
        os.mkdir(os.path.join(dir, 'initial_motion_res'))
    print("=> Saving the textured meshes...")
    save_path = os.path.join(dir, 'motion_res')
    for idx, mesh in enumerate(tqdm(meshes)):
        mesh.vertices = mesh.vertices.detach().cpu() + mesh.vertex_normals.detach().cpu() * pred_normal
        save_rendered_results(args, save_path, final_color, mesh, str(idx))
    create_video(os.path.join(save_path + '/%04d.png'), dir + '/motion.mp4')


def save_rendered_results(args, dir, final_color, mesh, n_iter='final', clip_sim_render=None, encoded_text=None):
    kal_render = Renderer(
        camera=kal.render.camera.generate_perspective_projection(np.pi / 4, 1280 / 720).to(device),
        dim=(1280, 720))

    ## If you want to render de-colorized mesh
    # default_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5, device=device)
    # mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(default_color.unsqueeze(0),
    #                                                                mesh.faces.to(device))
    # # MeshNormalizer(mesh)()
    # img, mask = kal_render.render_single_view(mesh, args.frontview_center[1], args.frontview_center[0],
    #                                           radius=2.5,
    #                                           background=torch.tensor([1, 1, 1]).to(device).float(),
    #                                           return_mask=True)
    # img = img[0].cpu()
    # mask = mask[0].cpu()
    # Manually add alpha channel using background color
    # alpha = torch.ones(img.shape[1], img.shape[2])
    # alpha[torch.where(mask == 0)] = 0
    # img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
    # img = transforms.ToPILImage()(img)
    # img.save(os.path.join(dir, "normal_{}.png".format(n_iter)))

    # Vertex colorings
    mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(final_color.unsqueeze(0).to(device),
                                                                   mesh.faces.to(device))
    img, mask = kal_render.render_single_view(mesh, args.frontview_center[1], args.frontview_center[0],
                                              radius=2.5,
                                              background=torch.tensor([1, 1, 1]).to(device).float(),
                                              return_mask=True)
    img = img[0].cpu()
    mask = mask[0].cpu()
    # Manually add alpha channel using background color
    alpha = torch.ones(img.shape[1], img.shape[2])
    alpha[torch.where(mask == 0)] = 0
    img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
    img = transforms.ToPILImage()(img)
    save_path = os.path.join(dir, "{0:0>4}.png".format(n_iter))
    img.save(save_path)


def update_mesh(pred_rgb, pred_normal, prior_color, sampled_mesh):
    sampled_mesh.face_attributes = prior_color + kaolin.ops.mesh.index_vertices_by_faces(pred_rgb.unsqueeze(0),
                                                                                         sampled_mesh.faces)
    sampled_mesh.vertices = sampled_mesh.vertices + sampled_mesh.vertex_normals * pred_normal


def update_mesh_sequences(pred_rgb, pred_normal, prior_color, meshes):
    L = len(meshes)
    for idx in range(L):
        sampled_mesh = meshes[idx]
        sampled_mesh.face_attributes = prior_color + kaolin.ops.mesh.index_vertices_by_faces(pred_rgb.unsqueeze(0),
                                                                                             sampled_mesh.faces)
        sampled_mesh.vertices = sampled_mesh.vertices + sampled_mesh.vertex_normals * pred_normal


# training
def main(args):
    exp_name = time.strftime('%Y%m%d', time.localtime()) + '_' + args.exp_name
    output_dir = cfg.log_path + exp_name
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    seed_all(args)

    # Load predefined npy files
    babel_raw_label = np.load(cfg.raw_label_path)
    en_raw_label = np.load(cfg.encoded_raw_label_path)
    en_raw_label = torch.tensor(en_raw_label).to(device)

    # Initialize wandb logging
    if args.wandb_logging:
        wandb.login()
        wandb.init(project="project_name", entity="your_wandb_user_name",
                   name=time.strftime('%Y%m%d', time.localtime()) + '_' + args.exp_name)
        wandb.config.update(args)

    # Define the renderer
    render = Renderer()

    # Get text prompt and tokenize it (and save it)
    prompt = "a 3D rendering of " + args.prompt[0] + " in unreal engine"
    prompt_token = clip.tokenize([prompt]).to(device)
    # Normalized CLIP text embedding
    encoded_text_unnorm = clip_model.encode_text(prompt_token)
    encoded_text = F.normalize(encoded_text_unnorm)

    # Retrieve pose from BABEL
    sent2vec = Sent2Vec()
    idx = retrieve_raw_label(encoded_text_unnorm, en_raw_label)
    pose_label = sent2vec.compute_distance(args.prompt[0], babel_raw_label, idx)
    body_pose, right_hand_pose, left_hand_pose = rtr.retrieval_motion(pose_label)
    L = body_pose.shape[0]

    print("Retrieved pose label: " + pose_label)
    del babel_raw_label, en_raw_label

    # define image transforms for CLIP loss
    if args.use_2d_aug:
        clip_transform, full_transform, local_transform, dispaug_transform = get_augmentation(args)
    else:
        clip_transform, _, _, _ = get_augmentation(args)
    background = torch.tensor(args.background, device=device)

    # get body mesh and convert to custom HumanMesh class
    temp_vertices, temp_faces = get_body_mesh(args, device)
    smpl_vertices, smpl_faces = get_body_meshes(body_pose, right_hand_pose, left_hand_pose, args, device)

    print("=> Loading template mesh ...")
    mesh_t = HumanMesh(v=temp_vertices, f=temp_faces, args=args)

    meshes = []
    anchor_rdr_all = torch.cuda.FloatTensor()
    print("=> Loading mesh sequences and analyzing anchor views ...")
    for idx in tqdm(range(L)):
        mesh_i = HumanMesh(v=smpl_vertices[idx].unsqueeze(0), f=smpl_faces, args=args)
        meshes.append(mesh_i)

    for m_i in range(L):
        rdr = render.render_single_view(meshes[m_i], background=background)
        anchor_rdr_all = torch.cat((anchor_rdr_all, rdr), dim=0)

    with torch.no_grad():
        anchor_rdr = clip_transform(anchor_rdr_all)
        encoded_anchor_rdr = F.normalize(clip_model.encode_image(anchor_rdr))
        anchor_sim = torch.cosine_similarity(encoded_anchor_rdr, encoded_text)

        _, anchor_idx = torch.topk(anchor_sim, k=args.topk)

    del anchor_rdr_all, anchor_rdr, rdr

    if args.input_verts == 'posed':
        if args.anchor_mesh == 'center':
            frame_idx = [int(round(L / 2))]
        else:
            frame_idx = [int(anchor_idx[0].item())]
        mesh_input = copy.deepcopy(meshes[frame_idx[0]])

    else:
        if args.anchor_mesh == 'center':
            frame_idx = [int(round(L / 2))]
        elif args.anchor_mesh == 'top1':
            frame_idx = [int(anchor_idx[0].item())]
        else:
            frame_idx = [int(k.item()) for k in anchor_idx]
        mesh_input = copy.deepcopy(mesh_t)

    prior_color = torch.full(size=(mesh_t.faces.shape[0], 3, 3), fill_value=0.5, device=device)
    vertices = copy.deepcopy(mesh_input.vertices)
    network_input = copy.deepcopy(vertices)

    losses = []

    # get model and optimizer
    mlp, optim, activate_scheduler, lr_scheduler = get_model(args)

    if args.symmetry:
        network_input[:, 2] = torch.abs(network_input[:, 2])

    if args.standardize:
        # Each channel into z-score
        network_input = (network_input - torch.mean(network_input, dim=0)) / torch.std(network_input, dim=0)

    # Main training loop
    for i in tqdm(range(args.n_iter)):
        optim.zero_grad()
        mesh_list = [copy.deepcopy(meshes[j]) for j in frame_idx]

        pred_rgb, pred_normal = mlp(network_input)
        update_mesh_sequences(pred_rgb, pred_normal, prior_color, mesh_list)

        loss = 0.0
        # Compute S_full (from paper, Eq.(2)) and update the entire network
        rendered_images_all = torch.cuda.FloatTensor()
        masks_all = torch.cuda.FloatTensor()
        if args.use_3d_aug:
            n_azim = args.n_azim
        else:
            n_azim = 1

        for sampled_mesh in mesh_list:
            rendered_images, masks, elev, azim = render.render_views(sampled_mesh, num_azim=n_azim,
                                                                     num_elev=args.n_elev,
                                                                     center_azim=args.frontview_center[0],
                                                                     center_elev=args.frontview_center[1],
                                                                     elev_div=args.elev_div,
                                                                     std=args.frontview_std,
                                                                     return_views=True,
                                                                     background=background,
                                                                     limited_elev=args.limited_elev,
                                                                     rand_cam_distance=args.rand_cam_distance,
                                                                     args=args)
            rendered_images_all = torch.cat((rendered_images_all, rendered_images), dim=0)
            masks_all = torch.cat((masks_all, masks), dim=0)

        for _ in range(args.n_augs):
            if args.use_2d_aug:
                augmented_image = full_transform(rendered_images_all)
            else:
                augmented_image = rendered_images_all

            encoded_renders = clip_model.encode_image(augmented_image)
            if args.clipavg == "view":
                loss += 1 - torch.cosine_similarity(F.normalize(torch.mean(encoded_renders, dim=0, keepdim=True)),
                                                    encoded_text).squeeze()
            else:
                loss += 1 - torch.mean(torch.cosine_similarity(encoded_renders, encoded_text)).squeeze()

        loss.backward(retain_graph=True)
        if args.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=args.max_grad_norm)

        # Normal augment transform
        if args.n_normaugs > 0:
            # Compute S_local (from paper, Eq.(3)) and update the entire network
            normloss = 0.0
            for _ in range(args.n_normaugs):
                if args.use_2d_aug:
                    augmented_image = local_transform(rendered_images_all)
                else:
                    augmented_image = rendered_images_all
                encoded_renders = clip_model.encode_image(augmented_image)

                if args.clipavg == "view":
                    normloss += args.normweight * (1 - torch.cosine_similarity(
                        F.normalize(torch.mean(encoded_renders, dim=0, keepdim=True)), encoded_text))

                else:
                    normloss += args.normweight * (
                            1 - torch.mean(torch.cosine_similarity(encoded_renders, encoded_text)))

            normloss.backward(retain_graph=True)
            if args.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=args.max_grad_norm)

        # Also run separate loss on the uncolored displacements
        if args.geoloss:
            # Compute S_displ (from paper, Eq.(4)) update only backbone and displacement branch
            geoloss = 0.0
            default_color = 0.5 * torch.ones((len(mesh_t.vertices), 3), device=device)
            geo_renders_all = torch.cuda.FloatTensor()
            geo_masks_all = torch.cuda.FloatTensor()
            for sampled_mesh in mesh_list:
                sampled_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(default_color.unsqueeze(0),
                                                                                       sampled_mesh.faces)
                geo_renders, geo_masks, elev, azim = render.render_views(sampled_mesh, num_azim=n_azim,
                                                                         num_elev=args.n_elev,
                                                                         center_azim=args.frontview_center[0],
                                                                         center_elev=args.frontview_center[1],
                                                                         elev_div=args.elev_div,
                                                                         std=args.frontview_std,
                                                                         return_views=True,
                                                                         background=background,
                                                                         limited_elev=args.limited_elev,
                                                                         rand_cam_distance=args.rand_cam_distance,
                                                                         args=args)
                geo_renders_all = torch.cat((geo_renders_all, geo_renders), dim=0)
                geo_masks_all = torch.cat((geo_masks_all, geo_masks), dim=0)

                updated_mesh_p3d = Meshes(sampled_mesh.vertices.unsqueeze(0), sampled_mesh.faces.unsqueeze(0))
                norm_laplacian_loss = args.w_laplacian * mesh_laplacian_smoothing(updated_mesh_p3d)
                geoloss += norm_laplacian_loss

            if args.n_normaugs > 0:
                for _ in range(args.n_normaugs):
                    if args.weighted_clip_mean:
                        if args.use_2d_aug:
                            n_renders = n_azim * args.n_elev * args.topk
                            geo_crops_all = torch.cuda.FloatTensor()
                            geo_cropped_images = torch.cuda.FloatTensor()
                            geo_crop_emb_weights = torch.zeros((n_renders,), device=device)
                            geo_render_embeddings = torch.cuda.FloatTensor()

                            for crop_i in range(n_azim * args.n_elev * args.topk):
                                geo_crop = transforms.RandomResizedCrop(224)
                                geo_crop_i, geo_crop_j, geo_crop_h, geo_crop_w \
                                    = geo_crop.get_params(geo_renders_all, scale=[args.normmincrop, args.normmincrop], ratio=[3. / 4., 4. / 3.])
                                geo_crop_params = torch.tensor([geo_crop_i, geo_crop_j, geo_crop_h, geo_crop_w],
                                                               device=device).unsqueeze(0)
                                geo_crops_all = torch.cat((geo_crops_all, geo_crop_params), dim=0)

                                geo_cropped_img = \
                                    transforms.functional.resized_crop(geo_renders_all[crop_i], geo_crop_i, geo_crop_j,
                                                                       geo_crop_h, geo_crop_w, (224, 224)).unsqueeze(0)
                                geo_cropped_images = torch.cat((geo_cropped_images, geo_cropped_img), dim=0)

                                # Compute cropped mask
                                geo_cropped_mask = geo_masks_all[crop_i, geo_crop_i:geo_crop_i + geo_crop_h,
                                                   geo_crop_j:geo_crop_j + geo_crop_w]
                                geo_crop_emb_weights[crop_i] = torch.sum(geo_cropped_mask) / (geo_crop_h * geo_crop_w)

                            if torch.max(geo_crop_emb_weights) > 0.0:
                                if args.weight_th > 0.0:
                                    weight_thresh = torch.nn.Threshold(args.weight_th, 0)
                                    geo_crop_emb_weights = weight_thresh(geo_crop_emb_weights)  # gradient flow through geo_embedding_weight

                                augmented_image = dispaug_transform(geo_cropped_images)
                                encoded_renders_no_weight = clip_model.encode_image(augmented_image)
                                valid_render_embeddings = encoded_renders_no_weight[torch.nonzero(geo_crop_emb_weights)].squeeze(1)

                                geoloss += 1 - torch.cosine_similarity(
                                    F.normalize(torch.mean(valid_render_embeddings, dim=0, keepdim=True)), encoded_text).squeeze()

                            else:
                                continue

                        else:
                            encoded_renders = clip_model.encode_image(geo_renders_all)
                            geoloss += 1 - torch.cosine_similarity(
                                F.normalize(torch.mean(encoded_renders, dim=0, keepdim=True)), encoded_text).squeeze()

                    else:
                        augmented_image = dispaug_transform(geo_renders_all)
                        encoded_renders = clip_model.encode_image(augmented_image)
                        geoloss += 1 - torch.cosine_similarity(
                            F.normalize(torch.mean(encoded_renders, dim=0, keepdim=True)), encoded_text).squeeze()

                geoloss.backward(retain_graph=True)
                if args.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=args.max_grad_norm)

        optim.step()

        if activate_scheduler:
            lr_scheduler.step()

        with torch.no_grad():
            losses = dict(
                global_loss=loss.item(),
                local_loss=normloss.item(),
                geo_loss=geoloss.item()
            )

        with torch.no_grad():
            mesh = copy.deepcopy(meshes[frame_idx[0]])
            vis_vertices = copy.deepcopy(mesh.vertices)
            pred_rgb, pred_normal = mlp(network_input)
            update_mesh(pred_rgb, pred_normal, prior_color, mesh)

        if i % 100 == 0:
            if args.score_views == 'front':
                score_views = clip_transform(render.render_front_views(mesh, num_views=1, background=background))
            else:
                score_views = clip_transform(render.render_uniform_views(mesh, background=background))
            clip_score = torch.mean(torch.cosine_similarity(F.normalize(clip_model.encode_image(score_views)),
                                                            encoded_text)).squeeze().item()
            report_process(args, output_dir, i, losses, rendered_images_all, exp_name, clip_score, pose_label)

    with torch.no_grad():
        if args.score_views == 'front':
            score_views = clip_transform(render.render_front_views(mesh, num_views=1, background=background))
        else:
            score_views = clip_transform(render.render_uniform_views(mesh, background=background))
        final_clip_score = torch.mean(
            torch.cosine_similarity(F.normalize(clip_model.encode_image(score_views)), encoded_text)).squeeze().item()
        export_final_results(args, output_dir, losses, mesh, pred_rgb, pred_normal, vis_vertices, exp_name)
        report_process(args, output_dir, args.n_iter, losses, rendered_images_all, exp_name, final_clip_score, pose_label)
        export_motion_results(args, output_dir, mesh_t, pred_rgb, pred_normal, meshes)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = parse_args()
    main(args)
