#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import math
import torch
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim, apply_dog_filter, edge_loss
import sys
from scene import *
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import line_chart
import torch.nn.functional as F
from gui_utils.cam_utils import OrbitCamera
import dearpygui.dearpygui as dpg
from PIL import Image
from piq import ssim as ssim_func, LPIPS
lpips = LPIPS()
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



gs_dict = {'GS': GaussianModel, 'DRK': DRKModel}


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, fid):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.fid = fid
        self.c2w = c2w

        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda().float()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda().float()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        # self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class Trainer:
    def __init__(self, args, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, evaluate=False, load_iteration=-1):
        self.args = args
        self.dataset = dataset
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations
        self.checkpoint_iterations = checkpoint_iterations
        self.checkpoint = checkpoint
        self.debug_from = debug_from

        gs_model_type = gs_dict[self.dataset.gs_type]
        model_path_list = self.dataset.model_path.split('/')
        model_path_list[-1] += f'_{self.dataset.gs_type}'
        self.dataset.model_path = '/'.join(model_path_list)

        if opt.pure_train:
            # Detect if there is a trained model in the model path
            if os.path.exists(os.path.join(self.dataset.model_path, "point_cloud/iteration_30000")) or os.path.exists(os.path.join(self.dataset.model_path, "point_cloud/iteration_35000")):
                print("Finished training, no need to train again for {}".format(self.dataset.model_path))
                if not evaluate:
                    print('Exit because there is no evaluation task!')
                    exit(0)
            elif os.path.exists(os.path.join(self.dataset.model_path)):
                print("Found a half-trained model, deleting it and starting from scratch!")
                os.system(f'rm -r {os.path.join(self.dataset.model_path)}')
            else:
                print("No trained model found, starting from scratch!")

        self.first_iter = 0
        self.tb_writer = prepare_output_and_logger(self.dataset)
        
        self.gaussians = gs_model_type(self.dataset.sh_degree)
        scene = Scene(self.dataset, self.gaussians, load_iteration=load_iteration, shuffle=not evaluate)
        self.scene = scene
        self.first_iter = scene.first_iteration
        print(f'First iteration is: {self.first_iter}')
        self.gaussians.training_setup(self.opt)
        if self.checkpoint:
            (model_params, first_iter) = torch.load(self.checkpoint)
            self.gaussians.restore(model_params, self.opt)

        self.bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.gui_background = self.background.clone()

        self.iter_start = torch.cuda.Event(enable_timing = True)
        self.iter_end = torch.cuda.Event(enable_timing = True)
        self.iteration = 1 if self.scene.loaded_iter is None else self.scene.loaded_iter
        self.gaussians.update(self.iteration)

        self.viewpoint_stack = None
        self.best_psnr = 0.0
        self.best_psnr_val = 0.0
        self.best_ssim = 0.0
        self.best_lpips = np.inf
        self.best_iteration = -1
        self.ema_loss_for_log = 0.0
        self.progress_bar = tqdm(range(self.first_iter, self.opt.iterations), desc="Training progress")
        self.first_iter += 1

        # For UI
        self.gui = args.gui # enable gui
        self.W = args.W
        self.H = args.H
        self.cam = OrbitCamera(args.W, args.H, r=args.radius, fovy=args.fovy)
        
        self.vis_scale_rate = 1.
        self.vis_acutance_rate = 0.
        self.vis_l1l2rate_rate = .5

        self.mode = "render"
        self.seed = "random"
        self.sh_degree = None
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.training = False
        self.video_speed = 1.
        self.should_save_screenshot = False
        self.feat_transform_func = lambda x: np.copy(x[..., :3])
        
        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()
    
    def register_dpg(self):
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window
        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")

                # save current model
                all_modes = ['render', 'depth', 'alpha', 'normal', 'acu', 'opa', 'lrate']
                idx = np.linspace(0, len(all_modes), 4).astype('int')
                mode_seqs = [all_modes[idx[i]: idx[i+1]] for i in range(len(idx)-1)]
                for mode_seq in mode_seqs:
                    with dpg.group(horizontal=True):
                        dpg.add_text("Mode: ")

                        def callback_vismode(sender, app_data, user_data):
                            self.mode = user_data
                            if user_data == 'feat':
                                from utils.other_utils import get_transform
                                self.feat_transform_func = get_transform(self.gaussians.get_extra_features.detach().cpu().numpy(), out_dim=3)
                        for mode in mode_seq:
                            dpg.add_button(
                                label=mode,
                                tag=f"_button_vis_{mode}",
                                callback=callback_vismode,
                                user_data=mode,
                            )
                            dpg.bind_item_theme(f"_button_vis_{mode}", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("Scale Rate: ")
                    def callback_vis_scale_const(sender):
                        self.vis_scale_rate = 10 ** dpg.get_value(sender)
                        self.need_update = True
                    dpg.add_slider_float(
                        label="Log vis_scale_const (For debugging)",
                        default_value=0.,
                        max_value=5.,
                        min_value=-5.,
                        callback=callback_vis_scale_const,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("Acutance: ")
                    def callback_vis_acutance_rate(sender):
                        self.vis_acutance_rate = dpg.get_value(sender)
                        self.need_update = True
                    dpg.add_slider_float(
                        label="Log vis_acutance_rate (For debugging)",
                        default_value=0.,
                        max_value=.99,
                        min_value=-.99,
                        callback=callback_vis_acutance_rate,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("l1l2_rate: ")
                    def callback_l1l2rate_rate(sender):
                        self.vis_l1l2rate_rate = dpg.get_value(sender)
                        self.need_update = True
                    dpg.add_slider_float(
                        label="Log vis_acutance_rate (For debugging)",
                        default_value=.5,
                        max_value=1.,
                        min_value=0.,
                        callback=callback_l1l2rate_rate,
                    )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                        self.scene.save(self.iteration)

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    def callback_screenshot(sender, app_data):
                        self.should_save_screenshot = True
                    dpg.add_button(
                        label="screenshot", tag="_button_screenshot", callback=callback_screenshot
                    )
                    dpg.bind_item_theme("_button_screenshot", theme_button)

                    def callback_load_cam(sender, app_data):
                        from utils.pickle_utils import load_obj
                        self.cam = load_obj(os.path.join(self.dataset.model_path, 'screenshot', 'screenshot_camera.pickle'))
                    dpg.add_button(
                        label="load_cam", tag="_button_load_cam", callback=callback_load_cam
                    )
                    dpg.bind_item_theme("_button_load_cam", theme_button)

                    def callback_switch_bg(sender, app_data):
                        self.gui_background = 1 - self.gui_background
                    dpg.add_button(
                        label="switch_bg", tag="_button_switch_bg", callback=callback_switch_bg
                    )
                    dpg.bind_item_theme("_button_switch_bg", theme_button)
                    
            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            # self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                    def callback_metric(sender, app_data):
                        training_report(None, self.iteration, None, None, l1_loss, None, [self.iteration], self.scene, self.gaussians.render_func, (self.pipe, self.background), True, self)
                    dpg.add_button(
                        label="metric", tag="_button_metric", callback=callback_metric
                    )
                    dpg.bind_item_theme("_button_metric", theme_button)

                    def callback_visibility(sender, app_data):
                        self.gaussians.visualize_visibility()
                    dpg.add_button(
                        label="visibility", tag="_button_visibility", callback=callback_visibility
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_psnr")
                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    all_modes,
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # sh_degree combo
                def callback_change_sh_degree(sender, app_data):
                    self.sh_degree = None if app_data == 'None' else int(app_data)
                    self.need_update = True

                dpg.add_combo(
                    ['None', '0', '1', '2', '3'],
                    label="sh_degree",
                    default_value='None',
                    callback=callback_change_sh_degree,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )
   
        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True
                
        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="Gaussian3D",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        dpg.show_viewport()
    
    # gui mode
    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def train(self):
        while self.iteration < self.opt.iterations+1:
            self.train_step()
    
    def train_step(self):
        self.gaussians.train()
        self.gaussians.current_opt_step = self.iteration
        self.gaussians.update(self.iteration)

        self.iter_start.record()

        self.gaussians.update_learning_rate(self.iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))

        # Render
        if (self.iteration - 1) == self.debug_from:
            self.pipe.debug = True

        random_bg_cond = self.opt.random_background and viewpoint_cam.gt_alpha_mask is not None and self.iteration < self.opt.densify_until_iter
        bg = torch.rand((3), device="cuda") if random_bg_cond else self.background

        render_pkg = self.gaussians.render_func(viewpoint_cam, self.gaussians, self.pipe, bg)
        image, visibility_filter, radii = render_pkg["render"], render_pkg["visibility_filter"], render_pkg["radii"]


        gt_image = viewpoint_cam.original_image.cuda()
        #cur_size = (int(gt_image.shape[1] * down_sampling), int(gt_image.shape[2] * down_sampling))
        #gt_image = F.interpolate(gt_image.unsqueeze(0), size=cur_size, mode='bilinear', align_corners=False).squeeze(0)
        if random_bg_cond:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            gt_image = gt_image * gt_alpha_mask + bg[:, None, None] * (1 - gt_alpha_mask)
        if self.opt.lambda_im_laplace > 0:
            freq = (self.iteration / self.opt.iterations) * 100
            mask = apply_dog_filter(image.unsqueeze(0), freq=freq, scale_factor=self.opt.im_laplace_scale_factor).squeeze(0)
            mask_loss = l1_loss(image * mask, gt_image * mask)
        else:
            mask_loss = 0.
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + self.opt.lambda_im_laplace * mask_loss

        if hasattr(self.gaussians, 'regularization_loss') and self.iteration > self.opt.densify_from_iter and self.iteration < self.opt.densify_until_iter:
            loss = loss + self.gaussians.regularization_loss()
        
        loss.backward()

        if self.iteration < self.opt.densify_until_iter:
            self.gaussians.grad_postprocess()

        self.iter_end.record()

        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}", "GS": f"{self.gaussians._xyz.shape[0]}", "opa_pru": "%.3f" % self.gaussians.min_opacity_pruning})
                self.progress_bar.update(10)
            if self.iteration == self.opt.iterations:
                self.progress_bar.close()

            # Log and save
            cur_psnr, cur_ssim, cur_lpips, val_psnr = training_report(self.tb_writer, self.iteration, Ll1, loss, l1_loss, self.iter_start.elapsed_time(self.iter_end), self.testing_iterations, self.scene, self.gaussians.render_func, (self.pipe, self.background), trainer=self)
            if self.iteration in self.testing_iterations:
                if cur_psnr.item() > self.best_psnr:
                    self.best_psnr = cur_psnr.item()
                    self.best_iteration = self.iteration
                    self.best_ssim = cur_ssim.item()
                    self.best_lpips = cur_lpips.item()
                if val_psnr.item() > self.best_psnr_val:
                    self.best_psnr_val = val_psnr.item()
                    self.scene.save(self.iteration)
                elif cur_psnr.item() < 20:
                    print('Test right after opacity resetting!')
            
            if (self.iteration in self.saving_iterations):
                print("\n[ITER {}] Saving gaussians".format(self.iteration))
                self.scene.save(self.iteration)

            # Densification
            if self.iteration < self.opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                viewspace_point_tensor_densify = render_pkg["viewspace_points_densify"]
                if "opacity_grad_densify" in render_pkg:
                    opacity_grad_densify = render_pkg['opacity_grad_densify']
                else:
                    opacity_grad_densify = None
                self.gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter, opacity_tensor=opacity_grad_densify)

                if self.iteration > self.opt.densify_from_iter and self.iteration % self.gaussians.densification_interval == 0:
                    size_threshold = 20 if self.iteration > self.gaussians.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.gaussians.densify_grad_threshold, self.gaussians.min_opacity_pruning, self.scene.cameras_extent, size_threshold)
            
                if self.iteration !=0 and self.iteration % self.gaussians.opacity_reset_interval == 0 or (self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            # Optimizer step
            if self.iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none = True)

            if (self.iteration in self.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(self.iteration))
                torch.save((self.gaussians.capture(), self.iteration), self.scene.model_path + "/chkpnt" + str(self.iteration) + ".pth")

        self.iteration += 1

        if self.gui:
            dpg.set_value(
                "_log_train_log",
                f"step = {self.iteration: 5d} loss = {loss.item():.4f}",
            )
            dpg.set_value(
                "_log_train_psnr",
                "Best PSNR={} in Iteration {}, SSIM={}, LPIPS={}, GS number={}".format('%.5f' % self.best_psnr, self.best_iteration, '%.5f' % self.best_ssim, '%.5f' % self.best_lpips, '%d' % self.gaussians._xyz.shape[0])
            )
        else:
            self.progress_bar.set_description("Best PSNR={} in Iteration {}, SSIM={}, LPIPS={}, GS number={}".format('%.5f' % self.best_psnr, self.best_iteration, '%.5f' % self.best_ssim, '%.5f' % self.best_lpips, '%d' % self.gaussians._xyz.shape[0]))

    @torch.no_grad()
    def test_step(self):

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        if os.path.exists(os.path.join(self.dataset.model_path, 'screenshot_camera.pickle')) and self.should_save_screenshot:
            print('Use fixed camera for screenshot: ', os.path.join(self.dataset.model_path, 'screenshot', 'screenshot_camera.pickle'))
            from utils.pickle_utils import load_obj
            # cur_cam = load_obj(os.path.join(self.dataset.model_path, 'screenshot_camera.pickle'))
            cam = load_obj(os.path.join(self.dataset.model_path, 'screenshot_camera.pickle'))
            cur_cam = MiniCam(
                cam.pose,
                self.W,
                self.H,
                cam.fovy,
                cam.fovx,
                cam.near,
                cam.far,
                fid = 0
            )
        else:
            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                fid = 0
            )
        gaussians = self.gaussians

        vis_acutance_rate = self.vis_acutance_rate if self.mode == 'ctr_acu' or self.mode == 'ctr' else None
        vis_l1l2rate_rate = self.vis_l1l2rate_rate if self.mode == 'ctr_lra' or self.mode == 'ctr' else None
        mode_key = self.mode
        if self.mode == 'acu':
            override_color = self.gaussians.get_normalized_acutance
            override_color = torch.cat([override_color, torch.zeros_like(override_color), 1 - override_color], dim=-1)
            mode_key = "render"
        elif self.mode == 'opa':
            override_color = self.gaussians.get_opacity
            override_color = torch.cat([override_color, torch.zeros_like(override_color), 1 - override_color], dim=-1)
            mode_key = "render"
        elif self.mode == 'lrate':
            override_color = self.gaussians.get_l1l2rates
            override_color = torch.cat([override_color, torch.zeros_like(override_color), 1 - override_color], dim=-1)
            mode_key = "render"
        else:
            override_color = None

        out = gaussians.render_func(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.gui_background, vis_scale_rate=self.vis_scale_rate, vis_acutance_rate=vis_acutance_rate, vis_l1l2rate_rate=vis_l1l2rate_rate, opaque_mode=self.mode=='opaque', override_color=override_color, vis_contour=self.mode=='contour', gs2d_ewa=self.mode=='2dgs-ewa', sh_degree=self.sh_degree)

        if self.mode == 'normal':
            out["normal"] = (out["normal"] + 1) / 2
            mode_key = 'normal'

        buffer_image = out[mode_key]  # [3, H, W]

        if self.mode in ['depth', 'alpha']:
            buffer_image = buffer_image.repeat(3, 1, 1)
            if self.mode == 'depth':
                buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

        if self.should_save_screenshot:
            alpha = out['alpha'] if 'alpha' in out else None
            if alpha is not None:
                sv_image = torch.cat([buffer_image, alpha], dim=0).clamp(0,1).permute(1,2,0).detach().cpu().numpy()
            else:
                sv_image = buffer_image.clamp(0,1).permute(1,2,0).detach().cpu().numpy()
            def save_image(image, image_dir):
                os.makedirs(image_dir, exist_ok=True)
                idx = len([file for file in os.listdir(image_dir) if file.endswith('.png')])
                print('>>> Saving image to %s' % os.path.join(image_dir, '%05d.png'%idx))
                Image.fromarray((image * 255).astype('uint8')).save(os.path.join(image_dir, '%05d.png'%idx))                
                from utils.pickle_utils import save_obj
                save_obj(os.path.join(image_dir, '%05d_cam.pickle'% idx), self.cam)
            save_image(sv_image, os.path.join(self.dataset.model_path, 'screenshot'))
            self.should_save_screenshot = False

        buffer_image = torch.nn.functional.interpolate(
            buffer_image.unsqueeze(0),
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        self.buffer_image = (
            buffer_image.permute(1, 2, 0)
            .contiguous()
            .clamp(0, 1)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
        )

        self.need_update = True

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        buffer_image = self.buffer_image
        if self.gui:
            dpg.set_value(
                "_texture", buffer_image
            )  # buffer must be contiguous, else seg fault!
        return buffer_image

    def calculate_visibility(self):
        from gaussian_renderer import render_visibility
        viewpoint_stack = self.scene.getTrainCameras().copy()
        total_visibility = torch.zeros((self.gaussians._xyz.shape[0], 1), device='cuda')
        count_visibility = torch.zeros((self.gaussians._xyz.shape[0], 1), device='cuda')
        for viewpoint_cam in tqdm(viewpoint_stack):
            visibility = render_visibility(viewpoint_cam, self.gaussians, self.pipe)
            total_visibility += visibility
            count_visibility += (visibility > 0).float()
        self.gs_visibility = total_visibility
        self.gs_visibility_count = count_visibility
        save_dir = os.path.join(self.scene.model_path, 'visibility')
        os.makedirs(save_dir, exist_ok=True)
        pcl = self.gaussians._xyz.detach().cpu().numpy()
        color = self.gs_visibility_count.squeeze().detach().cpu().numpy()
        color = (color - color.min()) / (color.max() - color.min() + 1e-20)
        color = np.stack([color, np.zeros_like(color), 1 - color], axis=-1)
        color = np.array(color * 255, dtype='uint8')
        from plyfile import PlyData, PlyElement
        vertex = np.array([tuple(pcl[i]) + tuple(color[i]) for i in range(pcl.shape[0])] , dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el]).write(os.path.join(save_dir, 'visibility.ply'))

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, save_result=False, trainer=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = torch.tensor(0.0)
    test_ssim = torch.tensor(0.0)
    test_lpips = torch.tensor(1e5)
    val_psnr = torch.tensor(0.0)
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)] if not save_result else scene.getTrainCameras()},
                              {'name': 'val', 'cameras' : scene.getValCameras()})

        for config in validation_configs:

            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                psnr_raw = 0.0
                psnr_depth = 0.0
                psnr_list, ssim_list, lpips_list, l1_list = [], [], [], []
                psnr_masked_list, ssim_masked_list, lpips_masked_list = [], [], []
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if save_result:
                        os.makedirs(os.path.join(trainer.scene.model_path, 'metric', config['name']), exist_ok=True)
                        max_error_to_clip = 2e-1
                        error_map = ((image - gt_image).abs().sum(dim=0) / max_error_to_clip).clip(0., 1.)
                        error_map = torch.stack([error_map, torch.zeros_like(error_map), 1-error_map], dim=0)
                        vis_map = (torch.cat([gt_image, image, error_map], dim=2).detach().cpu().permute(1,2,0).numpy() * 255).astype('uint8')
                        Image.fromarray(vis_map).save(os.path.join(trainer.scene.model_path, 'metric', config['name'], 'errormap_%05d.png' % idx))
                        Image.fromarray((image.detach().cpu().permute(1,2,0).numpy() * 255).astype('uint8')).save(os.path.join(trainer.scene.model_path, 'metric', config['name'], 'render_%05d.png' % idx))
                        Image.fromarray((gt_image.detach().cpu().permute(1,2,0).numpy() * 255).astype('uint8')).save(os.path.join(trainer.scene.model_path, 'metric', config['name'], 'gt_%05d.png' % idx))
                    l1_list.append(l1_loss(image[None], gt_image[None]).mean())
                    psnr_list.append(psnr(image, gt_image).mean().double().mean())                  
                    ssim_list.append(ssim_func(image[None], gt_image[None], data_range=1.).mean())
                    lpips_list.append(lpips(image[None], gt_image[None]).mean())
                    if viewpoint.gt_alpha_mask is not None and trainer.dataset.metric_masked:
                        image_mask = image * viewpoint.gt_alpha_mask.to(image.device)
                        image_masked = image * image_mask
                        gt_image_masked = gt_image * image_mask
                        psnr_masked_list.append(psnr(image_masked, gt_image_masked).mean().double())
                        ssim_masked_list.append(ssim_func(image_masked[None], gt_image_masked[None], data_range=1.).mean())
                        lpips_masked_list.append(lpips(image_masked[None], gt_image_masked[None]).mean())
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim_func(image[None], gt_image[None], data_range=1.).mean()
                    lpips_test += lpips(image[None], gt_image[None]).mean()
                    if 'render_raw' in render_pkg and render_pkg["render_raw"] is not None:
                        image_raw =  torch.clamp(render_pkg["render_raw"], 0.0, 1.0)
                        psnr_raw  += psnr(image_raw, gt_image).mean().double()
                    if 'depth' in render_pkg and hasattr(viewpoint, 'depth_image'):
                        depth = render_pkg['depth']
                        gt_depth = viewpoint.depth_image
                        psnr_depth += psnr(depth, gt_depth).mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                psnr_raw /= len(config['cameras'])
                psnr_depth /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} GS Num {}".format(iteration, config['name'], l1_test, psnr_test, scene.gaussians._xyz.shape[0]))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                    if psnr_raw > 0.:
                        tb_writer.add_scalar(config['name'] + '/raw_image - psnr', psnr_raw, iteration)
                    if psnr_depth > 0.:
                        tb_writer.add_scalar(config['name'] + '/depth - psnr', psnr_depth, iteration)
                    if len(psnr_masked_list) > 0:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_masked', torch.stack(psnr_masked_list).mean().item(), iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim_masked', torch.stack(ssim_masked_list).mean().item(), iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips_masked', torch.stack(lpips_masked_list).mean().item(), iteration)
                l1_test = torch.stack(l1_list).mean()
                psnr_test = torch.stack(psnr_list).mean()
                ssim_test = torch.stack(ssim_list).mean()
                lpips_test = torch.stack(lpips_list).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                    test_ssim = ssim_test
                    test_lpips = lpips_test
                if config['name'] == 'val':
                    val_psnr = psnr_test
                if save_result:
                    os.makedirs(os.path.join(trainer.scene.model_path, 'metric'), exist_ok=True)
                    log_save_path = os.path.join(trainer.scene.model_path, 'metric', config['name'] + '_' + str(trainer.iteration) + '.txt')
                    with open(log_save_path, 'w') as f:
                        f.write("L1: %.5f\n" % l1_test)
                        f.write("PSNR: %.5f\n" % psnr_test)
                        f.write("SSIM: %.5f\n" % ssim_test)
                        f.write("LPIPS: %.5f\n" % lpips_test)
                        f.write("GS number: %d\n" % scene.gaussians._xyz.shape[0])
                        if len(psnr_masked_list) > 0:
                            f.write("PSNR Masked: %.5f\n" % torch.stack(psnr_masked_list).mean().item())
                            f.write("SSIM Masked: %.5f\n" % torch.stack(ssim_masked_list).mean().item())
                            f.write("LPIPS Masked: %.5f\n" % torch.stack(lpips_masked_list).mean().item())
                    line_chart(torch.stack(psnr_list).detach().cpu().numpy(), os.path.join(trainer.scene.model_path, 'metric', config['name'] + '_' + str(trainer.iteration) + '.png'), 'PSNR', 10., 70.)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
    return test_psnr, test_ssim, test_lpips, val_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--elevation', type=float, default=0, help="default GUI camera elevation")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 24_000, 30_000, 35_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 24_000, 30_000, 35_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--metric', action='store_true', help="Error analysis")
    parser.add_argument('--load_iteration', type=int, default=-1, help="load_iteration")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    evaluation = args.metric
    trainer = Trainer(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, evaluate=evaluation)

    if args.metric:
        training_report(None, trainer.iteration, None, None, l1_loss, None, [trainer.iteration], trainer.scene, trainer.gaussians.render_func, (trainer.pipe, trainer.background), True, trainer)
        print("\nMetric complete.")
    elif args.gui:
        trainer.render()
        print("\nGUI complete.")
    else:
        trainer.train()
        print("\nTraining complete.")

