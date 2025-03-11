import os
import torch
import math
import colorsys
import numpy as np
from PIL import Image
import dearpygui.dearpygui as dpg
from scene import DRKModel, RGB2SH
from scipy.spatial.transform import Rotation as R


def generate_quaternions(K):
    quaternions = []
    for n in range(K):  # Step by 2 for even rotations
        theta = math.pi * n / K
        q = (
            math.sin(theta / 2),
            math.cos(theta / 2),
            0,
            0,
        )
        quaternions.append(q)
    return quaternions


def generate_distinct_colors(K):
    colors = []
    for i in range(K):
        # Use HSV space to distribute colors evenly, then convert to RGB
        hue = i / K
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append([r, g, b])
    return colors


def generate_camera_matrices(cam_to_world, distance, num_frames):
    """
    Generate a series of camera-to-world matrices for a camera rotating around a point in its xz-plane.

    Parameters:
        cam_to_world (np.ndarray): The initial 4x4 camera-to-world matrix.
        distance (float): Distance along the camera's z-axis to the rotation center.
        num_frames (int): Number of frames (matrices) to generate.

    Returns:
        list of np.ndarray: A list of 4x4 camera-to-world matrices.
    """
    # Ensure input is a numpy array
    cam_to_world = np.array(cam_to_world)
    if cam_to_world.shape != (4, 4):
        raise ValueError("Input cam_to_world must be a 4x4 matrix.")

    # Extract the camera position and orientation
    cam_position = cam_to_world[:3, 3]  # Translation vector
    cam_z_axis = cam_to_world[:3, 2]   # Forward vector (z-axis)
    cam_y_axis = cam_to_world[:3, 1]   # Up vector (y-axis)

    # Compute the rotation center
    rotation_center = cam_position + distance * cam_z_axis

    # Generate rotation matrices
    angle_step = 2 * np.pi / num_frames  # Angle increment for each frame
    rotation_matrices = []
    for i in range(num_frames):
        # Compute the angle for this frame
        angle = angle_step * i

        # Compute the new position on the circle
        x_offset = np.sin(angle) * distance
        z_offset = np.cos(angle) * distance
        new_position = rotation_center + x_offset * cam_to_world[:3, 0] - z_offset * cam_to_world[:3, 2]

        # Compute the new camera-to-world matrix
        # The new x-axis is perpendicular to the up vector and the vector to the rotation center
        new_z_axis = rotation_center - new_position
        new_z_axis /= np.linalg.norm(new_z_axis)  # Normalize

        new_x_axis = np.cross(cam_y_axis, new_z_axis)
        new_x_axis /= np.linalg.norm(new_x_axis)  # Normalize

        new_y_axis = np.cross(new_z_axis, new_x_axis)  # Ensure orthogonality

        # Construct the new camera-to-world matrix
        new_cam_to_world = np.eye(4)
        new_cam_to_world[:3, 0] = new_x_axis  # X-axis
        new_cam_to_world[:3, 1] = new_y_axis  # Y-axis
        new_cam_to_world[:3, 2] = new_z_axis  # Z-axis
        new_cam_to_world[:3, 3] = new_position  # Position

        rotation_matrices.append(new_cam_to_world)

    return rotation_matrices


def getProjectionMatrix(znear, zfar, fovX=None, fovY=None, tanHalfFovX=None, tanHalfFovY=None):
    if tanHalfFovX is None:
        tanHalfFovX = math.tan(fovX / 2) if type(fovY) is float else torch.tan(fovY / 2)
    if tanHalfFovY is None:
        tanHalfFovY = math.tan(fovY / 2) if type(fovY) is float else torch.tan(fovY / 2)

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
    def __init__(self, width, height, fx, fy, c2w=None):
        self.image_width = width
        self.image_height = height
        self.znear = 1e-8
        self.zfar = 1e8
        
        self.FoVx = torch.atan(width/2/fx) * 2
        self.FoVy = torch.atan(height/2/fy) * 2

        self.fx = fx
        self.fy = fy

        if c2w is not None:
            w2c = c2w.inverse() 
        else:
            c2w = torch.eye(4).float().cuda()
            w2c = c2w.clone()
        self.world_view_transform = w2c.detach().clone().transpose(0, 1).cuda().float()
        
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda().float()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = c2w[:3, 3]

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, near=0.01, far=100):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = np.deg2rad(fovy)  # deg 2 rad
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        # self.rot = R.from_matrix(np.eye(3))
        self.rot = R.from_matrix(np.array([[1., 0., 0.,],
                                           [0., 0., -1.],
                                           [0., 1., 0.]]))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.side = np.array([1, 0, 0], dtype=np.float32)

    @property
    def fovx(self):
        return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        return self.pose[:3, 3]

    # pose (c2w)
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = - self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view (w2c)
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # projection (perspective)
    @property
    def perspective(self):
        y = np.tan(self.fovy / 2)
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    @property
    def mvp(self):
        return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        up = self.rot.as_matrix()[:3, 1]
        rotvec_x = up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0001 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])


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


from scene import get_range_activation, get_range_inv_activation
class DRKDemoModel(DRKModel):
    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)
        self.acutance_min, self.acutance_max = -.99, .99
        self.acutance_activation = get_range_activation(self.acutance_min, self.acutance_max)
        self.inv_acutance_activation = get_range_inv_activation(self.acutance_min, self.acutance_max)
        # Single DRK Demo
        self.single_drk_args = {'xyz': torch.zeros([1, 3]).float().cuda(),
                                'l1l2_rate': torch.zeros([1, 1]).cuda(),
                                'scale': torch.zeros([1, 8]).cuda(),
                                'theta': torch.zeros([1, 8]).cuda(),
                                'acutance': 1 * torch.ones([1, 1]).cuda(),
                                'color': torch.tensor([[170., 178., 255.]]).cuda() / 255.,
                                'rotation': torch.tensor([[2**.5/2, 2**.5/2, 0., 0.]]).cuda()
                                }
        
        # Multiple DRK Demo
        drk_num = 16
        self.multiple_drk_args = {'xyz': torch.zeros([drk_num, 3]).float().cuda(),
                                  'l1l2_rate': torch.zeros([drk_num, 1]).cuda(),
                                  'scale': torch.zeros([drk_num, 8]).cuda(),
                                  'theta': torch.zeros([drk_num, 8]).cuda(),
                                  'acutance': 5 * torch.ones([drk_num, 1]).cuda(),
                                  'color': torch.tensor(generate_distinct_colors(drk_num)).cuda(),
                                  'rotation': torch.tensor(generate_quaternions(drk_num)).cuda()
                                  }
    
    def re_init(self, xyz=None, scale=None, rotation=None, color=None, theta=None, acutance=None, l1l2_rate=None):
        self._xyz = xyz
        self._rotation = rotation
        self._scaling = scale
        self._features_dc = RGB2SH(color)[:, None]
        self._features_rest = torch.zeros([xyz.shape[0], (self.max_sh_degree+1)**2-1, 3]).float().cuda()
        self._thetas = theta
        self._acutance = acutance
        self._l1l2_rates = l1l2_rate
        self._opacity = self.inverse_opacity_activation(.98 * torch.ones_like(xyz[..., :1]))

    def re_init_single(self):
        self.re_init(**self.single_drk_args)
    
    def re_init_multiple(self):
        self.re_init(**self.multiple_drk_args)



class DRKUI:
    def __init__(self, xyz=None, scale=None, rotation=None, color=None, intr=None, W=512, H=512, theta=None, acutance=None, l1l2_rate=None, radius=5, white_background=True, gui_mode=True, drk=None) -> None:

        self.background = 1. if white_background else 0.
        self.should_save_screenshot = False

        if drk is not None:
            self.drk = drk
        else:
            drk = DRKDemoModel(sh_degree=3)
            drk.re_init(xyz=xyz, scale=scale, rotation=rotation, color=color, theta=theta, acutance=acutance, l1l2_rate=l1l2_rate)
            self.drk = drk

        # For UI
        self.visualization_mode = 'RGB'
        self.W = int(W)
        self.H = int(H)
        self.focal_x, self.focal_y = intr[:2]
        fovy = 2 * torch.arctan(intr[2] / intr[0])
        self.cam = OrbitCamera(W, H, r=radius, fovy=fovy.item())
        self.mode = "render"
        self.buffer_image = np.ones((self.H, self.W, 3), dtype=np.float32)
        self.cache_sort = True
        self.tile_culling = False
        self.gui_mode = gui_mode

        if self.gui_mode:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        dpg.destroy_context()

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

            with dpg.group(horizontal=True):
                dpg.add_text("DRK Demo: ")

                def callback_single_demot(sender, app_data):
                    if type(self.drk) is not DRKDemoModel:
                        self.drk = DRKDemoModel(sh_degree=3)
                    self.drk.re_init_single()
                dpg.add_button(
                    label="Single DRK",
                    tag="_button_single_demo",
                    callback=callback_single_demot
                )
                dpg.bind_item_theme("_button_single_demo", theme_button)

                def callback_multiple_demot(sender, app_data):
                    if type(self.drk) is not DRKDemoModel:
                        self.drk = DRKDemoModel(sh_degree=3)
                    self.drk.re_init_multiple()
                dpg.add_button(
                    label="Multiple DRK",
                    tag="_button_multiple_demo",
                    callback=callback_multiple_demot
                )
                dpg.bind_item_theme("_button_multiple_demo", theme_button)

            with dpg.group(horizontal=True):
                dpg.add_text("Render Options: ")

                def callback_cachesort(sender, app_data):
                    self.cache_sort = not self.cache_sort
                    if self.cache_sort:
                        dpg.configure_item("_button_sort", label="wo_sort")
                    else:
                        dpg.configure_item("_button_sort", label="w_sort")
                dpg.add_button(
                    label="wo_sort" if self.cache_sort else "w_sort",
                    tag="_button_sort",
                    callback=callback_cachesort
                )
                dpg.bind_item_theme("_button_sort", theme_button)

                def callback_tile_culling(sender, app_data):
                    self.tile_culling = not self.tile_culling
                    if self.tile_culling:
                        dpg.configure_item("_button_tile_culling", label="wo_cull")
                    else:
                        dpg.configure_item("_button_tile_culling", label="w_cull")
                dpg.add_button(
                    label="wo_cull" if self.tile_culling else "w_cull",
                    tag="_button_tile_culling",
                    callback=callback_tile_culling
                )
                dpg.bind_item_theme("_button_tile_culling", theme_button)

                def callback_screenshot(sender, app_data):
                    self.should_save_screenshot = True
                dpg.add_button(
                    label="screenshot", tag="_button_screenshot", callback=callback_screenshot
                )
                dpg.bind_item_theme("_button_screenshot", theme_button)

            # rendering options
            mode_list = ["render", "depth", "alpha", "normal"]
            with dpg.group(horizontal=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True
                dpg.add_combo(
                    mode_list,
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

            def get_button_change_mode_func(mode):
                def callback_change_mode(sender, app_data):
                    self.mode = mode
                    self.need_update = True
                return callback_change_mode
            # Add buttons to control vis mode
            with dpg.group(horizontal=True):
                dpg.add_text("Render Modes: ")
                for mode in mode_list:
                    dpg.add_button(
                        label=mode,
                        tag=f"_button_{mode}",
                        callback=get_button_change_mode_func(mode))
                    dpg.bind_item_theme(f"_button_{mode}", theme_button)
         
            with dpg.group(horizontal=True):
                def callback_set_l1l2rate(sender, app_data, idx=0):
                    self.drk._l1l2_rates.data[:] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label=f"eta",
                    min_value=-5.,
                    max_value=5.,
                    default_value=0.,
                    callback=callback_set_l1l2rate,
                    width=100,
                )

                def callback_set_acutance(sender, app_data, idx=0):
                    self.drk._acutance.data[:] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label=f"tau",
                    min_value=-.5,
                    max_value=5.,
                    default_value=0.,
                    callback=callback_set_acutance,
                    width=100,
                )

                def callback_set_opacity(sender, app_data, idx=0):
                    self.drk._opacity.data[:] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label=f"opacity",
                    min_value=-5.,
                    max_value=5.,
                    default_value=0.,
                    callback=callback_set_opacity,
                    width=100,
                )

            def get_idx_scale_callback(idx=0):
                def callback_set_scale(sender, app_data):
                    self.drk._scaling.data[:, idx] = app_data
                    self.need_update = True
                return callback_set_scale
            with dpg.group(horizontal=True):
                for i in range(4):
                    dpg.add_slider_float(
                        label=f"scale_{i}",
                        min_value=-1.2,
                        max_value=1.2,
                        default_value=0.,
                        callback=get_idx_scale_callback(i),
                        width=70,
                    )
            with dpg.group(horizontal=True):
                for i in range(4, 8):
                    dpg.add_slider_float(
                        label=f"scale_{i}",
                        min_value=-1.2,
                        max_value=1.2,
                        default_value=0.,
                        callback=get_idx_scale_callback(i),
                        width=70,
                    )

            def get_idx_theta_callback(idx=0):
                def callback_set_theta(sender, app_data):
                    self.drk._thetas.data[:, idx] = app_data
                    self.need_update = True
                return callback_set_theta
            with dpg.group(horizontal=True):
                for i in range(4):
                    dpg.add_slider_float(
                        label=f"theta_{i}",
                        min_value=-5.,
                        max_value=5.,
                        default_value=0.,
                        callback=get_idx_theta_callback(i),
                        width=70,
                    )
            with dpg.group(horizontal=True):
                for i in range(4, 8):
                    dpg.add_slider_float(
                        label=f"theta_{i}",
                        min_value=-5.,
                        max_value=5.,
                        default_value=0.,
                        callback=get_idx_theta_callback(i),
                        width=70,
                    )

            with dpg.group(horizontal=True):
                def callback_recenter(sender, app_data):
                    self.drk.recenter()
                dpg.add_button(
                    label="recenter", tag="_button_recenter", callback=callback_recenter
                )
                dpg.bind_item_theme("_button_recenter", theme_button)
                    
        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.mouse_loc = np.array(app_data)

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
            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)

        dpg.create_viewport(
            title="DRK",
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
        if self.gui_mode:
            while dpg.is_dearpygui_running():
                self.test_step()
                dpg.render_dearpygui_frame()
        else:
            while True:
                dxy = np.random.rand(2) * 10.
                self.cam.pan(dx=dxy[0], dy=dxy[1])
                self.test_step()

    @torch.no_grad()
    def test_step(self, specified_c2w=None):

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        if os.path.exists('output/screenshots/screenshot_camera.pickle') and self.should_save_screenshot:
            print('Use fixed camera for screenshot: ', 'output/screenshots/screenshot_camera.pickle')
            from utils.pickle_utils import load_obj
            cam_tmp = load_obj('output/screenshots/screenshot_camera.pickle')
            cam = MiniCam(height=self.H, width=self.W, fx=self.focal_x, fy=self.focal_y, c2w=torch.from_numpy(cam_tmp.pose).float().cuda())
        else:
            c2w = self.cam.pose if specified_c2w is None else specified_c2w
            cam = MiniCam(height=self.H, width=self.W, fx=self.focal_x, fy=self.focal_y, c2w=torch.from_numpy(c2w).float().cuda())
        bg_color = torch.tensor([self.background, self.background, self.background]).float().cuda()
        render_pkg   = self.drk.render_func(viewpoint_camera=cam, pc=self.drk, bg_color=bg_color, cache_sort=self.cache_sort, tile_culling=self.tile_culling)
        buffer_image = render_pkg[self.mode]

        if self.mode == 'depth':
            alpha = render_pkg['alpha'] if 'alpha' in render_pkg else torch.ones_like(buffer_image)
            valid_depth = buffer_image.reshape([-1])[alpha.reshape([-1]) > 0]
            valid_min, valid_max = valid_depth.min(), valid_depth.max()
            buffer_image = (buffer_image - valid_min) / (valid_max - valid_min + 1e-5)
            buffer_image = torch.cat([buffer_image, buffer_image, buffer_image], dim=0)
            buffer_image = buffer_image.clamp(0, 1)
        elif self.mode == 'alpha':
            buffer_image = torch.cat([buffer_image, buffer_image, buffer_image], dim=0)
        elif self.mode == 'normal':
            buffer_image = (buffer_image + 1) / 2

        buffer_image = torch.nn.functional.interpolate(
            buffer_image.unsqueeze(0),
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if self.should_save_screenshot:
            alpha = render_pkg['alpha'] if 'alpha' in render_pkg else None
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
            save_image(sv_image, 'output/screenshots/')
            self.should_save_screenshot = False

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

        if self.gui_mode:
            dpg.set_value(
                    "_texture", self.buffer_image
                ) 

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        return buffer_image
