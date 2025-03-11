import os
import torch
import numpy as np
from PIL import Image
from gui_utils.drk_gui import DRKUI
from plyfile import PlyData
import tqdm



class DefaultPipe:
    def __init__(self) -> None:
        self.convert_SHs_python = False
        self.debug = False
        self.compute_cov3D_python = False
default_pipe = DefaultPipe()


def parse_mtl(mtl_filename):
    materials = {}
    current_material = None

    with open(mtl_filename, 'r') as mtl_file:
        for line in mtl_file:
            if line.startswith('newmtl'):
                current_material = line.split()[1]
                materials[current_material] = {}
            elif line.startswith('Kd') and current_material:
                materials[current_material]['Kd'] = list(map(float, line.split()[1:4]))
            elif line.startswith('map_Kd') and current_material:
                materials[current_material]['map_Kd'] = line.split()[1]

    return materials

def parse_obj(obj_filename, materials):
    vertices = []
    faces = []
    uvs = []
    face_materials = []

    with open(obj_filename, 'r') as obj_file:
        current_material = None

        for line in obj_file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.split()[1:4])))
            elif line.startswith('vt '):
                uvs.append(list(map(float, line.split()[1:3])))
            elif line.startswith('usemtl'):
                current_material = line.split()[1]
            elif line.startswith('f '):
                face = []
                uv_face = []
                for v in line.split()[1:]:
                    vals = v.split('/')
                    face.append(int(vals[0]) - 1)
                    if len(vals) > 1 and vals[1]:
                        uv_face.append(int(vals[1]) - 1)
                faces.append((face, uv_face, current_material))

    return np.array(vertices), faces, uvs

def calculate_face_color(face, uvs, materials, material_name, texture_img=None):
    if texture_img:
        uv_coords = [uvs[i] for i in face[1]]
        colors = [texture_img.getpixel((uv[0] * texture_img.width, (1 - uv[1]) * texture_img.height)) for uv in uv_coords]
        avg_color = np.mean(colors, axis=0) / 255
    elif materials is None:
        avg_color = np.array([0.5, 0.5, 0.5])
    else:
        avg_color = materials[material_name]['Kd']

    return avg_color

def load_texture_image(materials, material_name):
    if materials is None:
        return None
    if 'map_Kd' in materials[material_name]:
        texture_path = materials[material_name]['map_Kd']
        return Image.open(os.path.join('/home/yihua/disk8T/siga2024/data/lowpoly/meshes', texture_path))
    return None


def process_obj_with_mtl(obj_filename, mtl_filename=None):
    materials = parse_mtl(mtl_filename) if mtl_filename is not None else None
    vertices, faces, uvs = parse_obj(obj_filename, materials)

    face_colors = []
    for face in tqdm.tqdm(faces):
        material_name = face[2]
        texture_img = load_texture_image(materials, material_name)
        color = calculate_face_color(face, uvs, materials, material_name, texture_img)
        face_colors.append(color)
    face_vertices = np.array([vertices[f[0]] for f in faces])
    face_colors = np.array(face_colors)
    return face_vertices, face_colors


def process_ply(ply_filename):
    ply_data = PlyData.read(ply_filename)
    
    vertex_data = ply_data['vertex']
    vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
    
    if 'red' in vertex_data and 'green' in vertex_data and 'blue' in vertex_data:
        vertex_colors = np.vstack((vertex_data['red'], vertex_data['green'], vertex_data['blue'])).T
    else:
        vertex_colors = None
    
    face_data = ply_data['face']
    faces = np.array(face_data['vertex_indices'])
    
    face_colors = []
    for face in faces:
        if vertex_colors is not None:
            color = np.mean(vertex_colors[face], axis=0) / 255
        else:
            color = np.array([128, 128, 128]) / 255
        face_colors.append(color)

    face_vertices = np.array([vertices[f] for f in faces])
    face_colors = np.array(face_colors)
    return face_vertices, face_colors


def process_triangles(vertices):
    centroids = np.mean(vertices, axis=1)
    rotation_matrices = np.zeros((vertices.shape[0], 3, 3))
    polar_coordinates = np.zeros((vertices.shape[0], 3, 2))
    
    for i, triangle in tqdm.tqdm(enumerate(vertices)):
        v0 = triangle[0] - centroids[i]
        ab = triangle[1] - triangle[0]
        ac = triangle[2] - triangle[0]
        
        x_axis = v0 / np.linalg.norm(v0)
        
        normal = np.cross(ab, ac)
        normal /= np.linalg.norm(normal)
        
        y_axis = np.cross(normal, x_axis)
        
        rotation_matrix = np.array([x_axis, y_axis, normal]).T
        rotation_matrices[i] = rotation_matrix
        
        transformed_vertices = np.dot(rotation_matrix.T, (triangle - centroids[i]).T).T
        
        for j, vertex in enumerate(transformed_vertices):
            r = np.linalg.norm(vertex[:2])
            theta = np.arctan2(vertex[1], vertex[0])
            theta = np.where(theta < 0, theta + 2 * np.pi, theta)
            polar_coordinates[i, j] = np.array([r, theta])
    
    return centroids, rotation_matrices, polar_coordinates


def adjust_colors(normals, base_colors, light_direction):
    light_direction = light_direction / torch.norm(light_direction)
    adjusted_colors = torch.zeros_like(base_colors)
    normals = normals / torch.norm(normals, dim=1, keepdim=True)
    cos_theta = torch.clamp(torch.sum(normals * light_direction, dim=1), min=0)
    adjusted_colors = base_colors * (0.8 + 0.2 * cos_theta.unsqueeze(1))
    return adjusted_colors


from scene.gaussian_model import DRKModel, RGB2SH
class MeshKernel(DRKModel):
    def __init__(self, sh_degree, means3D, scales, thetas, rotations, colors):
        super().__init__(sh_degree, kernel_K=8)
        self._xyz = means3D
        self._scaling = scales
        self._thetas = thetas
        self._rotations = rotations
        self._features_dc = RGB2SH(colors)[:, None]
        self._features_rest = torch.zeros([xyz.shape[0], (self.max_sh_degree+1)**2-1, 3]).float().cuda()  # N, D, 3
    @property
    def get_xyz(self):
        return self._xyz
    @property
    def get_rotation(self):
        return self._rotations.reshape([-1, 9])
    @property
    def get_scaling(self):
        return self._scaling
    @property
    def get_thetas(self):
        return self._thetas
    @property
    def get_l1l2rates(self):
        return torch.ones([self._xyz.shape[0], 1]).cuda()
    @property
    def get_acutance(self):
        return .99 * torch.ones([self._xyz.shape[0], 1]).cuda()
    @property
    def get_opacity(self):
        return torch.ones([self._xyz.shape[0], 1]).cuda()
    

class ComposedKernel(DRKModel):
    def __init__(self, sh_degree, mesh_drk=None, recon_drk=None, **kwargs):
        super().__init__(sh_degree)
        self.mesh_drk = mesh_drk
        self.recon_drk = recon_drk
        if self.recon_drk._xyz.shape[0] == 0:
            self.recon_drk = None
    
    @property
    def get_xyz(self):
        if self.recon_drk is None:
            return self.mesh_drk.get_xyz
        else:
            return torch.cat([self.mesh_drk.get_xyz, self.recon_drk.get_xyz], dim=0)
    @property
    def get_rotation(self):
        if self.recon_drk is None:
            return self.mesh_drk.get_rotation
        else:
            return torch.cat([self.mesh_drk.get_rotation, self.recon_drk.get_rotation], dim=0)
    @property
    def get_scaling(self):
        if self.recon_drk is None:
            return self.mesh_drk.get_scaling
        else:
            return torch.cat([self.mesh_drk.get_scaling, self.recon_drk.get_scaling], dim=0)
    @property
    def get_thetas(self):
        if self.recon_drk is None:
            return self.mesh_drk.get_thetas
        else:
            return torch.cat([self.mesh_drk.get_thetas, self.recon_drk.get_thetas], dim=0)
    @property
    def get_l1l2rates(self):
        if self.recon_drk is None:
            return self.mesh_drk.get_l1l2rates
        else:
            return torch.cat([self.mesh_drk.get_l1l2rates, self.recon_drk.get_l1l2rates], dim=0)
    @property
    def get_acutance(self):
        if self.recon_drk is None:
            return self.mesh_drk.get_acutance
        else:
            return torch.cat([self.mesh_drk.get_acutance, self.recon_drk.get_acutance], dim=0)
    @property
    def get_opacity(self):
        if self.recon_drk is None:
            return self.mesh_drk.get_opacity
        else:
            return torch.cat([self.mesh_drk.get_opacity, self.recon_drk.get_opacity], dim=0)
    @property
    def get_features(self):
        if self.recon_drk is None:
            return self.mesh_drk.get_features
        else:
            return torch.cat([self.mesh_drk.get_features, self.recon_drk.get_features], dim=0)


if __name__ == "__main__":
    scene_path = 'THE-PATH-TO-DRK-CHECKPOINT/point_cloud.ply'
    mesh_path_list = ['./meshes/dog.obj']

    drk = DRKModel(3)
    if os.path.exists(scene_path):
        drk.load_ply(scene_path)
    
    face_vertices_list, face_colors_list = [], []
    for mesh_path in mesh_path_list: 
        if not os.path.exists(mesh_path):
            mesh_path = mesh_path.replace('.obj', '.ply') if mesh_path.endswith('.obj') else mesh_path.replace('.ply', '.obj')
        if mesh_path.endswith('.ply'):
            face_vertices, face_colors = process_ply(mesh_path)
        else:
            face_vertices, face_colors = process_obj_with_mtl(mesh_path, mesh_path.replace('.obj', '.mtl'))
        face_vertices_list.append(face_vertices)
        face_colors_list.append(face_colors)
    face_vertices = np.concatenate(face_vertices_list, axis=0)
    face_colors = np.concatenate(face_colors_list, axis=0)

    means3D, rotations, polar_coords = process_triangles(face_vertices)
    K = 8
    gs_num = means3D.shape[0]
    xyz = torch.tensor(means3D).float().cuda()
    l1l2_rate = - 99999 * torch.ones([gs_num, 1]).cuda()
    scale = torch.zeros([gs_num, K]).cuda()
    scale[:, 0:3] = torch.tensor(polar_coords[:, 0:3, 0]).cuda().float()
    scale = scale * ((np.log(2))**.5) * (1 + 2e-2)
    scale_ = scale.clone()
    scale[:, 1:4] = scale_[:, 1:2]
    scale[:, 4:K] = scale_[:, 2:3]

    theta = torch.zeros([gs_num, K]).cuda()
    theta[:, 0:2] = torch.tensor(polar_coords[:, 1:3, 1]).cuda().float() / np.pi / 2
    theta[:, 2:] = 1
    theta_ = theta.clone()
    theta[:, 0:3] = theta_[:, 0:1]
    theta[:, 3:K-1] = theta_[:, 1:2]
    theta[:, K-1:]  = 1.

    color = torch.tensor(face_colors).cuda().float()
    rotation = torch.tensor(rotations).cuda().float()

    light_direction = torch.tensor([-1., -1., -1.]).cuda().float()
    adjusted_colors = adjust_colors(rotation.reshape([-1, 3, 3])[:, 2], color, light_direction)

    mesh_gs = MeshKernel(3, xyz, scale, theta, rotation, adjusted_colors)

    compose_gs = ComposedKernel(3, mesh_gs, drk)

    W, H = 800, 800
    fov = 45 * torch.pi / 180
    focal = W / 2 / np.tan(fov / 2)

    pp = torch.tensor((W/2, H/2)).cuda()
    intr = torch.tensor([focal, focal, pp[0], pp[1]]).cuda().float()

    gui = DRKUI(intr=intr, W=W, H=H, drk=compose_gs, white_background=True)
    gui.render()
