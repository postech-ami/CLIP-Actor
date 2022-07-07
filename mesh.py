import torch
import utils
from utils import device
import copy
import numpy as np
import PIL
from pytorch3d.structures.meshes import Meshes
from pytorch3d.ops import SubdivideMeshes

class HumanMesh():
    def __init__(self, v, f, args, color=torch.tensor([0.5, 0.5, 0.5])):
        p3d_mesh_coarse = Meshes(v, f)  # PyTorch3D Mesh representation

        if args.mesh_subdivide:
            subdivide_mesh = SubdivideMeshes()  # PyTorch3D mesh subdivision class
            self.p3d_mesh = subdivide_mesh(p3d_mesh_coarse)
        else:
            self.p3d_mesh = p3d_mesh_coarse

        self.vertices = self.p3d_mesh.verts_list()[0].to(device)  # [6890, 3]
        self.faces = self.p3d_mesh.faces_list()[0].to(device)  # [13776, 3]

        mesh_vertex_normals = self.p3d_mesh.verts_normals_list()[0]
        self.vertex_normals = mesh_vertex_normals.to(device).float()
        self.vertex_normals = torch.nn.functional.normalize(self.vertex_normals)

        self.texture_map = None
        self.face_uvs = None
        self.face_normals = None
        self.set_mesh_color(color)


    def detach_all(self):
        self.vertices = self.vertices.detach()
        self.faces = self.faces.detach()
        self.vertex_normals = self.vertex_normals.detach()
        self.face_attributes = self.face_attributes.detach()

    def standardize_mesh(self, inplace=False):
        mesh = self if inplace else copy.deepcopy(self)
        return utils.standardize_mesh(mesh)

    def normalize_mesh(self, inplace=False):

        mesh = self if inplace else copy.deepcopy(self)
        return utils.normalize_mesh(mesh)

    def update_vertex(self, verts, inplace=False):

        mesh = self if inplace else copy.deepcopy(self)
        mesh.vertices = verts
        return mesh

    def set_mesh_color(self, color):
        self.texture_map = utils.get_texture_map_from_color(self,color)
        self.face_attributes = utils.get_face_attributes_from_color(self,color)

    def set_image_texture(self, texture_map, inplace=True):

        mesh = self if inplace else copy.deepcopy(self)

        if isinstance(texture_map, str):
            texture_map = PIL.Image.open(texture_map)
            texture_map = np.array(texture_map,dtype=np.float) / 255.0
            texture_map = torch.tensor(texture_map,dtype=torch.float).to(device).permute(2,0,1).unsqueeze(0)


        mesh.texture_map = texture_map
        return mesh

    def divide(self, inplace=True):

        mesh = self if inplace else copy.deepcopy(self)
        new_vertices, new_faces, new_vertex_normals, _ = utils.add_vertices(mesh)
        mesh.vertices = new_vertices
        mesh.faces = new_faces
        # mesh.face_uvs = new_face_uvs
        mesh.vertex_normals = new_vertex_normals
        return mesh

    def export(self, file, color=None):
        with open(file, "w+") as f:
            for vi, v in enumerate(self.vertices):
                if color is None:
                    f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
                else:
                    f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
                if self.vertex_normals is not None:
                    f.write("vn %f %f %f\n" % (self.vertex_normals[vi, 0], self.vertex_normals[vi, 1], self.vertex_normals[vi, 2]))
            for face in self.faces:
                f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))
