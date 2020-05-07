import bpy
import os
import sys
import time
import math
import random
import numpy as np


def blender_print(*args, **kwargs):
    print (*args, **kwargs, file=sys.stderr)


def yup_to_zup(x, y, z):
    return x, -z, y


def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3, 3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3, 3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R


class RenderpplBlenderTool:
    def __init__(self):
        # As of now, for each instance of blender executive,
        # we only load one model at a time (newly load one will clear up old one)
        # this stores the blender objective instance of the model
        self.current_model = None
        self.current_geo = None
        self.current_material = None
        self.camera = None
        # action_pool is a mapping of action names to tuple (property, data)
        # where property specifies the definition of animation data
        self.action_pool = {}

        self.current_frame = 0

        self.reset()
        self.init_camera()

    def reset(self):
        # clean up the whole scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False, confirm=False)

        # deselect all the objects
        bpy.ops.object.select_all(action='DESELECT')

        # reset variables
        self.current_model = None
        self.current_geo = None
        self.camera = None
        self.action_pool = {}
        self.current_frame = 0

    def import_model_fbx(self, fbxpath, replace_name=None, children_id=0):
        bpy.ops.import_scene.fbx(filepath=fbxpath, use_anim=True)
        if replace_name is not None:
            bpy.context.object.name = replace_name
        model = bpy.context.scene.objects[bpy.context.object.name]
        obj = model.children[children_id]
        for c in model.children:
            if 'geo' in c:
                obj = c
        self.current_model = model
        self.current_geo = obj

    def import_action_fbx(self, fbxpath, replace_name=None):
        bpy.ops.import_scene.fbx(filepath=fbxpath, use_anim=True)
        if replace_name is not None:
            bpy.context.object.name = replace_name
        action_name = bpy.context.object.name
        properties = [p.identifier for p in bpy.context.scene.objects[action_name].animation_data.bl_rna.properties if
                      not p.is_readonly]
        self.action_pool[action_name] = (properties, bpy.context.scene.objects[action_name].copy().animation_data)

    def import_material(self, base_color_path, normal_map_path, material_name=None):
        # Define names and paths
        if material_name is None:
            material_name = "Example_Material"

        # Create a material
        material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links

        # Since you want a Principled BSDF and the Material Output node
        # in your material, we can re-use the nodes that are automatically
        # created.
        principled_bsdf = nodes.get("Principled BSDF")
        material_output = nodes.get("Material Output")

        # Create Image Texture node and load the base color texture
        if base_color_path is not None:
            base_color = nodes.new('ShaderNodeTexImage')
            base_color.image = bpy.data.images.load(base_color_path)

            # Connect the base color texture to the Principled BSDF
            links.new(principled_bsdf.inputs["Base Color"], base_color.outputs["Color"])

        # Create Image Texture node and load the normal map
        if normal_map_path is not None:
            normal_tex = nodes.new('ShaderNodeTexImage')
            normal_tex.image = bpy.data.images.load(normal_map_path)

            # Set the color space to non-color, since normal maps contain
            # the direction of the surface normals and not color data
            normal_tex.image.colorspace_settings.name = "Non-Color"

            # Create the Displacement node
            displacement = nodes.new('ShaderNodeDisplacement')

            # Connect the normal map to the Displacement node
            links.new(displacement.inputs["Height"], normal_tex.outputs["Color"])

            # Connect the Displacement node to the Material Output node
            links.new(material_output.inputs["Displacement"], displacement.outputs["Displacement"])

        # Assign it to object
        obj = self.current_geo
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

        self.current_material = material

    def import_world_lighting(self, hdripath):
        bpy.data.scenes['Scene'].render.engine = 'CYCLES'
        world = bpy.data.worlds['World']
        world.use_nodes = True
        bg = world.node_tree.nodes['Background']

        if hdripath is None:
            bg.inputs[0].default_value[:3] = (0.5, .1, 0.6)
            bg.inputs[1].default_value = 1.0
        else:
            enode = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
            enode.image = bpy.data.images.load(hdripath)
            world.node_tree.links.new(bg.inputs["Color"], enode.outputs["Color"])

    def apply_action(self, action_name):
        model = self.current_model
        model.animation_data_clear()
        model.animation_data_create()

        bpy.ops.object.select_all(action='DESELECT')
        model.select_set(True)

        properties, act = self.action_pool[action_name]
        for prop in properties:
            setattr(model.animation_data, prop, getattr(act, prop))

    def set_frame(self, frame_id):
        bpy.context.scene.frame_set(frame_id)

    def set_resolution(self, width, height):
        bpy.context.scene.render.resolution_x = width
        bpy.context.scene.render.resolution_y = height

    def set_render(self, num_samples=1000, use_motion_blur=True, use_transparent_bg=True, use_denoising=True):
        scene = bpy.data.scenes["Scene"]
        scene.camera = self.camera

        scene.render.image_settings.file_format = 'PNG'
        scene.render.engine = 'CYCLES'
        scene.render.use_motion_blur = use_motion_blur

        scene.render.film_transparent = use_transparent_bg
        scene.view_layers[0].cycles.use_denoising = use_denoising

        scene.cycles.samples = num_samples

    def render_to_img(self, imgpath, calipath=None):
        scene = bpy.data.scenes["Scene"]
        scene.render.filepath = imgpath
        bpy.ops.render.render(write_still=1)

        if calipath is not None:
            model_view, projection = self.get_calibration()
            np.savetxt(calipath, np.concatenate([model_view, projection], axis=0))

    def get_calibration(self):
        # model_view
        model_view_zup = np.eye(4)
        rot_cam = make_rotate(*self.camera.rotation_euler)
        rot_mat_zup = np.linalg.inv(rot_cam)
        model_view_zup[:3, :3] = rot_mat_zup
        model_view_zup[:3, 3] = -np.dot(rot_mat_zup, self.camera.location)
        yup_to_zup_mat = np.eye(4)
        yup_to_zup_mat[:3, :3] = make_rotate(math.radians(90), 0, 0)
        model_view = np.matmul(model_view_zup, yup_to_zup_mat)

        # projection
        projection_mat = np.eye(4)
        if self.camera.data.type == 'ORTHO':
            ortho_scale = self.camera.data.ortho_scale
            near = self.camera.data.clip_start
            far = self.camera.data.clip_end
            projection_mat[0, 0] = 2 / ortho_scale
            projection_mat[1, 1] = 2 / ortho_scale
            projection_mat[2, 2] = -2 / (far - near)
            projection_mat[2, 3] = -(far + near) / (far - near)
        else:
            print('ask Zeng to implement matrix for perspective')

        return model_view, projection_mat

    def init_camera(self):
        bpy.ops.object.camera_add(location=yup_to_zup(0, 1.0, 1.0))
        self.camera = bpy.context.object
        self.camera.rotation_euler[0] = math.radians(90)
        self.camera.data.type = 'ORTHO'
        self.camera.data.clip_start = 0.1
        self.camera.data.clip_end = 1000
        self.camera.data.ortho_scale = 2.56

    # this set camera sets a camera looking at lookat from dist far, positioned at rad degree between x axis, and at the same height as lookat.
    def set_camera_ortho_pifu(self, lookat, dist, rad, near, far, ortho_scale=2.56):

        cam_position = lookat
        cam_position[0] += math.sin(-rad) * dist
        cam_position[2] += math.cos(-rad) * dist
        self.camera.location = yup_to_zup(*cam_position)
        self.camera.rotation_euler[0] = math.radians(90)
        self.camera.rotation_euler[2] = -rad
        self.camera.data.clip_start = near
        self.camera.data.clip_end = far
        self.camera.data.ortho_scale = ortho_scale

    def export_mesh(self, objpath):
        bpy.ops.object.select_all(action='DESELECT')
        obj = self.current_geo
        obj.select_set(True)
        bpy.ops.export_scene.obj(filepath=objpath,
                                 check_existing=True,
                                 use_animation=False,
                                 use_normals=True,
                                 use_uvs=True,
                                 use_materials=True, #@ruilong
                                 use_selection=True)

    def export_skeleton(self, skpath):
        model = self.current_model
        with open(skpath, "w") as file:
            for bone in model.pose.bones:
                bone_world = model.matrix_world @ bone.head * 100.0
                file.write("%s %f %f %f \n" % (bone.name,
                                               bone_world[0],
                                               bone_world[2],
                                               -bone_world[1]))

    # @ruilong
    def get_action_duration(self, action_name):
        _, act = self.action_pool[action_name]
        start_frame = act.action.frame_range[0]
        end_frame = act.action.frame_range[1]
        return int(end_frame - start_frame + 1)

    # @ruilong
    def get_skeleton(self):
        model = self.current_model
        names = []
        skeleton = []
        for bone in model.pose.bones:
            bone_world = model.matrix_world @ bone.head * 100.0
            names.append(bone.name)
            skeleton.append([bone_world[0], bone_world[2], -bone_world[1]])
        skeleton = np.array(skeleton, dtype=np.float32)
        return names, skeleton

    # @ruilong
    def enable_gpus(self, device_ids=[0], use_cpus=False):
        preferences = bpy.context.preferences
        cycles_preferences = preferences.addons["cycles"].preferences
        devices, _ = cycles_preferences.get_devices()
        
        gpu_devices = []
        for device in devices:
            if device.type == "CPU":
                device.use = use_cpus
            else:
                gpu_devices.append(device)
    
        for id, device in enumerate(gpu_devices):
            if id in device_ids:
                device.use = True
            else:
                device.use = False
                
        cycles_preferences.compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = "GPU"


if __name__ == '__main__':
    tool = RenderpplBlenderTool()

    test_model_name = 'rp_adanna_rigged_001'
    SRC_ROOT = 'E:\\blender_tool'
    test_model_file = os.path.join(SRC_ROOT, '%s_u3d.fbx' % test_model_name)
    test_action_name = 'T-Pose'
    test_hdri_file = os.path.join(SRC_ROOT, 'small_cathedral_1k.hdr')
    test_action_file = os.path.join(SRC_ROOT, './T-Pose.fbx')
    test_dif_tex = os.path.join(SRC_ROOT, 'tex', 'rp_adanna_rigged_001_dif.jpg')

    tool.import_model_fbx(test_model_file, test_model_name)
    tool.import_material(test_dif_tex, None)
    tool.import_world_lighting(test_hdri_file)
    tool.import_action_fbx(test_action_file, test_action_name)

    tool.apply_action(test_action_name)
    tool.set_frame(1)
    tool.set_resolution(512, 512)
    tool.set_render()
    tool.set_camera_ortho_pifu([0, 1, 0], 3, math.radians(90), 1.5, 4.5)

    tool.export_mesh(os.path.join(SRC_ROOT, 'export.obj'))
    tool.export_skeleton(os.path.join(SRC_ROOT, 'export_sk.txt'))
    tool.render_to_img(os.path.join(SRC_ROOT, 'export.png'), os.path.join(SRC_ROOT, 'export_calib.txt'))
