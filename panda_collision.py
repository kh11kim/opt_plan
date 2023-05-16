import kinjax
from jaxlie import SO3, SE3
import jax
from jax import Array
import jax.numpy as jnp
import jax_dataclasses as jdc
from typing import Tuple
import trimesh

"""Forward Kinematics of PANDA"""

def T_yaw(theta):
    return SE3.from_rotation(SO3.from_rpy_radians(0,0,theta))

def get_fk_fn(urdf_path):
    link_dict, joint_dict = kinjax.get_link_joint_dict(urdf_path)
    for name, info in joint_dict.items():
        xyz = info["origin_xyz"]
        rot = SO3.from_rpy_radians(*info["origin_rpy"])
        info["T_offset"] = SE3.from_rotation_and_translation(rot, xyz)

    def panda_fk(q, finger_width=0.08):
        T0 = SE3.identity()
        T1 = T0 @ joint_dict['panda_joint1']["T_offset"] @ T_yaw(q[0])
        T2 = T1 @ joint_dict['panda_joint2']["T_offset"] @ T_yaw(q[1])
        T3 = T2 @ joint_dict['panda_joint3']["T_offset"] @ T_yaw(q[2])
        T4 = T3 @ joint_dict['panda_joint4']["T_offset"] @ T_yaw(q[3])
        T5 = T4 @ joint_dict['panda_joint5']["T_offset"] @ T_yaw(q[4])
        T6 = T5 @ joint_dict['panda_joint6']["T_offset"] @ T_yaw(q[5])
        T7 = T6 @ joint_dict['panda_joint7']["T_offset"] @ T_yaw(q[6])
        Thand = T7 @ joint_dict['panda_hand_joint']["T_offset"]
        Tfinger_l = Thand @ joint_dict['panda_finger_joint1']["T_offset"] \
            @ SE3.from_rotation_and_translation(SO3.identity(), jnp.array([0,finger_width/2,0]))
        rot_pi = SE3.from_rotation(SO3.from_rpy_radians(0,0,jnp.pi))
        Tfinger_r = Thand @ rot_pi @ joint_dict['panda_finger_joint2']["T_offset"] \
            @ SE3.from_rotation_and_translation(SO3.identity(), jnp.array([0,finger_width/2,0]))
        Tee = Thand @ joint_dict['panda_grasptarget_hand']["T_offset"]
        return jnp.vstack([T.parameters() for T in \
                        (T0, T1, T2, T3, T4, T5, T6, T7, Thand, Tfinger_l, Tfinger_r, Tee)])
    return panda_fk

@jdc.pytree_dataclass
class LinkPointCloud:
    link_num: int
    _points: Array
    num_points: int
    def transform(self, pose:SE3):
        return jax.vmap(pose.apply)(self._points)
    
@jdc.pytree_dataclass
class RobotPointCloud:
    pcs: Tuple[LinkPointCloud]
    num_points: int
    def apply_transforms(self, poses:Array):
        robot_points = jnp.zeros((self.num_points, 3))
        idx = 0
        for i in range(len(self.pcs)):
            link_points = self.pcs[i].transform(SE3(poses[i]))
            robot_points = robot_points.at[idx:idx+len(link_points)].set(link_points)
            idx += len(link_points)
        return robot_points

def get_pointclouds(mesh_path, obj_file_names, num_points_per_link=100):
    linkpcs = []
    num_points = 0
    for name in obj_file_names:
        mesh = trimesh.load(mesh_path/ (name+".obj"))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()
        points, _ = trimesh.sample.sample_surface_even(mesh, num_points_per_link)
        linkpcs.append(LinkPointCloud(name, points, len(points)))
        num_points += len(points)
    robotpc = RobotPointCloud(tuple(linkpcs), num_points)
    return robotpc

