import numpy as np
import transforms3d


def matrix_from_t_m(t, m):
    return np.array(
        [[m[0][0], m[0][1], m[0][2], t[0]], [m[1][0], m[1][1], m[1][2], t[1]], [m[2][0], m[2][1], m[2][2], t[2]],
         [0, 0, 0, 1]])


def matrix_from_pose(pose):
    m = transforms3d.euler.euler2mat(pose[3], pose[4], pose[5], 'sxyz')
    return matrix_from_t_m(pose[:3], m)


def mul_matrix(m1, m2):
    return np.dot(m1, m2)


def xcmg_arm_fk(angles):
    boom_len = 3.6957
    arm_len = 1.62233
    operating_arm_height = 1.4
    boom_init_angle = 1.0
    arm_init_angle = 1.5
    arm_offset_x = 0.0
    arm_offset_y = -0.1
    xcmg_simple_fk = [[0, 0, 0, 0, boom_init_angle, 0],
                      [0, 0, boom_len, 0, arm_init_angle, 0],
                      [0, 0, arm_len, 0, 0, 0],
                      [0, 0, operating_arm_height, 0, 0, 0],
                      [arm_offset_x, arm_offset_y, 0, 0, 0, 0]]
    tf_arm1_base = matrix_from_pose(xcmg_simple_fk[0])
    tf_arm1new_arm1 = matrix_from_pose([0, 0, 0, 0, angles[0], 0])
    tf_arm2_arm1 = matrix_from_pose(xcmg_simple_fk[1])
    tf_arm2new_arm2 = matrix_from_pose([0, 0, 0, 0, angles[1], 0])
    tf_gripper_arm2 = matrix_from_pose(xcmg_simple_fk[2])
    tf_grippernew_gripper = matrix_from_pose([0, 0, 0, 0, angles[2], 0])
    tf_grippernew_arm2 = mul_matrix(tf_gripper_arm2, tf_grippernew_gripper)
    tf_arm2new_arm1 = mul_matrix(tf_arm2_arm1, tf_arm2new_arm2)
    tf_arm1new_base = mul_matrix(tf_arm1_base, tf_arm1new_arm1)
    tf_grippernew_arm1new = mul_matrix(tf_arm2new_arm1, tf_grippernew_arm2)
    tf_grippernew_base = mul_matrix(tf_arm1new_base, tf_grippernew_arm1new)
    tf_uppernew_upper = matrix_from_pose([0, 0, 0, 0, 0, angles[3]])
    tf_uppernew_base = mul_matrix(matrix_from_pose(xcmg_simple_fk[3]), tf_uppernew_upper)
    tf_armbase_upper = matrix_from_pose(xcmg_simple_fk[4])
    tf_swingbase_arm = mul_matrix(np.linalg.inv(tf_armbase_upper), np.linalg.inv(tf_uppernew_base))
    tf_grippernew_swingbase = mul_matrix(np.linalg.inv(tf_swingbase_arm), tf_grippernew_base)
    return tf_grippernew_swingbase


def get_arm_end_xyz(pos):
    pos = np.array(pos)
    if pos.shape[0] == 3:
        pos = np.insert(pos, 2, 0)
    return xcmg_arm_fk(pos)[:3, 3]
