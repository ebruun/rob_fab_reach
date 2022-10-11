import math
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm

from compas.datastructures import Mesh

from compas_fab.backends import RosClient
from compas_fab.backends import RosValidationError

from compas_fab.robots import PlanningScene
from compas_fab.robots import CollisionMesh

from compas.geometry import Point
from compas.geometry import Frame
from compas.geometry import Plane


def get_directory(b, c):

    a = os.getcwd()

    path = os.path.join(a, b, c)

    return path


def save_JsonFile(path, name, vec, result_dict):

    filepath = os.path.join(path, "{}.json".format(name))

    try:
        # if file exists
        with open(filepath, "w") as f:
            v = list(vec)

            data = {}
            data["vector"] = v
            data["results"] = result_dict

            json.dump(data, f, indent=4)
            f.close()
    except TypeError:
        print("write to file not working")


def merge_JsonFiles(in_filepaths, out_filepath):

    data_combined = {}

    with open(in_filepaths[0], "r") as f:
        data_combined = json.load(f)
        f.close()

    for in_filepath in in_filepaths[1:]:
        with open(in_filepath, "r") as f:
            data = json.load(f)
            data_combined["results"].update(data["results"])
            f.close()

    with open(out_filepath, "w") as f:
        json.dump(data_combined, f, indent=4)
        f.close()


def generate_filepaths(path, rob_num, index, merge_num):
    filepaths = []
    for i in range(1, merge_num + 1):
        name = "{}_vec{:0>2}_{}".format(rob_num, index, i)
        filepaths.append(os.path.join(path, "{}.json".format(name)))

    return filepaths


def plot_dots(idx_soln, idx_no_soln):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    for p in idx_soln:
        ax.scatter(p.x, p.y, p.z, color="blue")

    for p in idx_no_soln:
        ax.scatter(p.x, p.y, p.z, color="red")

    ax.set_xlabel("$X$", fontsize=20)
    ax.set_ylabel("$Y$")
    ax.set_zlabel("$Z$")

    plt.show()


####################################


def connect_and_scene():
    ros_client = RosClient("localhost")
    ros_client.run()

    robot = ros_client.load_robot()

    scene = PlanningScene(robot)
    mesh = Mesh.from_stl("./3dm/ground.stl")
    cm = CollisionMesh(mesh, "ground")
    scene.add_collision_mesh(cm)

    return robot


def points_ranges(rob_num, n):

    if rob_num == "rob1":
        # ranges = {
        #     "i" : np.arange(-1.4,4.21,0.2),
        #     "j" : np.arange(0.4,4.61,0.2),
        #     "k" : np.arange(-0.4,3.41,0.2),
        #     "name" : "{}_vec{:0>2}_1".format(rob_num,n+1)
        # }
        # ranges = {
        #     "i" : np.arange(-1.5,4.31,0.2),
        #     "j" : np.arange(0.3,4.51,0.2),
        #     "k" : np.arange(-0.3,3.31,0.2),
        #     "name" : "{}_vec{:0>2}_2".format(rob_num,n+1)
        # }
        # ranges = {
        #     "i" : np.arange(-1.4,4.21,0.2),
        #     "j" : np.arange(0.4,4.61,0.2),
        #     "k" : np.arange(-0.3,3.31,0.2),
        #     "name" : "{}_vec{:0>2}_3".format(rob_num,n+1)
        # }
        ranges = {
            "i": np.arange(-1.5, 4.31, 0.2),
            "j": np.arange(0.3, 4.51, 0.2),
            "k": np.arange(-0.4, 3.41, 0.2),
            "name": "{}_vec{:0>2}_4".format(rob_num, n + 1),
        }

    elif rob_num == "rob2":
        # ranges = {
        #     "i" : np.arange(-1.4,4.21,0.2),
        #     "j" : np.arange(-1.2,3.01,0.2),
        #     "k" : np.arange(-0.4,3.41,0.2),
        #      "name" : "{}_vec{:0>2_1".format(rob_num,n+1)
        # }
        # ranges = {
        #     "i": np.arange(-1.5, 4.31, 0.2),
        #     "j": np.arange(-1.1, 3.11, 0.2),
        #     "k": np.arange(-0.3, 3.31, 0.2),
        #     "name": "{}_vec{:0>2}_2".format(rob_num, n + 1),
        # }
        # ranges = {
        #     "i" : np.arange(-1.4,4.21,0.2),
        #     "j" : np.arange(-1.2,3.01,0.2),
        #     "k" : np.arange(-0.3,3.31,0.2),
        #      "name" : "{}_vec{:0>2_3".format(rob_num,n+1)
        # }
        # ranges = {
        #     "i" : np.arange(-1.5,4.31,0.2),
        #     "j" : np.arange(-1.1,3.11,0.2),
        #     "k" : np.arange(-0.4,3.41,0.2),
        #     "name" : "{}_vec{:0>2}_4".format(rob_num,n+1)
        # }
        ranges = {
            "i": np.arange(-1.5, 4.31, 0.1),
            "j": np.arange(-1.2, 3.11, 0.1),
            "k": np.arange(-0.4, 3.41, 0.1),
            "name": "{}_vec{:0>2}_combo".format(rob_num, n + 1),
        }
    elif rob_num == "rob3":
        # ranges = {
        #     "i" : np.arange(1.8,4.21,0.2),
        #     "j" : np.arange(0.4,3.01,0.2),
        #     "k" : np.arange(-0.4,3.41,0.2),
        #     "name" : "{}_vec{:0>2}_1".format(rob_num,n+1)
        # }
        ranges = {
            "i": np.arange(1.7, 4.31, 0.2),
            "j": np.arange(0.3, 3.11, 0.2),
            "k": np.arange(-0.3, 3.31, 0.2),
            "name": "{}_vec{:0>2}_2".format(rob_num, n + 1),
        }
        # ranges = {
        #     "i" : np.arange(1.8,4.21,0.2),
        #     "j" : np.arange(0.4,3.01,0.2),
        #     "k" : np.arange(-0.3,3.31,0.2),
        #     "name" : "{}_vec{:0>2}_3".format(rob_num,n+1)
        # }
        # ranges = {
        #     "i" : np.arange(1.7,4.31,0.2),
        #     "j" : np.arange(0.3,3.11,0.2),
        #     "k" : np.arange(-0.4,3.41,0.2),
        #     "name" : "{}_vec{:0>2}_4".format(rob_num,n+1)
        # }

    num_points = len(ranges["i"]) * len(ranges["j"]) * len(ranges["k"])

    return ranges, num_points


def points_in_box(corner, axis, ranges):
    for i in ranges["i"]:
        for j in ranges["j"]:
            for k in ranges["k"]:
                x = round(i, 1) + corner.x
                y = round(j, 1) + corner.y
                z = round(k, 1) + corner.z

                plane = Plane((x, y, z), axis)
                f = Frame.from_plane(plane)

                yield [Frame(f.point, f.xaxis, f.yaxis)]


def axis_gen(samples=100):
    # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        vec = np.array([x, y, z])
        vec[abs(vec) < 1e-14] = 0.0
        vec = vec / np.linalg.norm(vec)

        yield vec


def robot_config(robot, rob_num):
    config_scaled = robot.zero_configuration()

    if rob_num == "rob1":
        config1 = [-90.0, 0.0, 0.0, 0.0, 90.0, 180, 100]
        config2 = [90.0, 0.0, 0.0, 0.0, 90.0, 180, 100]
        config3 = [180.0, 0.0, 0.0, 0.0, 90.0, 180.0]
    elif rob_num == "rob2":
        config1 = [90.0, 0.0, 0.0, 0.0, 90.0, 180, 3800]
        config2 = [90.0, 0.0, 0.0, 0.0, 90.0, 180, 3900]
        config3 = [180.0, 0.0, 0.0, 0.0, 90.0, 180.0]
    elif rob_num == "rob3":
        config1 = [-90.0, 0.0, 0.0, 0.0, 90.0, 180, 3800]
        config2 = [90.0, 0.0, 0.0, 0.0, 90.0, 180, 100]
        config3 = [180.0, 0.0, 0.0, 0.0, 90.0, 180.0]

    for i in range(6):
        config_scaled["r1_joint_{}".format(i + 1)] = math.radians(config1[i])
        config_scaled["r2_joint_{}".format(i + 1)] = math.radians(config2[i])
        config_scaled["r3_joint_{}".format(i + 1)] = math.radians(config3[i])

    # mm --> m
    config_scaled["r1_cart_joint"] = config1[6] / 1000
    config_scaled["r2_cart_joint"] = config2[6] / 1000

    return config_scaled


def ik_calc(robot, frame, start_config, planning_group):
    return robot.inverse_kinematics(frame, start_config, planning_group, options={"timeout": 0.01})


def main(robot, rob_num, planning_group, path, skip_rng):
    p = Point(0.0, 0.0, 0.0)
    start_config = robot_config(robot, rob_num)

    for n, vec in enumerate(axis_gen()):

        if n in skip_rng:
            continue

        ranges, total_points = points_ranges(rob_num, n)

        idx_no_soln = []
        idx_soln = []
        result_dict = {}

        bar = tqdm(
            points_in_box(p, vec, ranges),
            bar_format="{desc}{postfix} | {n_fmt}/{total_fmt} | {percentage:3.0f}%|{bar}| {elapsed}/{remaining}",
            total=total_points,
            desc="Progress Bar",
        )

        for frames in bar:
            for frame in frames:

                bar.set_postfix(
                    {
                        "Vector": "{}/{}: {}".format(n + 1, 100, vec),
                        "Point": frame.point,
                        "success": len(idx_soln),
                        "failure": len(idx_no_soln),
                    }
                )

                try:
                    _ = ik_calc(robot, frame, start_config, planning_group)
                    success = 1
                except RosValidationError:
                    success = 0

                if success:
                    idx_soln.append(frame.point)
                    aa = str([round(x + 0, 2) for x in frame.point]).strip("[]")  # +0 to avoid -0.0
                    result_dict[aa] = True
                else:
                    idx_no_soln.append(frame.point)
                    aa = str([round(x + 0, 2) for x in frame.point]).strip("[]")  # +0 to avoid -0.0
                    result_dict[aa] = False

        save_JsonFile(path, ranges["name"], vec, result_dict)
        # plot_dots(idx_soln, idx_no_soln)


if __name__ == "__main__":

    rob_num = "rob2"
    planning_group = "robot2_track_gripper"

    path = get_directory("_data", rob_num)

    calc = True
    merge = False

    if calc:
        skip_rng = range(110, 120)
        robot = connect_and_scene()
        main(robot, rob_num, planning_group, path, skip_rng)

    if merge:
        merge_index = range(1, 101)
        merge_num = 4

        for i in merge_index:
            in_fps = generate_filepaths(path, rob_num, i, merge_num)
            out_fp = os.path.join(path, "{}_vec{:0>2}_combined.json".format(rob_num, i))

            merge_JsonFiles(in_fps, out_fp)
