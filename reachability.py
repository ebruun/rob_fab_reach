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


#################
# I/O functions
#################


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


def append_JsonFiles(in_filepaths, out_filepath):
    """concatenate several JSON files"""
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


def combine_JsonFiles(in_filepaths, out_filepath):
    """combine data from several JSON files
    add up how many total TRUE values each point has
    """
    data_combined = {}

    # initialize the counting dictionary
    with open(in_filepaths[0], "r") as f:
        print("loading {}".format(in_filepaths[0]))
        data_combined = json.load(f)
        f.close()

        data_combined["vector"] = 1

        # replace True/False with 1/0
        for key, value in data_combined["results"].items():
            if value:
                data_combined["results"][key] = 1
            else:
                data_combined["results"][key] = 0

    # read in rest of data files
    for in_filepath in in_filepaths[1:]:
        with open(in_filepath, "r") as f:
            print("loading {}".format(in_filepath))
            data = json.load(f)
            f.close()

            data_combined["vector"] += 1

            # increment anywhere there is True with 1
            for key, value in data["results"].items():
                if value:
                    data_combined["results"][key] += 1

    with open(out_filepath, "w") as f:
        json.dump(data_combined, f, indent=4)
        f.close()


def generate_filepaths_append(path, filenames, rob_num, idx):
    filepaths = []

    for filename in filenames:
        filename = filename.format(rob_num, idx)
        filepaths.append(os.path.join(path, filename))

    return filepaths


def generate_filepaths_combined(path, filename, rob_num, merge_range):
    filepaths = []
    for i in merge_range:
        filename_i = filename.format(rob_num, i)
        filepaths.append(os.path.join(path, filename_i))

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


#################
# Path Planning Functions
#################


def connect_and_scene():
    """ROS connection, adding collision meshes"""
    ros_client = RosClient("localhost")
    ros_client.run()

    robot = ros_client.load_robot()

    scene = PlanningScene(robot)
    mesh = Mesh.from_stl("./3dm/ground.stl")
    cm = CollisionMesh(mesh, "ground")
    scene.add_collision_mesh(cm)

    return robot


def robot_config(robot, rob_num):
    """Starting configuration for each robot to be used in each IK run"""
    config_scaled = robot.zero_configuration()

    if rob_num == "rob1":
        config1 = [-90.0, 0.0, 0.0, 0.0, 90.0, 180, 1950]
        config2 = [90.0, 0.0, 0.0, 0.0, 90.0, 180, 100]
        config3 = [180.0, 0.0, 0.0, 0.0, 90.0, 180.0]
    elif rob_num == "rob2":
        config1 = [90.0, 0.0, 0.0, 0.0, 90.0, 180, 3800]
        config2 = [90.0, 0.0, 0.0, 0.0, 90.0, 180, 1950]
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
    """single IK calcultion, given failure timeout"""

    set_timeout = 0.05  # important variable, controls overall runtime

    # catch the error if timeout reached = no IK solution
    return robot.inverse_kinematics(
        frame, start_config, planning_group, options={"timeout": set_timeout}
    )


#################
# Geometry Functions
#################


def points_ranges(rob_num, n):
    """define the grid of points to search for each robot in each run"""
    if rob_num == "rob1":
        # ranges = {
        #     "i": np.arange(-1.5, 6.01, 0.1),
        #     "j": np.arange(0.3, 4.91, 0.1),
        #     "k": np.arange(-0.4, 3.41, 0.1),
        #     "name": "{}_vec{:0>3}".format(rob_num, n),
        # }
        ranges = {
            "i": np.arange(-3.2, -1.59, 0.1),
            "j": np.arange(0.3, 4.91, 0.1),
            "k": np.arange(-0.4, 3.41, 0.1),
            "name": "{}_vec{:0>3}_2".format(rob_num, n),
        }

    elif rob_num == "rob2":
        ranges = {
            "i": np.arange(-1.5, 6.01, 0.1),
            "j": np.arange(-1.5, 3.11, 0.1),
            "k": np.arange(-0.4, 3.41, 0.1),
            "name": "{}_vec{:0>3}".format(rob_num, n),
        }
        # ranges = {
        #     "i": np.arange(-3.2, -1.59, 0.1),
        #     "j": np.arange(-1.5, 3.11, 0.1),
        #     "k": np.arange(-0.4, 3.41, 0.1),
        #     "name": "{}_vec{:0>3}_2".format(rob_num, n),
        # }
    elif rob_num == "rob3":
        ranges = {
            "i": np.arange(1.7, 8.01, 0.1),
            "j": np.arange(-1.5, 4.91, 0.1),
            "k": np.arange(-0.4, 3.41, 0.1),
            "name": "{}_vec{:0>3}".format(rob_num, n),
        }

    num_points = len(ranges["i"]) * len(ranges["j"]) * len(ranges["k"])

    return ranges, num_points


def frame_gen(axis, ranges):
    """generator function for a frame at each point, given the axis for this run
    up to accuracy of 0.1m (based on round)
    """

    p = Point(0.0, 0.0, 0.0)  # if want to offset (for some reason?)

    for i in ranges["i"]:
        for j in ranges["j"]:
            for k in ranges["k"]:
                x = round(i, 1) + p.x
                y = round(j, 1) + p.y
                z = round(k, 1) + p.z

                plane = Plane((x, y, z), axis)
                f = Frame.from_plane(plane)

                yield [Frame(f.point, f.xaxis, f.yaxis)]


def axis_gen(samples=100):
    """generator function for an evenly spaced set of vectors in on a sphere
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    """
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


def main_calc(robot, rob_num, planning_group, path, analysis_rng):
    start_config = robot_config(robot, rob_num)

    # loop based on vector, go through each point with the same vector
    for n, vec in enumerate(axis_gen()):
        if (n + 1) in analysis_rng:
            # initialize variables to save data to for this run
            idx_no_soln = []
            idx_soln = []
            result_dict = {}

            ranges, total_points = points_ranges(rob_num, n + 1)

            bar = tqdm(
                frame_gen(vec, ranges),
                bar_format=(
                    "{desc}{postfix} | {n_fmt}/{total_fmt} | {percentage:3.0f}%|{bar}|"
                    " {elapsed}/{remaining}"
                ),
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
                        aa = str([round(x + 0, 2) for x in frame.point]).strip(
                            "[]"
                        )  # +0 to avoid -0.0
                        result_dict[aa] = True
                    else:
                        idx_no_soln.append(frame.point)
                        aa = str([round(x + 0, 2) for x in frame.point]).strip(
                            "[]"
                        )  # +0 to avoid -0.0
                        result_dict[aa] = False

            # After all points checked for particular vector
            save_JsonFile(path, ranges["name"], vec, result_dict)
            # plot_dots(idx_soln, idx_no_soln)


if __name__ == "__main__":
    calc = False
    append = False
    combine = True

    rob_nums = [
        "rob1",
        # "rob2",
        # "rob3"
    ]

    planning_groups = [
        "robot1_track_gripper",
        # "robot2_track_gripper",
        # "robot3_gripper"
    ]

    operations_range = range(1, 43)
    data_folder = "_data_track"

    for rob_num, planning_group in zip(rob_nums, planning_groups):
        path = get_directory(data_folder, rob_num)

        # perform calculation
        if calc:
            analysis_index = operations_range  # which steps to perform calculation
            robot = connect_and_scene()
            main_calc(robot, rob_num, planning_group, path, analysis_index)

        # append/merge separate JSON files
        if append:
            append_index = operations_range  # for which steps to do a merge
            append_files = ["{}_vec{:0>3}.json", "{}_vec{:0>3}_2.json"]

            for i in append_index:
                in_fps = generate_filepaths_append(path, append_files, rob_num, i)
                out_fp = os.path.join(path, "{}_vec{:0>3}__combined.json".format(rob_num, i))

                append_JsonFiles(in_fps, out_fp)

        # Combine a range of files into a single file, counting TRUE values
        if combine:
            combine_index = operations_range  # for which steps combine all the data

            filename = name = "{}_vec{:0>3}__combined.json"

            in_fps = generate_filepaths_combined(path, filename, rob_num, combine_index)
            out_fp = os.path.join(path, "_{}_TOTAL.json".format(rob_num))

            combine_JsonFiles(in_fps, out_fp)
