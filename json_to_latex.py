import os
import json

from collections import OrderedDict


def generate_filename(a, s, r):
    name = "a{:0>2}_s{:0>2}_r{:0>3}"

    angle = str(a).split(".")[0]
    span = str(s).replace(".", "")
    ratio = str(int(round(r * 100, 0))).replace(".", "")

    filename = name.format(angle, span, ratio)
    return filename


def generate_filepaths(path, angle, spans, ratios):
    filepaths = []
    filenames = [generate_filename(angle, s, r) for r in ratios for s in spans]

    for f in filenames:
        filepath = os.path.join(path, f)
        filepaths.append(filepath)

    return filepaths


def save_json(path, dims, data):
    filename = generate_filename(dims)
    filepath = os.path.join(path, filename)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
        f.close()


def read_json_data_info(files_in, txt):
    lines_data = []
    i = 0

    for file_in in files_in:
        with open(file_in, "r") as f:
            data_in = json.load(f)

            print("opening: {}".format(data_in["arch_info"]))

            ratio = data_in["arch_info"]["ratio"]

            bricks_total = data_in["reach_data_coop"]["bricks_total"]
            # mass_total = data_in["analysis_data"]["100%"]["mass_kg"]

            # correct self-weight, for some reason karamba output total weight wrong
            mass_total = bricks_total * 2.70928  # kg/brick

            if i % 6 == 0:
                lines_data.append(txt[0])
                lines_data.append(txt[1].format(ratio))

            lines_data.append(
                txt[6].format(
                    bricks_total,
                    mass_total,
                )
            )

            i += 1
            f.close()

    return lines_data


def read_json_data_reach(files_in, txt):
    lines_data = []
    i = 0

    for file_in in files_in:
        with open(file_in, "r") as f:
            data_in = json.load(f)

            print("opening: {}".format(data_in["arch_info"]))

            ratio = data_in["arch_info"]["ratio"]

            reach_ratio_coop = data_in["reach_data_coop"]["reach_ratio"]
            reach_ratio_single = data_in["reach_data_single"]["reach_ratio"]
            # avg_reach_score_coop = data_in["reach_data_coop"]["avg_reach_score"]
            avg_reach_score_single = data_in["reach_data_single"]["avg_reach_score"]

            if i % 6 == 0:
                lines_data.append(txt[0])
                lines_data.append(txt[1].format(ratio))

            if reach_ratio_single < 0.99 and reach_ratio_coop < 0.99:
                lines_data.append(
                    txt[3].format(
                        "lightred",
                        reach_ratio_single * 100,
                        reach_ratio_coop * 100,
                        avg_reach_score_single,
                    )
                )
            elif reach_ratio_single == 1.0 and reach_ratio_coop < 0.99:
                lines_data.append(
                    txt[3].format(
                        "lightyellow",
                        reach_ratio_single * 100,
                        reach_ratio_coop * 100,
                        avg_reach_score_single,
                    )
                )
            else:
                lines_data.append(
                    txt[2].format(
                        reach_ratio_single * 100,
                        reach_ratio_coop * 100,
                        avg_reach_score_single,
                    )
                )

            i += 1
            f.close()

    return lines_data


def read_json_data_force(files_in, txt, force_limit):
    lines_data = []
    i = 0

    for file_in in files_in:
        with open(file_in, "r") as f:
            data_in = json.load(f)

            print("opening: {}".format(data_in["arch_info"]))

            ratio = data_in["arch_info"]["ratio"]

            force_100 = data_in["analysis_data"]["100%"]["rob_support_kg"]
            force_75 = data_in["analysis_data"]["75%"]["rob_support_kg"]
            force_50 = data_in["analysis_data"]["50%"]["rob_support_kg"]

            if i % 6 == 0:
                lines_data.append(txt[0])
                lines_data.append(txt[1].format(ratio))

            if force_50 > force_limit:
                lines_data.append(
                    txt[5].format(
                        "lightred",
                        force_100,
                        force_75,
                        force_50,
                    )
                )
            elif force_75 > force_limit:
                lines_data.append(
                    txt[5].format(
                        "lightred",
                        force_100,
                        force_75,
                        force_50,
                    )
                )
            elif force_100 > force_limit:
                lines_data.append(
                    txt[5].format(
                        "lightyellow",
                        force_100,
                        force_75,
                        force_50,
                    )
                )
            else:
                lines_data.append(
                    txt[4].format(
                        force_100,
                        force_75,
                        force_50,
                    )
                )

            i += 1
            f.close()

    return lines_data


def write_text_data(path, file_out, lines_data):
    filepath_out = os.path.join(path, file_out)

    with open(filepath_out, "w") as f_out:
        for line in lines_data:
            f_out.write(line)
            f_out.write("\n")

        f_out.write("\\\\")
        f_out.close()


##################
# SET VARIABLES
##################

paths = [
    "C:/Users/Edvard/Documents/GitHub/rob_fab_reach/_data_track/results_planar_arches",
    "C:/Users/Edvard/Documents/GitHub/rob_fab_reach/_data_notrack/results_planar_arches",
]

force_limit = [70, 235]  # IRB5710-70/2.70  # IRB6700-235/2.65

angles = [0, 15, 30, 45]
spans = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
ratios = [0.25, 0.50, 0.75, 1.0, 1.5, 2.0]

txt = [
    "\\\\[-0.2em]\cmidrule{2-8}\\\\[-1.3em]",
    "&\multicolumn{{1}}{{c|}}{{\makecell[lc]{{\\\\{:.2f}}}}}",
    "&\\tiny{{\makecell[lc]{{S:{:.0f}\\%\\\\C:{:.0f}\\%\\\\fab:{}}}}}",
    "&\\tiny\cellcolor{{{}}}{{\makecell[lc]{{S:{:.0f}\\%\\\\C:{:.0f}\\%\\\\fab:{}}}}}",
    "&\\tiny{{\makecell[lc]{{max:{:.0f}\\\\75\\%:{:.0f}\\\\50\\%:{:.0f}}}}}",
    "&\\tiny\cellcolor{{{}}}{{\makecell[lc]{{max:{:.0f}\\\\75\\%:{:.0f}\\\\50\\%:{:.0f}}}}}",
    "&\\footnotesize{{\makecell[lc]{{n={}\\\\{:.0f}kg}}}}",
]

# generate these files
file_out = [
    "_a{:0>2}_reachability_latex.txt",
    "_a{:0>2}_force_latex.txt",
    "_a{:0>2}_info_latex.txt",
]


##################
# RUN CODE
##################

path_idx = 1  # 0 = track, 1 = no_track
angle_idxs = [0, 1, 2, 3]

files_in = generate_filepaths(paths[path_idx], angles[0], spans, ratios)


# GENERATE INFO FILE (same for all angles)
lines_data = read_json_data_info(files_in, txt)
lines_data.pop(0)  # remove 1st line
file_out_save = file_out[2].format(angles[0])
write_text_data(paths[path_idx], file_out_save, lines_data)

# GENERATE REACH & FORCE FILES (same for all angles)
for angle_idx in angle_idxs:
    files_in = generate_filepaths(paths[path_idx], angles[angle_idx], spans, ratios)

    lines_data = read_json_data_reach(files_in, txt)
    lines_data.pop(0)  # remove 1st line
    file_out_save = file_out[0].format(angles[angle_idx])
    write_text_data(paths[path_idx], file_out_save, lines_data)

    lines_data = read_json_data_force(files_in, txt, force_limit[path_idx])
    lines_data.pop(0)  # remove 1st line
    file_out_save = file_out[1].format(angles[angle_idx])
    write_text_data(paths[path_idx], file_out_save, lines_data)
