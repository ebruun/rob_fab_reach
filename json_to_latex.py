import os
import json

from collections import OrderedDict

def generate_filename(a,s,r):
    name = "a{:0>2}_s{:0>2}_r{:0>3}"
    
    angle = str(a).split('.')[0]
    span = str(s).replace('.','')
    ratio = str(int(round(r*100,0))).replace('.','')
    
    filename = name.format(angle,span,ratio)
    return filename

def generate_filepaths(path,angle,spans,ratios):

    filepaths = []
    filenames = [generate_filename(angle,s,r) for r in ratios for s in spans]

    for f in filenames:
        filepath = os.path.join(path, f)
        filepaths.append(filepath)
    
    return filepaths

def save_json(path,dims,data):
    filename = generate_filename(dims)
    filepath = os.path.join(path, filename)
    
    with open(filepath, "w") as f:
        json.dump(data,f,indent=4)
        f.close()

def read_json_data(files_in,txt):
    lines_data = []
    i = 0

    for file_in in files_in:

        with open(file_in, 'r') as f:
            data_in = json.load(f)

            print("opening: {}".format(data_in["arch_info"]))

            ratio = data_in["arch_info"]["ratio"]

            bricks_total = data_in["reach_data_coop"]["bricks_total"]
            reach_ratio_coop = data_in["reach_data_coop"]["reach_ratio"]
            reach_ratio_single = data_in["reach_data_single"]["reach_ratio"]
            avg_reach_score_coop = data_in["reach_data_coop"]["avg_reach_score"]
            avg_reach_score_single = data_in["reach_data_single"]["avg_reach_score"]

            if i%6 == 0:
                lines_data.append(txt[0])
                lines_data.append(txt[1].format(ratio))

            if reach_ratio_single < 1.0 and reach_ratio_coop < 1.0:
                lines_data.append(txt[3].format(
                    "lightred",
                    bricks_total,
                    reach_ratio_single*100,
                    "-",
                    reach_ratio_coop*100,
                    "-"
                    ))
            elif reach_ratio_single == 1.0 and reach_ratio_coop < 1.0:
                lines_data.append(txt[3].format(
                    "lightyellow",
                    bricks_total,
                    reach_ratio_single*100,
                    avg_reach_score_single,
                    reach_ratio_coop*100,
                    "-"
                    ))                
            else:
                lines_data.append(txt[2].format(
                    bricks_total,
                    reach_ratio_single*100,
                    avg_reach_score_single,
                    reach_ratio_coop*100,
                    avg_reach_score_coop
                    ))

            i += 1
            f.close()

    return lines_data

def write_text_data(path,file_out,lines_data):

    filepath_out = os.path.join(path, file_out)

    with open(filepath_out, 'w') as f_out:
        
        for line in lines_data:
            f_out.write(line)
            f_out.write('\n')
        
        f_out.write('\\\\')
        f_out.close()




##################
# SET VARIABLES
##################

paths = [
    'C:/Users/Edvard/Documents/GitHub/rob_fab_reach/_data_track/results_planar_arches',
    'C:/Users/Edvard/Documents/GitHub/rob_fab_reach/_data_notrack/results_planar_arches',
]

angles = [0,15,30,45]
spans = [2.0,3.0,4.0,5.0,6.0,7.0]
ratios = [0.25,0.50,0.75,1.0,1.5,2.0]

txt = [
    "\\\\\cmidrule{2-8}\\\\[-1.3em]",
    "&\multicolumn{{1}}{{c|}}{{\makecell[cc]{{\\\\{:.2f}}}}}",
    "&\\tiny{{\makecell[cc]{{n={}\\\\{:.0f}\\%:{}\\\\{:.0f}\\%:{}}}}}",
    "&\\tiny\cellcolor{{{}}}{{\makecell[cc]{{n={}\\\\{:.0f}\\%:{}\\\\{:.0f}\\%:{}}}}}",
]


files_out = [
    "_a{:0>2}_reachability_latex.txt",
]




##################
# RUN CODE
##################

angle_idx = 3
path_idx = 1

files_in = generate_filepaths(paths[path_idx],angles[angle_idx],spans,ratios)

lines_data = read_json_data(files_in,txt)

lines_data.pop(0) #remove 1st line
print(lines_data)

files_out = [file_out.format(angles[angle_idx]) for file_out in files_out]

write_text_data(paths[path_idx],files_out[0], lines_data)

