# rob_fab_reach

## Making clean environment.yml file

export only explictly downloaded packages:

`conda env export --from-history --name rob_fab_reach > environment.yml`

create new from environment.yml file:
`conda env create`

## Linting and Pre-commit

make sure .pre-commit-config.yaml file is up to date

language_version: python (use system version of python)

run this: `pre-commit install`

in .vscode/settings.json make sure formatonsave = TRUE

## Note

In compas_fab 0.27.0 the ROS IK timeout definition is updated to allow floats

saved data folders:
* default output data folder is called `_data` where saved will be over-written by reachability code
* new saved folders for notrack and track cases
* note that saved rob3 results are same in both folders (i.e., rob3 is never mobile)

reachability.py:
* generates the discretized reachability map for robots and vectors. combines them into a single total file. vizualize in grasshopper
* Need to compose-up on the `ECL_3rob_cell` docker file in `ECL_Robotic_cell` repo for this to run

json_to_latex.py:
* takes the output from grasshopper, which checks the planar arches against the reachability map, and is exported as a .json file
* turns all the output data into a latex table that can be copied into the table
* individual results are saved in `results_planar_arches` folder. output for the paper is in the format '_a##' based on the inclination angle
