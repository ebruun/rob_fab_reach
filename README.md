# rob_fab_reach

robotic reachability work

Need to compose-up on the `ECL_3rob_cell` docker file in `ECL_Robotic_cell` repo for this to run

This has no version number:

`conda env export --from-history --name compas_fab> environment.yml`

create new:

`conda env create`

if you change the conda environment, make sure to update the file .git/hooks/pre-commit file

`pre-commit install`

build from requirements in environment.yml.

But note that compas_fab 0.26.0 is updated from local fork where the ROS IK timeout definition is updated to allow floats

from local compas_fab branch directory run:
`pip install -e .`

