# rob_fab_reach

Need to compose-up on the `ECL_3rob_cell` docker file in `ECL_Robotic_cell` repo for this to run

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


