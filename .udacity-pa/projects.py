import argparse
import shutil
import os
from udacity_pa import udacity

nanodegree = 'nd889'
projects = ['cargo_planning']

def submit(args):
  filenames = ['my_air_cargo_problems.py', 'my_planning_graph.py', 'README.md']

  udacity.submit(nanodegree, projects[0], filenames, 
                 environment = args.environment,
                 jwt_path = args.jwt_path)
