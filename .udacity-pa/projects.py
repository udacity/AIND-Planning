import argparse
import shutil
import os
from udacity_pa import udacity

nanodegree = 'nd889'
projects = ['cargo_planning']
filenames_all = ['my_air_cargo_problems.py', 'my_planning_graph.py', 'heuristic_analysis.pdf', 'research_review.pdf']

def submit(args):
  filenames = []
  for filename in filenames_all:
      if os.path.isfile(filename):
          filenames.append(filename)

  udacity.submit(nanodegree, projects[0], filenames, 
                 environment = args.environment,
                 jwt_path = args.jwt_path)
