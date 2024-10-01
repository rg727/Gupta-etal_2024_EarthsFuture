import cython
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import shutil
import sys 
from configobj import ConfigObj
import json 
from distutils.util import strtobool
import h5py
from calfews_src.model_cy import Model
from calfews_src.inputter_cy import Inputter
from calfews_src.scenario import Scenario
from calfews_src.util import *
from calfews_src.plotter import *
from calfews_src.visualizer import Visualizer
from datetime import datetime


#Input arguement 
ensemble_number = sys.argv[1]


output_folder = "/home/fs02/pmr82_0001/rg727/CALFEWS-main/results/baseline_ensemble/"+ensemble_number+"/6/"


if not os.path.exists(output_folder):
  os.makedirs(output_folder)


base_inflow_filename = " /home/fs02/pmr82_0001/rg727/CALFEWS/"+ensemble_number+"/6/base_inflows.json"

command = "python run_main_cy.py " + output_folder + " 1 1" + base_inflow_filename 
os.system(command)


# results hdf5 file location from CALFEWS simulations
output_file = output_folder + 'results.hdf5'

