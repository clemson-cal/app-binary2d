#!/usr/bin/env python3

import numpy as np
import re
import sys
from os import path


# This function converts input string to numbers.
def iteration_conv(input_string):
    result = re.findall(b'[0-9]+', input_string)
    return float(result[0])


# This function takes in a list of numbers and produces the average of the
# sum of all numbers.
def calc_avg(input_list):
    accum = 0.0
    num_itr = 0.0
    for element in input_list:
        accum += element
        num_itr += 1.0

    result_float = accum / num_itr
    return result_float


# This function parses formatted names of the output file and extract useful
# info from it, and then put them in a dictionary. This dictionary will
# be grouped into a dict of dicts, where the keys are the folder name.
def parse_param(file_name_string):
    result_dict = {}
    file_ext = '.txt'
    new_name_string = ''
    if file_name_string.endswith(file_ext):
        new_name_string = file_name_string[:-(len(file_ext))]
    parm_list = new_name_string.split('_')
    parm_list.pop(0)

    for element in parm_list:
        element_type = re.findall(r'[a-zA-Z]+', element)[0]
        element_value = float(re.findall(r'\d+', element)[0])
        result_dict[element_type] = element_value

    return result_dict


# If not enough arguments supplied, raise exception.
if len(sys.argv) < 3:
    raise SyntaxError('Number of arguments insufficient.')


# Here, variables for the core operation are defined.
output_filename = sys.argv[1]
if path.exists(output_filename):
    file_pointer = open(output_filename, 'a')
else:
    file_pointer = open(output_filename, 'w')


list_files = sys.argv[2:]
for names in list_files:
    individual_dict = parse_param(names)
    read_kzps = np.loadtxt(names, comments='w', skiprows=37, usecols=2,
                           unpack=True, delimiter=' ',
                           converters={2: iteration_conv})
    kzps_avg = calc_avg(read_kzps)
    # print(names, end=" ", file=file_pointer)
    print(individual_dict['nb'], end=" ", file=file_pointer)
    print(individual_dict['bs'], end=" ", file=file_pointer)
    print(kzps_avg, file=file_pointer)

file_pointer.close()
