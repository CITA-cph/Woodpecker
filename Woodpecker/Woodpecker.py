from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import load_files as lf
import train_network as tn 
import continue_training as ct 
import training_settings as setts
import extract_model as extract


do_run = True 

while do_run: 
    make_new = input("(t)rain new model, (c)ontinue training, (e)xtract model , e(x)it (t/c/e/x): ") 

    if make_new == "t": 
        make_new = ""
        tens_in, tens_out = lf.load_files(setts.inputs_file, setts.outputs_file, setts.train_size, setts.test_size, setts.valid_size)
        tn.train(tens_in, tens_out)
    elif make_new == "c":
        make_new = ""
        tens_in, tens_out = lf.load_files(setts.inputs_file, setts.outputs_file, setts.train_size, setts.test_size, setts.valid_size)
        ct.train(tens_in, tens_out)
    elif make_new == "e": 
        make_new = ""
        extract.extract_network()
    elif make_new == "x": 
        make_new = ""
        exit()