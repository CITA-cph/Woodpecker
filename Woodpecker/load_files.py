from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import owl_py as owl
import owl_py.owl_data_types as types
 
def load_files(inputs, outputs, train_size, test_size, validate_size): 

    # file paths
    inputs_file = inputs
    outputs_file = outputs
     
    # import files, have to have the same data type (float32/single int8/byte etc.)
    rnd_in = owl.tbin_numpy.load_tbin(inputs_file)
    rnd_out = owl.tbin_numpy.load_tbin(outputs_file)
    print("Loaded the training inputs from " + inputs_file)
    print("Loaded the training outputs from " + outputs_file)

    # construct TensorSets for easier training (it's a very crude class right now)
    tens_in = types.TensorSet(rnd_in, train_size, test_size, validate_size)  # note those tensorsets are not the same as the ones in the .net libs
    tens_out = types.TensorSet(rnd_out, train_size, test_size, validate_size)  # note those tensorsets are not the same as the ones in the .net libs

    return tens_in, tens_out
