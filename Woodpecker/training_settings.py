inputs_file = "d:/OneDrive/Documents/Visual Studio 2017/Projects/Woodpecker/Woodpecker/Data/inputs.tbin"
outputs_file = "d:/OneDrive/Documents/Visual Studio 2017/Projects/Woodpecker/Woodpecker/Data/outputs.tbin"

save_path_weights = "d:/onedrive/documents/visual studio 2017/Projects/Woodpecker/Woodpecker/Model/extraction_weights.tbin"
save_path_biases = "d:/onedrive/documents/visual studio 2017/Projects/Woodpecker/Woodpecker/Model/extraction_biases.tbin"

model_path = "d:/onedrive/documents/visual studio 2017/Projects/Woodpecker/Woodpecker/Model/"

train_size = 6000
test_size = 0
valid_size = 0

batch_size = 6000

epochs = 1000000
save_every = 100  # the model will be saved every save_every epochs
print_every = 1  # print the current loss every print_every epochs

## layer counts
#model_n_0 = 226 * 2
#model_n_1 = 300
#model_n_2 = 150
#model_n_3 = 100
#model_n_4 = 226

## layer counts / autoencoder
#model_n_0 = 226  
#model_n_1 = 100
#model_n_2 = 2
#model_n_3 = 100
#model_n_4 = 226

# layer counts / classification
model_n_0 = 1024
model_n_1 = 800
model_n_2 = 600
model_n_3 = 200
model_n_4 = 1
 