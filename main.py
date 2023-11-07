import argparse
import h5py
import pickle
import numpy as np
from os.path import basename

def translate(layer):
    name, weights = layer
    weights = np.array(weights).T
    max_pos_input = 0

    for neuron in weights:
        input_sum = 0
        input_sum = np.sum(neuron[neuron > 0])
        max_pos_input = max(max_pos_input, input_sum)
    
    new_layer = (np.array(weights) / max_pos_input).T

    return name, new_layer
    

def extract_weights(file):
    layers = {}
    for name in file["model_weights"]:  
        try:
            layer = file["model_weights"][name][name].get("kernel:0")[:]
            print(len(layer),"x",len(layer[0]))
            layers[name] = layer
        except:
            print("Layer is not Dense")

    return layers


def read_file(file_path):
    try:
        with h5py.File(file_path, 'r') as file:
            return extract_weights(file)
          
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="Path to the .h5 model file")

    args = parser.parse_args()

    layers = read_file(args.file_path)

    translated_layers = {name: weights for name, weights in map(translate, layers.items())}

    new_name = basename(args.file_path)[:-3]+".pkl"

    with open(new_name, "wb") as pickle_file:
        pickle.dump(translated_layers, pickle_file)


if __name__ == "__main__":
    main()
