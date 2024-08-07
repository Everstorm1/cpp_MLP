#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <math.h>
#include "network.h"

double randomDouble(){
    return static_cast<double>(rand()) / RAND_MAX;
}

double sigmoid(double neuronValue){
    return (1 / (1 + exp(-neuronValue)));
}

double relu(double neuronValue){
    if(neuronValue < 0){
        return 0;
    }
    return neuronValue;
}

Layer::Layer(int num_Inputs, int num_Neurons, std::string activation_func) : numInputs(num_Inputs), numNeurons(num_Neurons), activationFunction(activation_func){
    weights.resize(num_Neurons, std::vector<double>(num_Inputs));
        biases.resize(num_Neurons);

        //set activation function if not defined correctly
        if(activation_func != "relu" and activation_func != "sigmoid" and activation_func != ""){
            activation_func = "relu";
        }

        //random initialization
        for(int i = 0; i < num_Neurons; i++){
            biases[i] = randomDouble();
            //biases[i] = 0;
            for(int w = 0; w < num_Inputs; w++){
                weights[i][w] = randomDouble();
                //weights[i][w] = 0;
            }
        }
}

double handleActivationFunction(double& inputVal, std::string& activationFunction){
    if(activationFunction == "relu"){
        return relu(inputVal);
    }
    else if(activationFunction == "sigmoid"){
        return sigmoid(inputVal);
    }

    //if not one of the options --> error
    std::cout << "error: activation function not set correctly" << std::endl;
    return 0;
}

Network::Network(std::vector<Layer> layers_) : layers(layers_){}

std::vector<double> Network::forwardPass(std::vector<double> input){
    std::vector<double> output = input;
    for(Layer& layer : layers){
        output = layerForward(output, layer);
    }

    return output;
}


std::vector<double> Network::layerForward(std::vector<double>& input, Layer& layer){
    std::vector<double> output = std::vector<double>(layer.numNeurons);
    //iterate through each neuron
    for(int n = 0; n < layer.numNeurons; n++){
        output[n] = layer.biases[n];
        //iterate through input
        for(int i = 0; i < layer.numInputs; i++){
            output[n] += layer.weights[n][i] * input[i];
        }

        output[n] = handleActivationFunction(output[n], layer.activationFunction);
    }

    return output;
}