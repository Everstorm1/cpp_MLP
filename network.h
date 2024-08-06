#ifndef NETWORK_H
#define NETWORK_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <cmath>

double randomDouble();
double sigmoid(double neuronValue);
double relu(double neuronValue);

struct Layer {
    int numInputs;
    int numNeurons;
    std::string activationFunction;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    Layer(int num_Inputs, int num_Neurons, std::string activation_func);
};

double handleActivationFunction(double& inputVal, std::string& activationFunction);

struct Network {
    std::vector<Layer> layers;

    Network(std::vector<Layer> layers_);

    std::vector<double> forwardPass(std::vector<double> input);

private:
    std::vector<double> layerForward(std::vector<double>& input, Layer& layer);
};

#endif // NETWORK_H