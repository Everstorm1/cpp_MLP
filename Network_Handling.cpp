#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <math.h>

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

struct Layer{
    int numInputs;
    int numNeurons;

    std::string activationFunction;

    std::vector<std::vector<double>> weights;   //dim-0: which neuron; dim-1: weight per previous neuron in input
    std::vector<double> biases;                 //dim-0: bias per neuron

    //constructor
    Layer(int num_Inputs, int num_Neurons, std::string activation_func) : numInputs(num_Inputs), numNeurons(num_Neurons), activationFunction(activation_func){
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
};

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

struct Network{
    std::vector<Layer> layers;

    //constructor
    Network(std::vector<Layer> layers_) : layers(layers_){}

    //forward function
    std::vector<double> forwardPass(std::vector<double> input){
        std::vector<double> output = input;
        for(Layer& layer : layers){
            output = layerForward(output, layer);
        }

        return output;
    }
    
private:
    std::vector<double> layerForward(std::vector<double>& input, Layer& layer){
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
};

int main(){

    std::vector<double> input = {0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0};
    Layer hiddenLayer1(12, 256, "relu");
    Layer hiddenLayer2(256, 1024, "relu");
    Layer hiddenLayer3(1024, 2056, "relu");
    Layer hiddenLayer4(2056, 4112, "relu");
    Layer hiddenLayer5(4112, 1024, "relu");
    Layer outputLayer(1024, 2, "sigmoid");

    Network network({hiddenLayer1,
                     hiddenLayer2,
                     hiddenLayer3,
                     hiddenLayer4,
                     hiddenLayer5,
                     outputLayer  });

    std::vector<double> output = network.forwardPass(input);


    std::cout << "Output layer size: " << output.size() << "\n";
    for (double value : output) {
        std::cout << value << ", ";
    }
    std::cout << std::endl;
    
    return 0;
}