#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <math.h>

double randomDouble(){
    return static_cast<double>(rand()) / RAND_MAX;
}

double oneOrZero(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);
    return dis(gen);
}

double sigmoid(double& neuronValue){
    return (1 / (1 + exp(-neuronValue)));
}

double relu(double& neuronValue){
    if(neuronValue < 0){
        return 0;
    }
    return neuronValue;
}

struct HiddenLayer{
    int inputNeurons;
    int layerNeurons;

    std::string activationFunction;

    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    //constructor
    HiddenLayer(int num_inputs, int num_neurons, std::string activation_Function) : inputNeurons(num_inputs), layerNeurons(num_neurons), activationFunction(activation_Function){
        //resize
        weights.resize(num_neurons, std::vector<double>(num_inputs));
        biases.resize(num_neurons);

        //set activation function if not defined
        if(activation_Function != "relu" or activation_Function != "sigmoid" or activation_Function != ""){
            activation_Function = "relu";
        }

        //random initialization
        for(int i = 0; i < num_neurons; i++){
            biases[i] = randomDouble();
            //biases[i] = oneOrZero();
            //biases[i] = 0;
            for(int w = 0; w < num_inputs; w++){
                weights[i][w] = randomDouble();
                //weights[i][w] = oneOrZero();
                //weights[i][w] = 0;
            }
        }
    }
};

struct InputLayer{
    std::vector<double> inputValues;
    int num_inputs;

    //constructor
    InputLayer(int num_inputs, std::vector<double> inputValues) : num_inputs(num_inputs), inputValues(inputValues){
        //resize
        inputValues.resize(num_inputs);
    }
};

struct Container{
    int containerSize;
    std::vector<double> values;

    //constructor
    Container(int containerSize) : containerSize(containerSize){
        //resize
        values.resize(containerSize);
    }
};

void SingleForward(std::vector<double> input_layer, HiddenLayer& hidden_layer, Container& results){
    Container result = Container(hidden_layer.layerNeurons);
    for(int i = 0; i < hidden_layer.layerNeurons; i++){
        double singleNeuron = 0;
        for(int w = 0; w < input_layer.size(); w++){
            singleNeuron += input_layer[w] * hidden_layer.weights[i][w];
        }
        singleNeuron += hidden_layer.biases[i];
        result.values[i] = singleNeuron;
    }

    results = result;
}

void HandleActivationFunction(Container& currentValues, HiddenLayer& hidden_layer){
    if(hidden_layer.activationFunction == "relu"){
        for(int i = 0; i < currentValues.values.size(); i++){
            currentValues.values[i] = relu(currentValues.values[i]);
        }
    }else if(hidden_layer.activationFunction == "sigmoid"){
        for(int i = 0; i < currentValues.values.size(); i++){
            currentValues.values[i] = sigmoid(currentValues.values[i]);
        }
    }else if(hidden_layer.activationFunction == ""){
        std::cout << "no activation set for layer" << std::endl;
    }else{
        std::cout << "unknown activation function: " << hidden_layer.activationFunction << std::endl;
    }
}

Container FullForward(InputLayer& input_layer, std::vector<HiddenLayer>& ListOfHiddenLayer){
    //number of layers in model, one Inputlayer + one Outputlayer + x Hiddenlayer
    int num_hiddenlayer = ListOfHiddenLayer.size();

    //perform Inputlayer to first layer of Hiddenlayers calculations
    Container init_result = Container(ListOfHiddenLayer[0].layerNeurons);
    SingleForward(input_layer.inputValues, ListOfHiddenLayer[0], init_result);
    HandleActivationFunction(init_result, ListOfHiddenLayer[0]);

    //perform init_results to second hiddenlayer calculations and following
    for(int i = 1; i < num_hiddenlayer; i++){
        SingleForward(init_result.values, ListOfHiddenLayer[i], init_result);
        HandleActivationFunction(init_result, ListOfHiddenLayer[i]);
    }

    return init_result;
}

void loadModel(InputLayer& input_layer, std::vector<HiddenLayer>& hidden_layers){
    //Define .aywo txt file: 1. Line -> description (size,params), 2. Line -> empty, 3. Line -> start of weights and biases (one line=one layer) = layers separated by line (allocate space in file)

    //name of the model
    std::string name = "MLP-XOR.txt";
    std::string path = "models/" + name;

    //initialize model file and save description
    std::vector<double> modelsize;
    std::ifstream modelFileDescription(path);
    std::ifstream modelFile(path);
    std::string description;
    while(std::getline(modelFileDescription, description)){
        int posA = 0;
        while((posA = description.find("{")) != std::string::npos){
            description.erase(0, posA + 1);
            int posB = 0;
            while((posB = description.find(",")) != std::string::npos){
                modelsize.push_back(std::stod(description.substr(0, posB)));
                description.erase(0, posB + 1);
            }
        }
        break;
    }

    //read each layer as a block of lines with each neurons weights and biases being one line (to read model its structure needs to be known)
    //hidden_layer
    std::string line;
    int neuron_num = 0;
    int lineskip = 2 + (input_layer.num_inputs + 1);
    int skipCounter = 0;
    int layer = 0;

    while(std::getline(modelFile, line)){
        //skip description and input_layer
    
        if(skipCounter < lineskip){
            skipCounter++;
            continue;
        }else{
            //go through each hidden layer
            if(line.empty()){
                layer++;
                neuron_num = 0;
                continue;
            }else{
                /*
                std::cout << line << ", layer: " << layer << std::endl;
                */
                int pos = 0;
                int weight_num = 0;

                while((pos = line.find(",")) != std::string::npos){
                    if(weight_num < modelsize[layer]){
                        //must be weight
                        hidden_layers[layer].weights[neuron_num][weight_num] = std::stod(line.substr(0, pos)); //set weight
                    }else{
                        //must be bias
                        hidden_layers[layer].biases[neuron_num] = std::stod(line.substr(0, pos)); //set bias
                    }
                    line.erase(0, pos + 1); //erase weight/bias and ","
                    weight_num++;
                }
            }
        }
        neuron_num++;
    }
}

void writeModel(InputLayer& input_layer, std::vector<HiddenLayer>& hidden_layers){
    //Define .aywo txt file: 1. Line -> description (size,params), 2. Line -> empty, 3. Line -> start of weights and biases (one line=one layer) = layers separated by line (allocate space in file)

    //name of the model
    std::string name = "MLP-XOR.txt";
    std::string path = "models/" + name;

    //initialize model file (Description and empty assigned)
    std::ofstream modelFile(path);
    modelFile << "modellsize:{" << input_layer.inputValues.size() << ",";
    for(HiddenLayer layer : hidden_layers){
        modelFile << layer.layerNeurons << ",";
    }
    modelFile << "}" << "\n" << "\n";
    
    //write each layer as a block of lines with each neurons weights and biases being one line (to read model its structure needs to be known)
    //input_layer
    for(int i = 0; i < input_layer.num_inputs; i++){
        modelFile << input_layer.inputValues[i];
        modelFile << ",";
        modelFile << "\n";
    }
    modelFile << "\n";

    
    //hidden_layers
    for(HiddenLayer layer : hidden_layers){
        for(int i = 0; i < layer.layerNeurons; i++){
            //weights
            for(int c = 0; c < layer.weights[i].size(); c++){
                
                modelFile << layer.weights[i][c];
                modelFile << ",";
            }
            //biases
            modelFile << layer.biases[i];
            modelFile << ",";

            //new line for next neuron
            modelFile << "\n";
        }

        modelFile << "\n";
    }
    
    //close
    modelFile.close();
}

int main(){

    //Definde model structure
    std::vector<double> input = {1, 1};

    InputLayer input_layer = InputLayer(2, input);
    std::vector<HiddenLayer> hiddenLayerList = {
        HiddenLayer(2, 2, "relu"),
        HiddenLayer(2, 1, "sigmoid")
    };

    loadModel(input_layer, hiddenLayerList);

    //start forwardpropagation through network
    Container out = FullForward(input_layer, hiddenLayerList);

    //read contents of output_layer
    std::cout << "Output layer size: " << out.values.size() << "\n";
    for(int i = 0; i < out.values.size(); i++){
        std::cout << out.values[i] << ", ";
    }
    std::cout << std::endl;

    //writeModel(input_layer, hiddenLayerList);
    //loadModel(input_layer, hiddenLayerList);
    //writeModel(input_slayer, hiddenLayerList);



    return 0;
}