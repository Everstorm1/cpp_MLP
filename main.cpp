#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <math.h>
#include "network.h"

int main(){
 
    std::vector<double> input = {0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0};
    Layer hiddenLayer1(12, 16, "relu");
    Layer hiddenLayer2(16, 25, "relu");
    Layer outputLayer(25, 2, "sigmoid");

    Network network({hiddenLayer1,
                     hiddenLayer2,
                     outputLayer  });

    std::vector<double> output = network.forwardPass(input);


    std::cout << "Output layer size: " << output.size() << "\n";
    for (double value : output) {
        std::cout << value << ", ";
    }
    std::cout << std::endl;

    std::cin.get();
    
    return 0;
}