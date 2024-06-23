#include <iostream>
#include <cstdint>
#include <fstream>

int main(){
    std::string path = "value.txt";

    //initialize model file (Description and empty assigned)
    std::ofstream modelFile(path);

    return 0;
}