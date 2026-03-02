
#include <vector>
#include <iostream>
#include "GAAS.h"
#include "kernels.h"

int main(int argc, char** argv) {

    Argument arg;
    int dataset_No = 4;
    bool model = 1; 
    double epsilon = 0.5; 
    double eta = 2000; 
    int batch = 4; 
    int start_time = 0; 
    int reuse = true; 
    int beta = 8; 
    bool large = false;
    for (int i = 0; i < argc; i++) {
        if (argv[i] == std::string("-dataset_No")) 
            dataset_No = std::atoi(argv[i+1]);
        if (argv[i] == std::string("-model")) 
            model = std::atoi(argv[i+1]);
        if (argv[i] == std::string("-eps"))              
            epsilon = std::atof(argv[i+1]);
        if (argv[i] == std::string("-eta")) 
            eta = std::atof(argv[i+1]);
        if (argv[i] == std::string("-batch")) 
            batch = std::atoi(argv[i+1]);
        if (argv[i] == std::string("-start_time")) 
            start_time = std::atoi(argv[i+1]);
        if (argv[i] == std::string("-reuse")) 
            reuse = std::atoi(argv[i+1]);
        if (argv[i] == std::string("-beta")) 
            beta = std::atoi(argv[i+1]);
        if (argv[i] == std::string("-large")) 
            large = std::atoi(argv[i+1]);
    }
    arg.dataset_No = dataset_No;
    arg.model = model;
    arg.epsilon = epsilon;
    arg.eta = eta;
    arg.batch = batch;
    arg.start_time = start_time;
    arg.large = large;
    arg.reuse = reuse;
    arg.beta = beta;

    std::vector<std::string> dataset={"DBLP_sym","Youtube_sym","flickr","livejournal"};
    std::string file_name =  dataset[arg.dataset_No];
    const bool ICLT = arg.model;

    GAAS gaas(file_name, arg.epsilon, true, arg.large, ICLT);
    std::cout << "eta = " << arg.eta << ", eps =  " << arg.epsilon <<  ", batch = "<< arg.batch << ", model = " << (ICLT?"IC":"LT") << std::endl;
    gaas.seedminmization(arg);
}