#include "frft_engine.h"
#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cstring>

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " --alpha <value> --input <comma-separated-values>\n";
    std::cerr << "Example: " << prog_name << " --alpha 1.0 --input 2,1,-1,-2,-1,1\n";
}

std::vector<double> parse_input(const std::string& input_str) {
    std::vector<double> values;
    std::stringstream ss(input_str);
    std::string token;
    
    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        size_t start = token.find_first_not_of(" \t");
        size_t end = token.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos) {
            token = token.substr(start, end - start + 1);
        }
        
        try {
            values.push_back(std::stod(token));
        } catch (const std::exception& e) {
            std::cerr << "Error parsing value: " << token << "\n";
            return {};
        }
    }
    
    return values;
}

int main(int argc, char* argv[]) {
    double alpha = 0.0;
    std::string input_str;
    bool has_alpha = false;
    bool has_input = false;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--alpha") == 0 || strcmp(argv[i], "-a") == 0) {
            if (i + 1 < argc) {
                alpha = std::atof(argv[++i]);
                has_alpha = true;
            } else {
                std::cerr << "Error: --alpha requires a value\n";
                print_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "--input") == 0 || strcmp(argv[i], "-i") == 0) {
            if (i + 1 < argc) {
                input_str = argv[++i];
                has_input = true;
            } else {
                std::cerr << "Error: --input requires a value\n";
                print_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Error: Unknown argument: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (!has_alpha || !has_input) {
        std::cerr << "Error: Both --alpha and --input are required\n";
        print_usage(argv[0]);
        return 1;
    }
    
    // Parse input values
    std::vector<double> input_values = parse_input(input_str);
    if (input_values.empty()) {
        std::cerr << "Error: No valid input values found\n";
        return 1;
    }
    
    int N = input_values.size();
    
    std::cout << "Input signal: [";
    for (size_t i = 0; i < input_values.size(); ++i) {
        std::cout << input_values[i];
        if (i < input_values.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "Alpha: " << alpha << "\n";
    std::cout << "Signal length: " << N << "\n\n";
    
    // Prepare FRFT engine
    FRFTEngine engine;
    engine.prepare(N);
    
    // Prepare input buffers (real and imaginary parts)
    std::vector<double> real_in(N);
    std::vector<double> imag_in(N, 0.0);  // Imaginary part is zero for real signals
    std::vector<double> real_out(N);
    std::vector<double> imag_out(N);
    
    // Copy input values
    for (int i = 0; i < N; ++i) {
        real_in[i] = input_values[i];
    }
    
    // Compute FRFT
    bool success = engine.compute(
        real_in.data(), imag_in.data(),
        real_out.data(), imag_out.data(),
        N, alpha
    );
    
    if (!success) {
        std::cerr << "Error: FRFT computation failed\n";
        return 1;
    }
    
    // Print results
    std::cout << "FRFT Output:\n";
    std::cout << "─────────────────────────────────────────\n";
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "\nReal part:\n[";
    for (int i = 0; i < N; ++i) {
        std::cout << std::setw(10) << real_out[i];
        if (i < N - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    std::cout << "\nImaginary part:\n[";
    for (int i = 0; i < N; ++i) {
        std::cout << std::setw(10) << imag_out[i];
        if (i < N - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    // Print magnitude
    std::cout << "\nMagnitude:\n[";
    for (int i = 0; i < N; ++i) {
        double magnitude = std::sqrt(real_out[i] * real_out[i] + imag_out[i] * imag_out[i]);
        std::cout << std::setw(10) << magnitude;
        if (i < N - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    // Print phase
    std::cout << "\nPhase (radians):\n[";
    for (int i = 0; i < N; ++i) {
        double phase = std::atan2(imag_out[i], real_out[i]);
        std::cout << std::setw(10) << phase;
        if (i < N - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    return 0;
}