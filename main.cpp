#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>

enum class ActivationType {
    Sigmoid,
    ReLU,
    Tanh
};

double activate(double x, ActivationType activationType) {
    switch (activationType) {
        case ActivationType::Sigmoid:
            return 1.0 / (1.0 + std::exp(-x));
        case ActivationType::ReLU:
            return std::max(0.0, x);
        case ActivationType::Tanh:
            return std::tanh(x);
        default:
            return x;
    }
}

double activateDerivative(double x, ActivationType activationType) {
    switch (activationType) {
        case ActivationType::Sigmoid: {
            double sigmoidX = activate(x, ActivationType::Sigmoid);
            return sigmoidX * (1.0 - sigmoidX);
        }
        case ActivationType::ReLU:
            return x > 0.0 ? 1.0 : 0.0;
        case ActivationType::Tanh:
            return 1.0 - std::pow(activate(x, ActivationType::Tanh), 2);
        default:
            return 1.0;
    }
}

std::mt19937 rng(std::random_device{}());

class NeuralNetwork {
private:
    std::vector<std::vector<double>> layers;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> batchNormalizationParams;

public:
    NeuralNetwork(const std::vector<int>& layerSizes, ActivationType activationType) {
        int numLayers = layerSizes.size();
        layers.resize(numLayers);
        weights.resize(numLayers - 1);
        batchNormalizationParams.resize(numLayers - 2);

        for (int i = 0; i < numLayers; ++i) {
            layers[i].resize(layerSizes[i]);
        }

        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int i = 0; i < numLayers - 1; ++i) {
            int currentLayerSize = layerSizes[i];
            int nextLayerSize = layerSizes[i + 1];
            weights[i].resize(currentLayerSize, std::vector<double>(nextLayerSize));
            for (int j = 0; j < currentLayerSize; ++j) {
                for (int k = 0; k < nextLayerSize; ++k) {
                    weights[i][j][k] = dist(rng);
                }
            }
        }
    }

    std::vector<double> feedForward(const std::vector<double>& inputs, ActivationType activationType) {
        layers[0] = inputs;

        for (size_t i = 0; i < weights.size(); ++i) {
            int currentLayerSize = layers[i].size();
            int nextLayerSize = layers[i + 1].size();
            for (int j = 0; j < nextLayerSize; ++j) {
                double sum = 0.0;
                for (int k = 0; k < currentLayerSize; ++k) {
                    sum += layers[i][k] * weights[i][k][j];
                }
                layers[i + 1][j] = activate(sum, activationType);
            }
        }

        return layers.back();
    }

    void train(const std::vector<std::vector<double>>& trainingData, const std::vector<std::vector<double>>& targetOutputs,
               int numEpochs, double learningRate, int batchSize, double regularization, double dropoutRate, int& numErrors,
               std::ofstream& logfile, ActivationType activationType) {
        int numSamples = trainingData.size();
        int numLayers = layers.size();
        int outputSize = layers.back().size();
        int numBatches = numSamples / batchSize;

        std::vector<int> indices(numSamples);
        std::iota(indices.begin(), indices.end(), 0);

        for (int epoch = 0; epoch < numEpochs; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng);

            for (int batch = 0; batch < numBatches; ++batch) {
                std::vector<std::vector<double>> batchInputs(batchSize);
                std::vector<std::vector<double>> batchOutputs(batchSize);

                for (int i = 0; i < batchSize; ++i) {
                    int index = indices[batch * batchSize + i];
                    batchInputs[i] = trainingData[index];
                    batchOutputs[i] = targetOutputs[index];
                }

                for (int sample = 0; sample < batchSize; ++sample) {
                    feedForward(batchInputs[sample], activationType);
                    applyBatchNormalization(numLayers - 2, 0.9);  

                    std::vector<std::vector<double>> errors(numLayers, std::vector<double>(0));
                    errors.back().resize(outputSize);
                    for (int i = 0; i < outputSize; ++i) {
                        errors.back()[i] = batchOutputs[sample][i] - layers.back()[i];
                    }

                    for (int i = numLayers - 2; i >= 0; --i) {
                        int currentLayerSize = layers[i].size();
                        int nextLayerSize = layers[i + 1].size();
                        errors[i].resize(currentLayerSize);
                        for (int j = 0; j < currentLayerSize; ++j) {
                            double sum = 0.0;
                            for (int k = 0; k < nextLayerSize; ++k) {
                                sum += weights[i][j][k] * errors[i + 1][k];
                            }
                            errors[i][j] = sum * activateDerivative(layers[i][j], activationType);
                        }
                    }

                    for (int i = 0; i < numLayers - 1; ++i) {
                        int currentLayerSize = layers[i].size();
                        int nextLayerSize = layers[i + 1].size();
                        for (int j = 0; j < currentLayerSize; ++j) {
                            for (int k = 0; k < nextLayerSize; ++k) {
                                weights[i][j][k] += learningRate * layers[i][j] * errors[i + 1][k];
                            }
                        }
                    }
                }
            }
        }

        numErrors = 0;
        double loss = 0.0;
        for (int sample = 0; sample < numSamples; ++sample) {
            std::vector<double> output = feedForward(trainingData[sample], activationType);
            for (int i = 0; i < outputSize; ++i) {
                if (output[i] != targetOutputs[sample][i]) {
                    ++numErrors;
                    if (logfile.is_open()) {
                        logfile << "error in sample " << sample << ": expected " << targetOutputs[sample][i]
                                << ", got " << output[i] << std::endl;
                    }
                    break;
                }
            }

            for (int i = 0; i < outputSize; ++i) {
                double error = targetOutputs[sample][i] - output[i];
                loss += 0.5 * error * error;
            }
        }

        double regularizationLoss = 0.0;
        for (int i = 0; i < numLayers - 1; ++i) {
            int currentLayerSize = layers[i].size();
            int nextLayerSize = layers[i + 1].size();
            for (int j = 0; j < currentLayerSize; ++j) {
                for (int k = 0; k < nextLayerSize; ++k) {
                    regularizationLoss += 0.5 * regularization * std::pow(weights[i][j][k], 2);
                }
            }
        }

        loss += regularizationLoss;

        if (logfile.is_open()) {
            logfile << "Loss: " << loss << std::endl;
        }
    }

    void printOutput() const {
        for (double output : layers.back()) {
            std::cout << output << " ";
        }
        std::cout << std::endl;
    }

private:
    void applyBatchNormalization(int layerIndex, double decayRate) {
        int layerSize = layers[layerIndex].size();
        std::vector<double>& means = batchNormalizationParams[layerIndex];
        std::vector<double>& variances = batchNormalizationParams[layerIndex + 1];

        if (means.empty() && variances.empty()) {
            means.resize(layerSize);
            variances.resize(layerSize);
            for (int i = 0; i < layerSize; ++i) {
                means[i] = layers[layerIndex][i];
                variances[i] = std::pow(layers[layerIndex][i], 2);
            }
        } else {
            for (int i = 0; i < layerSize; ++i) {
                means[i] = decayRate * means[i] + (1.0 - decayRate) * layers[layerIndex][i];
                variances[i] = decayRate * variances[i] + (1.0 - decayRate) * std::pow(layers[layerIndex][i], 2);
            }
        }

        for (int i = 0; i < layerSize; ++i) {
            double mean = means[i];
            double variance = variances[i];
            double stdDev = std::sqrt(variance - std::pow(mean, 2));

            layers[layerIndex][i] = (layers[layerIndex][i] - mean) / stdDev;
        }
    }
};

int main() {
    std::vector<int> layerSizes = {2, 1000, 1000, 1000, 1000, 1};  
    NeuralNetwork nn(layerSizes, ActivationType::ReLU);

    std::vector<std::vector<double>> trainingData = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targetOutputs = {{0}, {1}, {1}, {0}};

    int numEpochs = 10000;
    double learningRate = 0.1;
    int batchSize = 2;
    double regularization = 0.01;
    double dropoutRate = 0.2;
    int numErrors = 0;

    std::ofstream logfile("logs.txt");
    if (logfile.is_open()) {
        nn.train(trainingData, targetOutputs, numEpochs, learningRate, batchSize, regularization, dropoutRate,
                 numErrors, logfile, ActivationType::ReLU);
        logfile.close();
    } else {
        std::cout << "Error opening log file." << std::endl;
    }

    nn.printOutput();

    return 0;
}
