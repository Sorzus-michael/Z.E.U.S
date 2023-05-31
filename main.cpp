#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>

 enum class ActivationType 
{ 
Sigmoid, 
ReLU, 
Tanh 
};

 
double
activate (double x, ActivationType activationType) 
{
  
switch (activationType)
    
    {
    
case ActivationType::Sigmoid:
      
return 1.0 / (1.0 + std::exp (-x));
    
case ActivationType::ReLU:
      
return std::max (0.0, x);
    
case ActivationType::Tanh:
      
return std::tanh (x);
    
default:
      
return x;
    
}

}


 
double
activateDerivative (double x, ActivationType activationType) 
{
  
switch (activationType)
    
    {
    
case ActivationType::Sigmoid:
      
      {
	
double sigmoidX = activate (x, ActivationType::Sigmoid);
	
return sigmoidX * (1.0 - sigmoidX);
      
}
    
case ActivationType::ReLU:
      
return x > 0.0 ? 1.0 : 0.0;
    
case ActivationType::Tanh:
      
return 1.0 - std::pow (activate (x, ActivationType::Tanh), 2);
    
default:
      
return 1.0;
    
}

}


 
std::mt19937 rng (std::random_device
		     {
		     }

		     ());

 
class NeuralNetwork 
{

private:
std::vector < std::vector < double >>
    layers;
  
std::vector < std::vector < std::vector < double >>>
    weights;

 
public:
NeuralNetwork (const std::vector < int >&layerSizes,
		  ActivationType activationType) 
  {
    
int
      numLayers = layerSizes.size ();
    
layers.resize (numLayers);
    
weights.resize (numLayers - 1);
    
 
std::uniform_real_distribution < double >
    dist (-1.0, 1.0);
    
 
for (int i = 0; i < numLayers; ++i)
      
      {
	
layers[i].resize (layerSizes[i]);
    
} 
 
for (int i = 0; i < numLayers - 1; ++i)
      
      {
	
int
	  currentLayerSize = layerSizes[i];
	
int
	  nextLayerSize = layerSizes[i + 1];
	
weights[i].resize (currentLayerSize,
			    std::vector < double >(nextLayerSize));
	
 
for (int j = 0; j < currentLayerSize; ++j)
	  
	  {
	    
for (int k = 0; k < nextLayerSize; ++k)
	      
	      {
		
weights[i][j][k] = dist (rng);
  
} 
} 
} 
} 
 
std::vector < double >
  feedForward (const std::vector < double >&inputs,
	       ActivationType activationType) 
  {
    
layers[0] = inputs;
    
 
for (size_t i = 0; i < weights.size (); ++i)
      
      {
	
int
	  currentLayerSize = layers[i].size ();
	
int
	  nextLayerSize = layers[i + 1].size ();
	
 
for (int j = 0; j < nextLayerSize; ++j)
	  
	  {
	    
double
	      sum = 0.0;
	    
 
for (int k = 0; k < currentLayerSize; ++k)
	      
	      {
		
sum += layers[i][k] * weights[i][k][j];
	      
} 
 
layers[i + 1][j] = activate (sum, activationType);
      
} 
} 
 
return layers.back ();
  
}
  
 
void
  train (const std::vector < std::vector < double >>&trainingData,
	 
const std::vector < std::vector < double >>&targetOutputs,
	 
int numEpochs, double learningRate, int batchSize,
	 
double regularization, std::string logFilename,
	 
ActivationType activationType) 
  {
    
int
      numSamples = trainingData.size ();
    
int
      numLayers = layers.size ();
    
 
std::ofstream logFile (logFilename);
    
 
for (int epoch = 1; epoch <= numEpochs; ++epoch)
      
      {
	
std::vector < int >
	indices (numSamples);
	
std::iota (indices.begin (), indices.end (), 0);
	
std::shuffle (indices.begin (), indices.end (), rng);
	
 
double
	  epochError = 0.0;
	
 
for (int i = 0; i < numSamples; i += batchSize)
	  
	  {
	    
std::vector < std::vector < std::vector < double >>>
	    gradients (weights.size ());
	  
 
for (auto & gradient:gradients)
	      
	      {
		
gradient.resize (gradient.size (),
				  std::vector < double >(gradient[0].size (),
							 0.0));
	    
} 
 
for (int j = i; j < std::min (i + batchSize, numSamples);
			 ++j)
	      
	      {
		
int
		  sampleIndex = indices[j];
		
std::vector < double >
		  output =
		  feedForward (trainingData[sampleIndex], activationType);
		
std::vector < double >
		  target = targetOutputs[sampleIndex];
		
std::vector < double >
		error (target.size ());
		
 
for (size_t k = 0; k < error.size (); ++k)
		  
		  {
		    
error[k] = target[k] - output[k];
		    
epochError += std::pow (error[k], 2);
		  
}
		
 
for (int layerIndex = numLayers - 2; layerIndex >= 0;
			--layerIndex)
		  
		  {
		    
int
		      currentLayerSize = layers[layerIndex].size ();
		    
int
		      nextLayerSize = layers[layerIndex + 1].size ();
		    
 
for (int k = 0; k < nextLayerSize; ++k)
		      
		      {
			
double
			  gradientSum = 0.0;
			
 
for (int n = 0; n < currentLayerSize; ++n)
			  
			  {
			    
gradientSum +=
			      error[k] * weights[layerIndex][n][k];
			    
gradients[layerIndex][n][k] +=
			      layers[layerIndex][n] *
			      activateDerivative (layers[layerIndex + 1][k],
						  activationType);
			  
} 
 
error[k] = gradientSum;
	    
} 
} 
} 
 
for (size_t layerIndex = 0;
			       layerIndex < weights.size (); ++layerIndex)
	      
	      {
		
int
		  currentLayerSize = layers[layerIndex].size ();
		
int
		  nextLayerSize = layers[layerIndex + 1].size ();
		
 
for (int j = 0; j < currentLayerSize; ++j)
		  
		  {
		    
for (int k = 0; k < nextLayerSize; ++k)
		      
		      {
			
weights[layerIndex][j][k] +=
			  learningRate * (gradients[layerIndex][j][k] /
					  batchSize -
					  regularization *
					  weights[layerIndex][j][k]);
	  
} 
} 
} 
} 
 
logFile << "Epoch: " << epoch << " Error: " <<
	  epochError << std::endl;
      
} 
 
logFile.close ();

} 
};


 
int
main () 
{
  
std::vector < std::vector < double >>
  trainingData = { 
{0.0, 0.0}, 
{0.0, 1.0}, 
{1.0, 0.0}, 
{1.0, 1.0} 
  };
  
 
std::vector < std::vector < double >>
  targetOutputs = { 
{0.0}, 
{1.0}, 
{1.0}, 
{0.0} 
  };
  
 
std::vector < int >
  layerSizes = { 2, 8, 1 };
  
NeuralNetwork network (layerSizes, ActivationType::ReLU);
  
 
int
    numEpochs = 5000;
  
double
    learningRate = 0.1;
  
int
    batchSize = 4;
  
double
    regularization = 0.0;
  
std::string logFilename = "log.txt";
  
 
network.train (trainingData, targetOutputs, numEpochs, learningRate,
		    batchSize, regularization, logFilename,
		    ActivationType::ReLU);
  
 
std::cout << "Training complete!" << std::endl;
  
 
std::cout << "Testing..." << std::endl;
  
for (size_t i = 0; i < trainingData.size (); ++i)
    
    {
      
std::
	cout << "Input: " << trainingData[i][0] << ", " << trainingData[i][1]
	<< std::endl;
      
std::vector < double >
	output = network.feedForward (trainingData[i], ActivationType::ReLU);
      
std::cout << "Output: " << output[0] << std::endl;
    
} 
 
return 0;

}


