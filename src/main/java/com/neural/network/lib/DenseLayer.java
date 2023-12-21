package com.neural.network.lib;

import com.neural.network.lib.activation.ActivationFunction;
import com.neural.network.lib.math.NeuralNetworkMath;
import com.neural.network.lib.utils.NeuralNetworkArrayUtils;

import java.util.concurrent.ThreadLocalRandom;

public class DenseLayer implements Layer{
    private Neuron[][] inputNeurons;
    private Neuron[][] outputNeurons;
    private double[][] weights;
    private double[] biases;
    private final int outputSize;

    private final ActivationFunction activationFunction;

    public DenseLayer(Neuron[][] inputNeurons, ActivationFunction activationFunction, int outputSize){
        this.inputNeurons = inputNeurons;
        this.outputSize = outputSize;
        this.outputNeurons = new Neuron[outputSize][inputNeurons.length];
        this.activationFunction = activationFunction;
        initWeights();
        initBiases();
    }

    private void initBiases() {
        biases = new double[outputSize];
        for(int i=0; i<outputSize; i++){
            biases[i] = getRandomNumber();
        }
    }

    public double[] getBiases(){
        return biases;
    }


    public void trainNeurons(){
        double[][] trainedNeurons = activationFunction.getActivatedValue(NeuralNetworkMath.getDotProductByTransposeOf(weights, inputNeurons));
        int rowLength = trainedNeurons.length;
        int columnLength = trainedNeurons[0].length;
        Neuron[][] neurons = new Neuron[rowLength][columnLength];
        for(int row=0; row < rowLength; row++){
            for(int column=0; column < columnLength; column++){
                neurons[row][column] = new Neuron(trainedNeurons[row][column] + biases[column]);
            }
        }
        outputNeurons = neurons;
    }

    public Neuron[][] getOutputNeurons(){
        return outputNeurons;
    }

    public double[][] getSpecificOutput(ActivationFunction activationFunction){
        return activationFunction.getActivatedValue(NeuralNetworkArrayUtils.getRawArray(outputNeurons));
    }

    private void initWeights() {
        int inputNeuronColumnLength = inputNeurons[0].length;
        weights = new double[outputSize][inputNeuronColumnLength];
        for(int row = 0; row < outputSize; row++){
            for(int column = 0; column < inputNeuronColumnLength; column++){
              weights[row][column] = getRandomNumber();
            }
        }
    }

    public double[][] getWeights(){
        return weights;
    }

    private double getRandomNumber(){
        ThreadLocalRandom randomNumber = ThreadLocalRandom.current();
        return randomNumber.nextDouble(-1.0, 1.0);
    }

    public void adjustWeightAndBias(double loss, double learningRate, int adjustIndex) {
        double adjustingParameterVectorValue = learningRate * loss;
        int weightColumnLength = weights[0].length;
        for(int column=0; column < weightColumnLength; column++){
            weights[adjustIndex][column] += adjustingParameterVectorValue;
        }
        biases[adjustIndex] -= adjustingParameterVectorValue;
    }

    public void setInputNeurons(Neuron[][] inputNeurons) {
        this.inputNeurons = inputNeurons;
    }
}
