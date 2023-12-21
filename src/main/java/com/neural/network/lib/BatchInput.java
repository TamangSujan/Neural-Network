package com.neural.network.lib;

public class BatchInput {
    private Neuron[][] neurons;
    public BatchInput(double[][] batchInputs){
        initBatchInput(batchInputs);
    }

    private void initBatchInput(double[][] batchInputs) {
        int rowLength = batchInputs.length;
        int columnLength = batchInputs[0].length;
        neurons = new Neuron[rowLength][columnLength];
        for(int row=0; row < rowLength; row++){
            for (int column=0; column < columnLength; column++){
                neurons[row][column] = new Neuron(batchInputs[row][column]);
            }
        }
    }

    public Neuron[][] getBatchNeurons(){
        return neurons;
    }

    private Neuron[] getNeurons(double[] inputs){
        int inputLength = inputs.length;
        Neuron[] neurons = new Neuron[inputLength];
        for(int i=0; i<inputLength; i++){
            neurons[i] = new Neuron(inputs[i]);
        }
        return neurons;
    }
}
