package com.neural.network.lib.activation;

public class SigmoidFunction implements ActivationFunction{
    @Override
    public double[][] getActivatedValue(double[][] inputs) {
        int inputRowLength = inputs.length;
        int inputColumnLength = inputs[0].length;
        double[][] outputs = new double[inputRowLength][inputColumnLength];
        for(int row=0; row < inputRowLength; row++){
            for(int column=0; column < inputColumnLength; column++){
                    outputs[row][column] = 1 / (1 + Math.exp(-inputs[row][column]));
            }
        }
        return outputs;
    }
}
