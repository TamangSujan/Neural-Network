package com.neural.network.lib.activation;

public class StepFunction implements ActivationFunction{
    @Override
    public double[][] getActivatedValue(double[][] inputs) {
        int inputRowLength = inputs.length;
        int inputColumnLength = inputs[0].length;
        double[][] outputs = new double[inputRowLength][inputColumnLength];
        for(int row=0; row < inputRowLength; row++){
            for(int column=0; column < inputColumnLength; column++){
                if(inputs[row][column] < 1)
                    outputs[row][column] = 0;
                else
                    outputs[row][column] = 1;
            }
        }
        return outputs;
    }
}
