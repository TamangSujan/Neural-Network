package com.neural.network.lib.activation;

import java.util.Objects;

public class ReLUFunction implements ActivationFunction{
    private static ReLUFunction reLUFunction;
    private ReLUFunction(){}

    public static ReLUFunction getInstance(){
        if(Objects.isNull(reLUFunction))
            reLUFunction = new ReLUFunction();
        return reLUFunction;
    }

    @Override
    public double[][] getActivatedValue(double[][] inputs) {
        int inputRowLength = inputs.length;
        int inputColumnLength = inputs[0].length;
        double[][] outputs = new double[inputRowLength][inputColumnLength];
        for(int row=0; row < inputRowLength; row++){
            for(int column=0; column < inputColumnLength; column++){
                outputs[row][column] = Math.max(0, inputs[row][column]);
            }
        }
        return outputs;
    }
}
