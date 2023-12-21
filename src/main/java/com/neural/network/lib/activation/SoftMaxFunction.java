package com.neural.network.lib.activation;

import com.neural.network.lib.math.PureMath;

import java.util.Objects;

public class SoftMaxFunction implements ActivationFunction{
    private static SoftMaxFunction softMaxFunction;

    public static SoftMaxFunction getInstance(){
        if(Objects.isNull(softMaxFunction))
            softMaxFunction = new SoftMaxFunction();
        return softMaxFunction;
    }
    @Override
    public double[][] getActivatedValue(double[][] inputs) {
        int inputRowLength = inputs.length;
        int inputColumnLength = inputs[0].length;
        double[][] outputs = new double[inputRowLength][inputColumnLength];
        for(int row=0; row < inputRowLength; row++){
            double rowMax = PureMath.getMaxOf(inputs[row]);
            for(int column=0; column < inputColumnLength; column++){
                double exponentUnderBufferValue = PureMath.getValueOfExponentiationToPower(inputs[row][column] - rowMax);
                outputs[row][column] = exponentUnderBufferValue;
            }
        }
        for(int row=0; row < inputRowLength; row++){
            double rowTotal = PureMath.getSumOf(outputs[row]);
            for(int column=0; column < inputColumnLength; column++){
                double exponentUnderBufferValue = PureMath.getNormalizationValue(outputs[row][column], rowTotal);
                outputs[row][column] = exponentUnderBufferValue;
            }
        }
        return outputs;
    }
}
