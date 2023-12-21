package com.neural.network.lib.loss;

public class OneHotEncoding implements CategoricalCrossEntropy {
    @Override
    public double getLossValue(double[] inputs, int target) {
        /*
        double[] oneHotEncodingVector = getEncodingVector(inputs.length, target);
        NeuralNetworkMath.printMatrix(inputs);
        double[] logInputValues = getLogValueInputs(inputs);
        double[] multipleOneHotEncodedValue = NeuralNetworkMath.multiplyVectors(logInputValues, oneHotEncodingVector);
        double loss = PureMath.getSumOf(multipleOneHotEncodedValue);
        System.out.println(-Math.log(inputs[target]));
         */
        //Zero is being clipped in order to avoid infinity if log calculation as log 0 is infinity
        double clippedZero = Math.min(0.9999999, Math.max(inputs[target],  0.0000001)); // 0.0000001 is 1e-7 && 0.9999999 is 1-1e-7
        return -Math.log(clippedZero);
    }

    private double[] getEncodingVector(int length, int target) {
        if(target < 0 || target >= length)
            throw new IllegalArgumentException("Invalid target!");
        double[] encodingVector = new double[length];
        encodingVector[target] = 1;
        return encodingVector;
    }

    private double[] getLogValueInputs(double[] inputs){
        int inputLength = inputs.length;
        double[] logInputs = new double[inputLength];
        for(int i=0; i < inputLength; i++){
            logInputs[i] = Math.log(inputs[i]);
        }
        return logInputs;
    }
}
