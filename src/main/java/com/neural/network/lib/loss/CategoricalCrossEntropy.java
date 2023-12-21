package com.neural.network.lib.loss;

public interface CategoricalCrossEntropy {
    double getLossValue(double[] inputs, int target);
}
