package com.neural.network.lib.utils;

import com.neural.network.lib.Neuron;

public class NeuralNetworkArrayUtils {
    private NeuralNetworkArrayUtils(){}

    public static double[][] getRawArray(Neuron[][] neurons){
        int rowLength = neurons.length;
        int columnLength = neurons[0].length;
        double[][] values = new double[rowLength][columnLength];
        for(int row=0; row < rowLength; row++){
            for(int column=0; column < columnLength; column++){
                values[row][column] = neurons[row][column].getInput();
            }
        }
        return values;
    }
}
