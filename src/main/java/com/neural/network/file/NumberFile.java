package com.neural.network.file;

public class NumberFile {
    int value;
    double[][] matrix;

    public NumberFile(int value, double[][] matrix){
        this.matrix = matrix;
        this.value = value;
    }

    public int getValue(){
        return value;
    }

    public double[][] getMatrixValue(){
        return matrix;
    }
}
