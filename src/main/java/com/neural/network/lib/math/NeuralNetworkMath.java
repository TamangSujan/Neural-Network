package com.neural.network.lib.math;

import com.neural.network.lib.Neuron;

public class NeuralNetworkMath {
    private NeuralNetworkMath(){}

    public static double[] getDotProductOf(double[][] weights, Neuron[] inputNeurons){
        int rowLength = weights.length;
        int columnLength = weights[0].length;
        double[] dotProduct = new double[rowLength];
        for(int row=0; row<rowLength; row++){
            for(int column = 0; column < columnLength; column++){
                dotProduct[row] += inputNeurons[column].getInput() * weights[row][column];
            }
        }
        return dotProduct;
    }

    public static double[][] getDotProductOf(double[][] weights, Neuron[][] batchInputNeurons){
        checkMatrixEligibility(weights, batchInputNeurons);
        double[][] resultMatrix = new double[batchInputNeurons.length][weights[0].length];
        int resultRowLength = resultMatrix.length;
        int resultColumnLength = resultMatrix[0].length;
        int multipleLoopLength = weights.length;
        for(int row=0; row < resultRowLength; row++){
            for(int column=0; column < resultColumnLength; column++){
                for(int k=0; k<multipleLoopLength; k++){
                    resultMatrix[row][column] += batchInputNeurons[row][k].getInput() * weights[k][column];
                }
            }
        }
        return resultMatrix;
    }

    public static double[][] getDotProductByTransposeOf(double[][] weights, Neuron[][] batchInputNeurons){
        double[][] transposeWeights = getTransposedMatrix(weights);
        checkTransposeMatrixEligibility(transposeWeights, batchInputNeurons);
        return getDotProductOf(transposeWeights, batchInputNeurons);
    }

    private static void checkTransposeMatrixEligibility(double[][] weights, Neuron[][] batchInputNeurons) {
        if (weights.length != batchInputNeurons[0].length)
            throw new IllegalArgumentException("Transpose Matrix A rows is not equal with Matrix B rows");
    }

    private static void checkTransposeMatrixEligibility(double[][] weights, Neuron[] batchInputNeurons) {
        if (weights.length != batchInputNeurons.length)
            throw new IllegalArgumentException("Transpose Matrix A rows is not equal with Matrix B rows");
    }

    private static double[][] getTransposedMatrix(double[][] matrix){
        int transposeRowLength = matrix[0].length;
        int transposeColumnLength = matrix.length;
        double[][] transposedMatrix = new double[matrix[0].length][matrix.length];
        for(int row=0; row < transposeRowLength; row++){
            for(int column=0; column < transposeColumnLength; column++){
                transposedMatrix[row][column] = matrix[column][row];
            }
        }
        return transposedMatrix;
    }

    private static void checkMatrixEligibility(double[][] weights, Neuron[][] batchInputNeurons){
        int weightColumnSize = weights.length;
        int batchInputRowSize = batchInputNeurons[0].length;
        if(weightColumnSize != batchInputRowSize)
            throw new IllegalArgumentException("Invalid Matrix Size, No of columns in weights is not equal to no of rows in neurons");
    }

    public static double[] addVectors(double[] a, double[] b){
        if(a.length != b.length)
            throw new IllegalArgumentException("Matrix A and Matrix B are unequal in size!");
        int matrixLength = a.length;
        double[] sum = new double[matrixLength];
        for(int i=0; i<matrixLength; i++){
            sum[i] = a[i] + b[i];
        }
        return sum;
    }

    public static double[] addVectors(double[][] a, double[] b){
        if(a.length != b.length)
            throw new IllegalArgumentException("Matrix A and Matrix B are unequal in size!");
        int matrixRowLength = a.length;
        int matrixColumnLength = a[0].length;
        double[] sum = new double[matrixRowLength];
        for(int i=0; i<matrixRowLength; i++){
            sum[i] = b[i];
            for(int j=0; j<matrixColumnLength; j++){
                sum[i] += a[i][j];
            }
        }
        return sum;
    }

    public static double[][] addVectors(double[] a, double[][] b){
        if(a.length != b.length)
            throw new IllegalArgumentException("Matrix A and Matrix B are unequal in size!");
        int matrixRowLength = b.length;
        int matrixColumnLength = b[0].length;
        double[][] sum = new double[matrixRowLength][matrixColumnLength];
        for(int i=0; i<matrixRowLength; i++){
            for(int j=0; j<matrixColumnLength; j++){
                sum[i][j] = b[i][j] + a[j];
            }
        }
        return sum;
    }

    public static double[] multiplyVectors(double[] a, double[] b){
        if(a.length != b.length)
            throw new IllegalArgumentException("Matrix A and Matrix B are unequal in size!");
        int matrixLength = a.length;
        double[] multiple = new double[matrixLength];
        for(int i=0; i<matrixLength; i++){
            multiple[i] = a[i] * b[i];
        }
        return multiple;
    }

    public static void printMatrix(double[] matrix){
        for(double row: matrix){
                System.out.print(row + " | ");
        }
        System.out.println();
    }

    public static void printMatrix(double[][] matrix){
        for(double[] row: matrix){
            for(double column: row){
                System.out.print(column + " | ");
            }
            System.out.println();
        }
    }

    public static void printMatrix(Neuron[][] matrix){
        for(Neuron[] row: matrix){
            for(Neuron column: row){
                System.out.print(column.getInput() + " | ");
            }
            System.out.println();
        }
    }

    public static void printMatrix(Neuron[] matrix){
        for(Neuron row: matrix){
            System.out.print(row.getInput() + " | ");
        }
        System.out.println();
    }
}
