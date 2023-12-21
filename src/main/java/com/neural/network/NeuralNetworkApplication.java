package com.neural.network;

import com.neural.network.file.CustomNeuralFileHandler;
import com.neural.network.file.NumberFile;
import com.neural.network.lib.*;
import com.neural.network.lib.activation.ReLUFunction;
import com.neural.network.lib.activation.SoftMaxFunction;
import com.neural.network.lib.loss.CategoricalCrossEntropy;
import com.neural.network.lib.loss.OneHotEncoding;
import com.neural.network.lib.math.NeuralNetworkMath;
import com.neural.network.lib.math.PureMath;

import java.io.IOException;
import java.util.*;

public class NeuralNetworkApplication {
    public static void main(String[] args) {
        Map<String, Integer> numberFileNames = getNumberFilesNames();
        List<NumberFile> numberFiles = getNumberFiles(numberFileNames);
        denseLayerTest(numberFiles);
    }

    private static List<NumberFile> getNumberFiles(Map<String, Integer> numberFileNames) {
        List<NumberFile> numberFiles = new LinkedList<>();
        numberFileNames.forEach((filename, value) ->{
            numberFiles.add(loadNumberFile(filename, value));
        });
        return numberFiles;
    }

    private static NumberFile loadNumberFile(String filename, int numberValue) {
        try {
            return CustomNeuralFileHandler.getNumberFileFrom(filename, numberValue);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static Map<String, Integer> getNumberFilesNames() {
        Map<String, Integer> numberFileNames = new HashMap<>();
        numberFileNames.put("one.txt", 1);
        numberFileNames.put("two.txt", 2);
        numberFileNames.put("three.txt", 3);
        numberFileNames.put("four.txt",4 );
        numberFileNames.put("five.txt", 5);
        numberFileNames.put("six.txt", 6);
        numberFileNames.put("seven.txt", 7);
        numberFileNames.put("eight.txt", 8);
        numberFileNames.put("nine.txt", 9);
        return numberFileNames;
    }

    private static void denseLayerTest(List<NumberFile> numberFiles){
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter which number to train from file: ");
        int fileIndexValue = scanner.nextInt();
        scanner.nextLine();
        double loss = 0;
        BatchInput batchInput = new BatchInput(numberFiles.get(fileIndexValue).getMatrixValue());
        Neuron[][] batchNeurons = batchInput.getBatchNeurons();
        DenseLayer inputLayer = new DenseLayer(batchNeurons, ReLUFunction.getInstance(), 16);
        inputLayer.trainNeurons();
        DenseLayer outputLayer = new DenseLayer(inputLayer.getOutputNeurons(), ReLUFunction.getInstance(), 10);
        outputLayer.trainNeurons();

        int firstHiddenLayerSize = inputLayer.getWeights().length;
        int secondHiddenLayerSize = outputLayer.getWeights().length;
        int firstTrainOffset = 0;
        int secondTrainOffset = 0;
        boolean firstLayerExecution = false;
        boolean secondLayerExecution = false;


        int trainingTimes = 100;
        double previousLoss = 0;
        double learningRate = 0.1;
        for(int i=0; i<trainingTimes; i++){
            double[][] outputs = outputLayer.getSpecificOutput(SoftMaxFunction.getInstance());
            NeuralNetworkMath.printMatrix(outputs);
            double[] losses = calculateLoss(outputs, new int[]{fileIndexValue});
            System.out.println("Losses: " + PureMath.getSumOf(losses));
            loss = losses[0];
            System.out.println("----------\n");
            if(loss > previousLoss) {
                if(firstLayerExecution){
                    inputLayer.adjustWeightAndBias(previousLoss, -learningRate, firstTrainOffset - 1 == -1 ? firstHiddenLayerSize - 1 : firstTrainOffset - 1);
                    inputLayer.trainNeurons();
                    outputLayer.setInputNeurons(inputLayer.getOutputNeurons());
                    outputLayer.trainNeurons();
                }else if(secondLayerExecution){
                    outputLayer.adjustWeightAndBias(previousLoss, -learningRate, secondTrainOffset - 1 == -1 ? secondHiddenLayerSize - 1 : secondTrainOffset - 1);
                    inputLayer.trainNeurons();
                    outputLayer.setInputNeurons(inputLayer.getOutputNeurons());
                    outputLayer.trainNeurons();
                }
            }
            if(firstTrainOffset < firstHiddenLayerSize){
                inputLayer.adjustWeightAndBias(loss, learningRate, firstTrainOffset);
                inputLayer.trainNeurons();
                outputLayer.setInputNeurons(inputLayer.getOutputNeurons());
                outputLayer.trainNeurons();
                firstLayerExecution = true;
                secondLayerExecution = false;
                firstTrainOffset++;
            }else if(secondTrainOffset < secondHiddenLayerSize){
                outputLayer.adjustWeightAndBias(loss, learningRate, secondTrainOffset);
                inputLayer.trainNeurons();
                outputLayer.setInputNeurons(inputLayer.getOutputNeurons());
                outputLayer.trainNeurons();
                firstLayerExecution = false;
                secondLayerExecution = true;
                secondTrainOffset++;
            }else {
                firstTrainOffset = 0;
                secondTrainOffset = 0;
                firstLayerExecution = false;
                secondLayerExecution = false;
            }
            previousLoss = loss;
        }

        System.out.println("Enter filename to enter: ");
        String inputFileName = scanner.nextLine();
        System.out.println("Enter file value: " );
        int fileValue = scanner.nextInt();
        try {
            NumberFile numberFile = CustomNeuralFileHandler.getNumberFileFrom(inputFileName, fileValue);
            inputLayer.setInputNeurons(new BatchInput(numberFile.getMatrixValue()).getBatchNeurons());
            inputLayer.trainNeurons();
            outputLayer.setInputNeurons(inputLayer.getOutputNeurons());
            int[] predictedValues = getMaxIndex(outputLayer.getSpecificOutput(SoftMaxFunction.getInstance()));
            System.out.println("Predicted Number is: " + predictedValues[0]);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static int[] getMaxIndex(double[][] inputValues){
        int rowLength = inputValues.length;
        int columnLength = inputValues[0].length;
        int[] predictedNumber = new int[rowLength];
        for(int row=0; row < rowLength; row++){
            double max = 0;
            for(int column = 0; column < columnLength; column++){
                if(max < inputValues[row][column]) {
                    max = inputValues[row][column];
                    predictedNumber[row] = column;
                }
            }
        }
        return predictedNumber;
    }

    private static double[] calculateLoss(double[][] outputs, int[] category) {
        int outputLength = outputs.length;
        if(outputLength!=category.length)
            throw new IllegalArgumentException("Number of category must be equal to number of inputs!");
        CategoricalCrossEntropy categoricalEntropy = new OneHotEncoding();
        double[] losses = new double[outputLength];
        for(int index=0; index < outputLength; index++){
            losses[index] = categoricalEntropy.getLossValue(outputs[index], category[index]);
        }
        return losses;
    }
}
