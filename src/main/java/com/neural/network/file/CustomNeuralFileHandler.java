package com.neural.network.file;

import java.io.*;
import java.util.LinkedList;
import java.util.List;

public class CustomNeuralFileHandler {
    private CustomNeuralFileHandler(){}

    public static NumberFile getNumberFileFrom(String filepath, int numberValue) throws IOException {
        try(BufferedReader fileReader = new BufferedReader(new FileReader(getFileFromResource(filepath)))){
            List<String> patterns = new LinkedList<>();
            String patternLine = "";
            while((patternLine = fileReader.readLine()) != null){
                patterns.add(patternLine);
            }
            double[][] matrix = new double[1][3 * 15];
            loadMatrixFromPatterns(patterns, matrix);
            return new NumberFile(numberValue, matrix);
        }
    }

    private static void loadMatrixFromPatterns(List<String> patterns, double[][] matrix) {
        int patternLength = patterns.size();;
        for(int i=0; i<patternLength; i++){
            String[] chars = patterns.get(i).split(" ");
            int charsLength = chars.length;
            for(int j=0; j<charsLength; j++){
                matrix[0][i * j + j] = chars[j].equals("*") ? 1 : 0;
            }
        }
    }

    private static String getFileFromResource(String filename){
        StringBuilder pathBuilder = new StringBuilder();
        pathBuilder.append(System.getProperty("user.dir")).append(File.separator)
                .append("src").append(File.separator)
                .append("main").append(File.separator)
                .append("java").append(File.separator)
                .append("resources").append(File.separator)
                .append(filename);
        return pathBuilder.toString();
    }
}
