package com.neural.network.lib.math;

public class PureMath {
    private PureMath(){}
    public static double getValueOfExponentiationToPower(double x){
        double exponentiation = 2.71828182846;
        return Math.pow(exponentiation, x);
    }

    /**
     * Also known for calculating probability distribution.
     * 2 , 1 -> 2/3, 1/3
     * @param x
     * @param total
     * @return
     */
    public static double getNormalizationValue(double x, double total){
        return x / total;
    }

    public static double getSumOf(double[] xs){
        double sum = 0;
        for(double value: xs){
            sum += value;
        }
        return sum;
    }

    public static double getMaxOf(double[] xs){
        double max = 0;
        for(double value: xs){
            max = Math.max(value, max);
        }
        return max;
    }
}
