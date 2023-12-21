package com.neural.network.lib;

public class Neuron {
    private double input;

    public Neuron(){

    }

    public Neuron(double input){
        this.input = input;
    }

    public double getInput() {
        return input;
    }

    public void setInput(double input) {
        this.input = input;
    }
}
