/*
    Copyright (c) Amar Persaud 2018
    A feed-forward back-propogation neural network
 */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkBase
{
    public class NeuralNetwork
    {
        public static Random r = new Random(DateTime.Now.Millisecond * DateTime.Now.Second);
        public Layer[] layers;
        public double lrate = 50f;
        public int lastLayer;
        public int lastHiddenLayer;
        public NeuralNetwork(int[] neuronCount, int inputs)
        {
            layers = new Layer[neuronCount.Length];

            lastLayer = layers.Length - 1;
            lastHiddenLayer = layers.Length - 2;

            for (int j = 0; j < layers.Length; j++)
            {
                layers[j] = new Layer();
                layers[j].neurons = new Neuron[neuronCount[j]];
                for (int k = 0; k < layers[j].neurons.Length; k++)
                {
                    layers[j].neurons[k] = new Neuron();
                    layers[j].neurons[k].bias = (2.0 * r.NextDouble()) - 1.0f;
                    layers[j].neurons[k].weights = new double[j != 0 ? neuronCount[j - 1] : inputs];
                    for (int l = 0; l < layers[j].neurons[k].weights.Length; l++)
                    {
                        layers[j].neurons[k].weights[l] = r.Next(-1, 2);
                    }
                }
            }
        }

        public NeuralNetwork(NeuralNetwork n)
        {
            this.layers = new Layer[n.layers.Length];
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Layer(n.layers[i]);
            }
            this.lastHiddenLayer = n.lastHiddenLayer;
            this.lastLayer = n.lastLayer;
            this.lrate = n.lrate;
        }

        public NeuralNetwork()
        {

        }

        public double getAverageError(TrainingData[] td)
        {
            double terr = 0;

            for (int j = 0; j < td.Length; j++)
            {
                terr += getTotalError(td[j]);
            }
            return terr;
        }

        public void train(TrainingData td)
        {
            //Calculate output
            double[] previousLayerOutput = td.input;
            for (int i = 0; i < layers.Length; i++)
            {
                previousLayerOutput = Layer.GetOutput(layers[i], previousLayerOutput);
                layers[i].output = previousLayerOutput;
            }
            //Calculate deltas
            for (int j = 0; j < layers[lastLayer].neurons.Length; j++)
            {
                layers[lastLayer].neurons[j].delta = getOutputNeuronDelta(layers[lastLayer].output[j], td.output[j]);
            }
            for (int i = lastHiddenLayer; i >= 0; i--)
            {
                for (int j = 0; j < layers[i].neurons.Length; j++)
                {
                    layers[i].neurons[j].delta = getHiddenNeuronDelta(j, layers[i].output[j], layers[i + 1]);
                }
            }
            //Adjust weights
            for (int i = lastLayer; i > 0; i--)
            {
                Layer l = layers[i];
                for (int j = 0; j < l.neurons.Length; j++)
                {
                    double lrateDt = lrate * l.neurons[j].delta;
                    layers[i].neurons[j].bias += lrateDt;
                    for (int k = 0; k < l.neurons[j].weights.Length; k++)
                    {
                        layers[i].neurons[j].weights[k] += lrateDt * layers[i - 1].output[k];
                    }
                }
            }
            for (int j = 0; j < layers[0].neurons.Length; j++)
            {
                double lrateDt = lrate * layers[0].neurons[j].delta;
                layers[0].neurons[j].bias += lrateDt;
                for (int k = 0; k < layers[0].neurons[j].weights.Length; k++)
                {
                    layers[0].neurons[j].weights[k] += lrateDt * td.input[k];
                }
            }
        }




        /// <summary>
        /// Calculate output of the network for a given input
        /// </summary>
        /// <param name="td">Training data object to get input from</param>
        /// <returns>Output of the Neural Network</returns>
        public double[] calculateOutput(TrainingData td)
        {
            return calculateOutput(td.input);
        }

        /// <summary>
        /// Calculate output of the network for a given input
        /// </summary>
        /// <param name="input">Array of double inputs</param>
        /// <returns>Output of the Neural Network</returns>
        public double[] calculateOutput(double[] input)
        {
            double[] previousLayerOutput = input;
            for (int i = 0; i < layers.Length; i++)
            {
                previousLayerOutput = Layer.GetOutput(layers[i], previousLayerOutput);
                layers[i].output = previousLayerOutput;
            }
            return previousLayerOutput;
        }

        public double getOutputNeuronDelta(double actualOutput, double expected)
        {
            return actualOutput * (1.0f - actualOutput) * (expected - actualOutput); //expected - actual is the errorFactor
        }
        public double getHiddenNeuronDelta(int neuron, double output, Layer nextLayer)
        {

            double errorFactor = 0.0f;
            for (int i = 0; i < nextLayer.neurons.Length; i++)
            {
                errorFactor += nextLayer.neurons[i].delta * nextLayer.neurons[i].weights[neuron];
            }
            return output * (1.0f - output) * errorFactor;
        }

        public double getTotalError(TrainingData td)
        {
            double error = 0;
            //get network output, with td's input
            calculateOutput(td);
            double[] output = layers.Last().output;
            for (int k = 0; k < output.Length; k++)
            {
                double diff = output[k] - td.output[k];
                error += 0.5f * (diff * diff);
            }
            return error;
        }

    }
}
