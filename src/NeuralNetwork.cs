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
        public Layer[] Layers;
        public double LRate = 50f;
        public int LastLayer;
        public int LastHiddenLayer;

        /// <summary>
        /// Creates a neural network.
        /// A neuronCount of {2, 2}, 3 inputs, and 1 Output would have the following layout:
        /// NN > NN > N.
        /// </summary>
        /// <param name="neuronCount">An array of the number of hidden neurons in each layer</param>
        /// <param name="inputs">The number of inputs. There are no Input neurons.</param>
        /// <param name="outputs">The number of outputs in the final Output layer</param>
        public NeuralNetwork(int[] neuronCount, int inputs, int outputs)
        {
            Layers = new Layer[neuronCount.Length + 1];

            LastLayer = Layers.Length - 1;
            LastHiddenLayer = Layers.Length - 2;

            for (int j = 0; j < Layers.Length; j++)
            {
                Layers[j] = new Layer();
                if (j == Layers.Length - 1)
                {
                    Layers[j].Neurons = new Neuron[outputs];
                }
                else
                {
                    Layers[j].Neurons = new Neuron[neuronCount[j]];
                }
                for (int k = 0; k < Layers[j].Neurons.Length; k++)
                {
                    Layers[j].Neurons[k] = new Neuron();
                    Layers[j].Neurons[k].Bias = (2.0 * r.NextDouble()) - 1.0f;
                    if(j == 0)
                    {
                        Layers[j].Neurons[k].Weights = new double[inputs];
                    }
                    else
                    {
                        Layers[j].Neurons[k].Weights = new double[neuronCount[j - 1]];
                    }
                    for (int l = 0; l < Layers[j].Neurons[k].Weights.Length; l++)
                    {
                        Layers[j].Neurons[k].Weights[l] = r.Next(-1, 2);
                    }
                }
            }
        }

        public NeuralNetwork(NeuralNetwork n)
        {
            this.Layers = new Layer[n.Layers.Length];
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i] = new Layer(n.Layers[i]);
            }
            this.LastHiddenLayer = n.LastHiddenLayer;
            this.LastLayer = n.LastLayer;
            this.LRate = n.LRate;
        }

        public NeuralNetwork()
        {

        }

        public double GetAverageError(TrainingData[] td)
        {
            double terr = 0;

            for (int j = 0; j < td.Length; j++)
            {
                terr += GetTotalError(td[j]);
            }
            return terr / td.Length;
        }

        public void Train(TrainingData td)
        {
            //Calculate Output to fill layer outputs
            CalculateOutput(td);

            //Calculate deltas
            for (int j = 0; j < Layers[LastLayer].Neurons.Length; j++)
            {
                Layers[LastLayer].Neurons[j].Delta = GetOutputNeuronDelta(Layers[LastLayer].Output[j], td.Output[j]);
            }
            for (int i = LastHiddenLayer; i >= 0; i--)
            {
                for (int j = 0; j < Layers[i].Neurons.Length; j++)
                {
                    Layers[i].Neurons[j].Delta = GetHiddenNeuronDelta(j, Layers[i].Output[j], Layers[i + 1]);
                }
            }
            //Adjust Weights
            for (int i = LastLayer; i > 0; i--)
            {
                //Output and middle layers use the previous layer's Output
                Layer l = Layers[i];
                for (int j = 0; j < l.Neurons.Length; j++)
                {
                    double lrateDt = LRate * l.Neurons[j].Delta;
                    Layers[i].Neurons[j].Bias += lrateDt;
                    for (int k = 0; k < l.Neurons[j].Weights.Length; k++)
                    {
                        Layers[i].Neurons[j].Weights[k] += lrateDt * Layers[i - 1].Output[k];
                    }
                }
            }
            //The first layer uses the Input from the training data, so do that separately
            //to avoid an excess if in the inner loops that would slow things down
            for (int j = 0; j < Layers[0].Neurons.Length; j++)
            {
                double lrateDt = LRate * Layers[0].Neurons[j].Delta;
                Layers[0].Neurons[j].Bias += lrateDt;
                for (int k = 0; k < Layers[0].Neurons[j].Weights.Length; k++)
                {
                    Layers[0].Neurons[j].Weights[k] += lrateDt * td.Input[k];
                }
            }
        }




        /// <summary>
        /// Calculate Output of the network for a given Input
        /// </summary>
        /// <param name="td">Training data object to get Input from</param>
        /// <returns>Output of the Neural Network</returns>
        public double[] CalculateOutput(TrainingData td)
        {
            return CalculateOutput(td.Input);
        }

        /// <summary>
        /// Calculate Output of the network for a given Input
        /// </summary>
        /// <param name="input">Array of double inputs</param>
        /// <returns>Output of the Neural Network</returns>
        public double[] CalculateOutput(double[] input)
        {
            double[] previousLayerOutput = input;
            for (int i = 0; i < Layers.Length; i++)
            {
                previousLayerOutput = Layers[i].GetOutput(previousLayerOutput);
                Layers[i].Output = previousLayerOutput;
            }
            return previousLayerOutput;
        }

        /// <summary>
        /// Get the amount of change necessary for a Output neuron
        /// </summary>
        /// <param name="actualOutput">Actual Output of the neuron</param>
        /// <param name="expected">Expected Output of the neuron</param>
        /// <returns>Output Neuron Delta</returns>
        private double GetOutputNeuronDelta(double actualOutput, double expected)
        {
            return actualOutput * (1.0f - actualOutput) * (expected - actualOutput); //expected - actual is the errorFactor
        }

        /// <summary>
        /// Gets the Delta (change) to apply to the previous layer so that its Output matches the expected of the next layer. 
        /// </summary>
        /// <param name="neuron">Which neuron in the current layer we are calculating the Delta for</param>
        /// <param name="output">The Output of that neuron, </param>
        /// <param name="nextLayer">The next layer of the network</param>
        /// <returns>Hidden Neuron Delta</returns>
        private double GetHiddenNeuronDelta(int neuron, double output, Layer nextLayer)
        {

            double errorFactor = 0.0f;
            for (int i = 0; i < nextLayer.Neurons.Length; i++)
            {
                errorFactor += nextLayer.Neurons[i].Delta * nextLayer.Neurons[i].Weights[neuron];
            }
            return output * (1.0f - output) * errorFactor;
        }

        /// <summary>
        /// Get the error of the network for a single Input
        /// </summary>
        /// <param name="td"></param>
        /// <returns></returns>
        public double GetTotalError(TrainingData td)
        {
            double error = 0;
            //get network Output
            double[] output = CalculateOutput(td);
            for (int k = 0; k < output.Length; k++)
            {
                    double diff = output[k] - td.Output[k];
                    error += 0.5 * (diff * diff);
            }
            return error / output.Length;
        }

        /// <summary>
        /// Get the layout of the network as a string for
        /// </summary>
        /// <returns></returns>
        public string GetLayout()
        {
            string str = "";
            for(int i = 0; i < Layers.Length; i++)
            {
                str += Layers[i].Neurons.Length + ",";
            }
            return str;
        }
        /// <summary>
        /// Returns true if both Neural Networks have the same layout
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool HaveSameLayout(NeuralNetwork a, NeuralNetwork b)
        {
            return a.GetLayout() == b.GetLayout();
        }

    }
}
