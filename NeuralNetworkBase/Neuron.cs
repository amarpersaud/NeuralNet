using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkBase
{
    public struct Neuron
    {
        /// <summary>
        /// Weights for each input
        /// </summary>
        public double[] weights;
        /// <summary>
        /// Bias value added to weighted sum
        /// </summary>
        public double bias;
        /// <summary>
        /// Delta for back-propogation
        /// </summary>
        public double delta;

        /// <summary>
        /// Neuron for Neural Network
        /// </summary>
        /// <param name="n">Neuron to deep clone</param>
        public Neuron(Neuron n)
        {
            this.bias = n.bias;
            this.delta = n.delta;
            this.weights = new double[n.weights.Length];
            for (int i = 0; i < n.weights.Length; i++)
            {
                this.weights[i] = n.weights[i];
            }
        }
        /// <summary>
        /// Get the output of an individual neuron from given inputs
        /// </summary>
        /// <param name="n">Neuron to test</param>
        /// <param name="inputs">Inputs to neuron</param>
        /// <returns>Output of the Neuron</returns>
        public static double GetOutput(Neuron n, double[] inputs){
            double result = n.bias;
            for (int i = 0; i < n.weights.Length; i++)
            {
                result += n.weights[i] * inputs[i];
            }
            return MathEx.sigmoid(result);
        }
    }
}
