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
        /// Weights for each Input
        /// </summary>
        public double[] Weights;
        /// <summary>
        /// Bias value added to weighted sum
        /// </summary>
        public double Bias;
        /// <summary>
        /// Delta for back-propogation
        /// </summary>
        public double Delta;

        /// <summary>
        /// Neuron for Neural Network
        /// </summary>
        /// <param name="n">Neuron to deep clone</param>
        public Neuron(Neuron n)
        {
            this.Bias = n.Bias;
            this.Delta = n.Delta;
            this.Weights = new double[n.Weights.Length];
            for (int i = 0; i < n.Weights.Length; i++)
            {
                this.Weights[i] = n.Weights[i];
            }
        }
        /// <summary>
        /// Get the Output of an individual neuron from given inputs
        /// </summary>
        /// <param name="n">Neuron to test</param>
        /// <param name="inputs">Inputs to neuron</param>
        /// <returns>Output of the Neuron</returns>
        public double GetOutput(double[] inputs){
            double result = Bias;
            for (int i = 0; i < Weights.Length; i++)
            {
                result += Weights[i] * inputs[i];
            }
            return MathEx.sigmoid(result);
        }
    }
}
