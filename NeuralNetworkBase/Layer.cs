using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkBase
{
    public struct Layer
    {
        /// <summary>
        /// Array of neurons in the layer
        /// </summary>
        public Neuron[] neurons;
        /// <summary>
        /// Output of this layer
        /// </summary>
        public double[] output;

        /// <summary>
        /// Layer of a neural network
        /// </summary>
        /// <param name="l">Deep Clone a Layer</param>
        public Layer(Layer l)
        {
            this.output = new double[l.output.Length];
            for (int i = 0; i < l.output.Length; i++)
            {
                output[i] = l.output[i];
            }
            this.neurons = new Neuron[l.neurons.Length];
            for (int i = 0; i < l.neurons.Length; i++)
            {
                this.neurons[i] = new Neuron(l.neurons[i]);
            }
        }

        public static double[] GetOutput(Layer l, double[] previousLayerOutput)
        {
            double[] result = new double[l.neurons.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = Neuron.GetOutput(l.neurons[i], previousLayerOutput);
            }
            return result;
        }
    }
}
