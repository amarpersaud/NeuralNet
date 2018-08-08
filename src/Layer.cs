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
        public Neuron[] Neurons;
        /// <summary>
        /// Output of this layer
        /// </summary>
        public double[] Output;

        /// <summary>
        /// Layer of a neural network
        /// </summary>
        /// <param name="l">Deep Clone a Layer</param>
        public Layer(Layer l)
        {
            this.Output = new double[l.Output.Length];
            for (int i = 0; i < l.Output.Length; i++)
            {
                Output[i] = l.Output[i];
            }
            this.Neurons = new Neuron[l.Neurons.Length];
            for (int i = 0; i < l.Neurons.Length; i++)
            {
                this.Neurons[i] = new Neuron(l.Neurons[i]);
            }
            Output = new double[Neurons.Length];
        }
        
        public double[] GetOutput(double[] previousLayerOutput)
        {
            for (int i = 0; i < Output.Length; i++)
            {
                Output[i] = Neurons[i].GetOutput(previousLayerOutput);
            }
            return (double[])Output.Clone();
        }
    }
}
