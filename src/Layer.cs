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
        /// <param name="Neurons">Array of the neurons in the network</param>
        public Layer(Neuron[] Neurons)
        {
            this.Neurons = (Neuron[])Neurons.Clone();
            this.Output = new double[Neurons.Length];
        }

        /// <summary>
        /// Deep Clone a Layer of a neural network. 
        /// </summary>
        /// <param name="l">Layer to clone</param>
        public Layer(Layer l)
        {
            //Shallow copy since content are doubles
            this.Output = (double[])l.Output.Clone();
            
            //Deep copy
            this.Neurons = new Neuron[l.Neurons.Length];
            
            for (int i = 0; i < l.Neurons.Length; i++)
            {
                this.Neurons[i] = new Neuron(l.Neurons[i]);
            }
        }

        /// <summary>
        /// Get the output of the layer
        /// </summary>
        /// <param name="input">The input to the layer, usually the output from the previous layer</param>
        /// <returns>nd array of</returns>
        public double[] GetOutput(double[] input)
        {
            for(int i = 0; i < Neurons.Length; i++)
            {
                Output[i] = Neurons[i].GetOutput(input);
            }
            return Output;
        }
    }
}
