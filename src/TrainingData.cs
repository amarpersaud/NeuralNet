using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkBase
{
    /// <summary>
    /// Struct for holding training data (inputs and expected outputs)
    /// </summary>
    public struct TrainingData
    {
        /// <summary>
        /// Create a TrainingData struct for training a ff-nn
        /// </summary>
        /// <param name="input">Inputs</param>
        /// <param name="output">Outputs</param>
        public TrainingData(double[] input, double[] output)
        {
            this.Input = input;
            this.Output = output;
        }

        /// <summary>
        /// Neural network inputs
        /// </summary>
        public double[] Input;
        /// <summary>
        /// Expected outputs
        /// </summary>
        public double[] Output;
    }
}
