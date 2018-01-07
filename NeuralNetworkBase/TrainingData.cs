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
        /// Neural network inputs
        /// </summary>
        public double[] input;
        /// <summary>
        /// Expected outputs
        /// </summary>
        public double[] output;
    }
}
