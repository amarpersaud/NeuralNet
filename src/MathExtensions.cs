using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkBase
{
    public static class MathEx
    {
        /// <summary>
        /// Fast absolute value
        /// </summary>
        /// <param name="i">Input</param>
        /// <returns>Absolute value of Input</returns>
        public static double abs(this double i)
        {
            return (i < 0) ? -i : i;
        }

        /// <summary>
        /// Sigmoid of x
        /// </summary>
        /// <param name="x">Input</param>
        /// <returns>Sigmoid of x</returns>
        public static double sigmoid(this double x)
        {
            return 1.0f / (1.0f + Math.Exp(-x));
        }

        /// <summary>
        /// Derivative of sigmoid
        /// </summary>
        /// <param name="x">Input</param>
        /// <returns>Derivative of sigmoid at x</returns>
        public static double sigmoidDerivative(this double x)
        {
            return x * (1.0f - x);
        }
    }
}
