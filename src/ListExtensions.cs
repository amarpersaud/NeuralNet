using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkBase
{
    public static class ListExtensions
    {
        public static Random r = new Random(DateTime.Now.Millisecond * DateTime.Now.Second);
        /// <summary>
        /// Shuffle a list
        /// </summary>
        /// <typeparam name="T">Type</typeparam>
        /// <param name="list">List to shuffle</param>
        /// <returns></returns>
        public static IList<T> Shuffle<T>(this IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = r.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
            return list;
        }
    }
}
