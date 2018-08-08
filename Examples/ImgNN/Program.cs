/*
    Copyright (c) Amar Persaud 2018
    A program to train a neural network to reconize 2x2 images
 */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.IO;
using NeuralNetworkBase;
using System.Diagnostics;

namespace ImgNN
{
    class Program
    {
        public const int Iterations = 1_000_000;
        
        public static string[] OutputNames = new string[] {

            "Vertical left",
            "Vertical right",
            "Horizontal Top",
            "Horizontal Bottom",
            "Diagonal /",
            "Diagonal \\",
            "Dot"
        };


        static void Main(string[] args)
        {
            Console.WriteLine("Copyright (c) Amar Persaud 2018");
            NeuralNetwork n = new NeuralNetwork(new int[] { 7 }, 4, 7);
            TrainingData[] t = new TrainingData[]
            {
                new TrainingData {Input= new double[]{1, 0, 0, 1}, Output = new double[] {0, 0, 0, 0, 0, 1, 0 } }, // Diagonal \
                new TrainingData {Input= new double[]{0, 1, 1, 0}, Output = new double[] {0, 0, 0, 0, 1, 0, 0 } }, // Diagonal /
                new TrainingData {Input= new double[]{0, 0, 1, 1}, Output = new double[] {0, 0, 0, 1, 0, 0, 0 } }, // Horizontal bottom
                new TrainingData {Input= new double[]{1, 1, 0, 0}, Output = new double[] {0, 0, 1, 0, 0, 0, 0 } }, // horizontal top
                new TrainingData {Input= new double[]{0, 1, 0, 1}, Output = new double[] {0, 1, 0, 0, 0, 0, 0 } }, // Vertical right
                new TrainingData {Input= new double[]{1, 0, 1, 0}, Output = new double[] {1, 0, 0, 0, 0, 0, 0 } }, // Vertical left
                                                                   
                new TrainingData {Input= new double[]{1, 0, 0, 0}, Output = new double[] {0, 0, 0, 0, 0, 0, 1 } }, // Dot
                new TrainingData {Input= new double[]{0, 1, 0, 0}, Output = new double[] {0, 0, 0, 0, 0, 0, 1 } }, // Dot
                new TrainingData {Input= new double[]{0, 0, 1, 0}, Output = new double[] {0, 0, 0, 0, 0, 0, 1 } }, // Dot
                new TrainingData {Input= new double[]{0, 0, 0, 1}, Output = new double[] {0, 0, 0, 0, 0, 0, 1 } }  // Dot
            };

            
            Stopwatch stop = new Stopwatch();
            stop.Start();
            Console.WriteLine("Starting");
            for (int i = 0; i < Iterations; i++)
            {
                for (int j = 0; j < t.Length; j++)
                {
                    n.Train(t[j]);
                }
            }
            stop.Stop();
            Console.WriteLine($"Done. Took {stop.Elapsed.TotalSeconds} seconds, {stop.Elapsed.TotalSeconds / (Iterations * t.Length)} per iteration");
            
            Console.WriteLine("Error: " + n.GetAverageError(t));

            Console.WriteLine("Type help for more information");
            while (true)
            {
                string s = Console.ReadLine().ToLower();
                string[] splitText = s.Split(' ');
                switch (splitText[0])
                {
                    case "exit":
                        Environment.Exit(0);
                        break;
                    case "test":
                        if(splitText.Length == 2) {
                            TestNetwork(n, Environment.CurrentDirectory + "\\img\\" + splitText[1] + ".png");
                        }
                        else
                        {
                            Console.WriteLine("Invalid number of arguments");
                        }
                        break;
                    case "help":
                    case "?":
                    case "/?":
                    case "h":
                    case "/h":
                        Console.WriteLine(@"Commands:
    Exit            Exits Program
    Test <number>   Tests image against network (1-10)
    Help            View this prompt");
                        break;
                    default:
                        Console.WriteLine("Unrecognized Command");
                        break;
                }
            }
        }

        public static void TestNetwork(NeuralNetwork nn, string path)
        {
            if (File.Exists(path))
            {
                using (Bitmap b = new Bitmap(Bitmap.FromFile(path)))
                {
                    if (b.Width == 2 && b.Height == 2)
                    {
                        Console.WriteLine();
                        double[] input = GetArrayFromBitmap(b);

                        double[] result = nn.CalculateOutput(input);
                        for (int i = 0; i < result.Length; i++)
                        {
                            result[i] = Math.Round(result[i]);
                        }

                        for(int y = 0; y < 2; y++)
                        {
                            for (int x = 0; x < 2; x++)
                            {
                                Console.Write((input[(y * 2) + x] == 0 ? "░░" : "▓▓"));
                            }
                            Console.WriteLine();
                        }
                        string o = "Input: ";
                        for (int i = 0; i < input.Length; i++)
                        {
                            o += input[i] + " ";
                        }
                        Console.WriteLine("\n" + o + "\n");
                        o = "Output: ";
                        for (int i = 0; i < result.Length; i++)
                        {
                            o += result[i] + " ";
                        }
                        Console.WriteLine(o + "\n");
                        int index = Array.IndexOf(result, 1);
                        if (index != -1)
                        {
                            Console.WriteLine($"Image is: {OutputNames[index]} \n");
                        }
                    }
                    else {
                        Console.WriteLine($"Dimension mistmatch. Expected 2 x 2. Got {b.Width} x {b.Height}");
                    }
                }
            }
            else
            {
                Console.WriteLine($"File Not Found: {path}");
            }
        }

        /// <summary>
        /// Convert a bitmap image into an array of doubles for use by the network
        /// </summary>
        /// <param name="b">The bitmap file</param>
        /// <returns>A double representation of the image (black and white)</returns>
        public static double[] GetArrayFromBitmap(Bitmap b)
        {
            double[] result = new double[b.Width * b.Height * 3];
            for (int y = 0; y < b.Height; y++)
            {
                for (int x = 0; x < b.Width; x++)
                {
                    result[(y * b.Width) + x] = GetGrayscaleValue(b.GetPixel(x, y)) > 128 ? 0 : 1;
                }
            }
            return result;
        }

        public static double GetGrayscaleValue(Color c)
        {
            return (int)((0.3 * c.R) + (0.59 * c.G) + (0.11 * c.B));
        }
    }


}
