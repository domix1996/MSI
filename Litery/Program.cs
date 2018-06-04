using System;
using System.IO;
using System.Linq;
using FANNCSharp;
#if FANN_FIXED
using FANNCSharp.Fixed;
using DataType = System.Int32;
#elif FANN_DOUBLE
using FANNCSharp.Double;
using DataType = System.Double;
#else
using FANNCSharp.Float;
using DataType = System.Single;
#endif

namespace TrainNumber
{
    class TrainNumbers
    {
        private const string Path = @"C:\Users\Dominik Narożyński\Desktop\Litery\Litery\numbers_test.txt";
        private const string Filename = @"C:\Users\Dominik Narożyński\Desktop\Litery\Litery\numbers_learn.data";
        private const string SaveFile = @"C:\Users\Dominik Narożyński\Desktop\Litery\Litery\numbers_net.net";

        private static int PrintAll(
            NeuralNet neural, 
            TrainingData training, 
            uint max_epochs, 
            uint epochs_between_reports,
            float desired_error, 
            uint epochs,
            Object userData
        )
        {
            Console.WriteLine(String.Format("Epochs     " + String.Format("{0:D}", epochs).PadLeft(8) + ". Current Error: " + neural.MSE));
            return 0;
        }

        private static void NumbersTest()
        {
            Console.WriteLine("\nNumbers test started.");

            const float learning_rate = 0.7f;
            const uint num_layers = 6;
            const uint num_input = 100;
            const uint num_hidden1 = 80;
            const uint num_hidden2 = 60;
            const uint num_hidden3 = 40;
            const uint num_hidden4 = 10;
            const uint num_output = 6;
            const float desired_error = 0.0002f;
            const uint max_iterations = 1000000;
            const uint iterations_between_reports = 10000;

            Console.WriteLine("\nCreating network.");

            using (NeuralNet network = new NeuralNet(NetworkType.LAYER, num_layers, num_input, num_hidden1, num_hidden2, num_hidden3, num_hidden4, num_output))
            {
                network.LearningRate = learning_rate;

                network.ActivationSteepnessHidden = 1.0F;
                network.ActivationSteepnessOutput = 1.0F;

                network.ActivationFunctionHidden = ActivationFunction.SIGMOID_SYMMETRIC_STEPWISE;
                network.ActivationFunctionOutput = ActivationFunction.SIGMOID_STEPWISE;

                Console.Write("\nNetworkType\t\t\t:  ");
                switch (network.NetworkType)
                {
                    case NetworkType.LAYER:
                        Console.WriteLine("LAYER");
                        break;
                    case NetworkType.SHORTCUT:
                        Console.WriteLine("SHORTCUT");
                        break;
                    default:
                        Console.WriteLine("UNKNOWN");
                        break;
                }

                network.PrintParameters();

                Console.WriteLine("\nTraining network.");

                using (TrainingData data = new TrainingData())
                {
                    if (data.ReadTrainFromFile(Filename))
                    {
                        network.InitWeights(data);

                        Console.WriteLine("Max Epochs " + max_iterations + ". Desired Error: " + desired_error);
                        network.SetCallback(PrintAll, null);
                        network.TrainOnData(data, max_iterations, iterations_between_reports, desired_error);

                        Console.WriteLine("\nTesting network.");
                        Console.WriteLine("\nSaving network.");

                        network.Save(SaveFile);
                        Console.WriteLine("\n numbers test completed.");
                        for (int i = 0; i < 100; i++)
                        {
                            Console.Write("=");
                        }
                        Console.WriteLine("\n");
                        CalculateOutputs(network);
                    }
                }
            }
        }

        public static void CalculateOutputs(NeuralNet net)
        {
            using (TextReader reader = File.OpenText(Path))
            {
                int converter = Array.ConvertAll(reader.ReadLine().Split(' '), Int32.Parse).First();

                for (int i = 0; i < converter; i++)
                {
                    var inputsSymbol = reader.ReadLine().Split(' ');
                    var outputsSymbol = reader.ReadLine().Split(' ');
                    var inputs = new float[100];
                    var outputs = new float[6];

                    for (int j = 0; j < inputsSymbol.Length; j++)
                    {
                        inputs[j] = Convert.ToSingle(inputsSymbol[j]);
                    }
                    for (int j = 0; j < outputsSymbol.Length; j++)
                    {
                        outputs[j] = Convert.ToSingle(outputsSymbol[j]);
                    }

                    float[] calc_out = net.Run(inputs);

                    for (int j = 0; j < 10; j++)
                    {
                        for (int k = 0; k < 10; k++)
                        {
                            Console.Write(inputs[10 * j + k] == 0 ? ' ' : '█');
                        }
                        Console.WriteLine();
                    }

                    Console.WriteLine(""+$"{outputs[0]} => {calc_out[0]} \t {outputs[1]} =>{calc_out[1]} \t{outputs[2]} => {calc_out[2]}");
                    Console.WriteLine("------------------------------------------------");
                }
            }
        }

        static int Main(string[] args)
        {
            NumbersTest();

            Console.ReadKey();
            return 0;
        }

        private static DataType FannAbs(DataType value) => (((value) > 0) ? (value) : -(value));
    }
}


