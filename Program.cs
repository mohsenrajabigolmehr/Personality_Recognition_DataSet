using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PR
{
    class Program
    {
        static void Main(string[] args)
        {
            var pythonPath = @"C:\Program Files\Python36";
            pythonPath = @"C:\Users\H-Hamid-R\AppData\Local\Programs\Python\Python36";
            Environment.SetEnvironmentVariable("PATH", $@"{pythonPath};" + Environment.GetEnvironmentVariable("PATH"));
            Environment.SetEnvironmentVariable("PYTHONHOME", pythonPath);
            Environment.SetEnvironmentVariable("PYTHONPATH ", $@"{pythonPath}\Lib");
            using (Py.GIL())
            {
                //dynamic np = Py.Import("numpy");
                //Console.WriteLine(np.cos(np.pi * 2));
                //dynamic yaml = Py.Import("yaml");
            }

            //var dpp = new DataPreProcess();
            //dpp.CreateData();
            //dpp.CreateDataAllWords();



           NeuralNetworkHandler.AllWords.Run(); // ROC     
           //NeuralNetworkHandler.AllWords.Predict();


            //SVMHandler.AllWords.Run();


            Console.ReadKey();
        }
    }
}


