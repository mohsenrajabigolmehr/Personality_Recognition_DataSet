using libsvm;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PR
{
    public class SVMHandler
    {
        public class AllWords
        {
            public static void Run()
            {
                using (var db = new EFDB.PersonalityRecognitionEntities())
                {
                    var allWordsTrains = db.AllWordsTrainResults.OrderBy(q => q.ID).ToList();
                    var problem = CreateProblem(allWordsTrains);

                    int C = 1;
                    int degree = 2;
                    double gamma = 1;
                    int r = 1;

                    var LinearKernel = KernelHelper.LinearKernel();
                    var PolynomialKernel = KernelHelper.PolynomialKernel(degree, gamma, r);
                    var SigmoidKernel = KernelHelper.SigmoidKernel(gamma, r);
                    var RadialBasisFunctionKernel = KernelHelper.RadialBasisFunctionKernel(gamma);

                    //var model = new C_SVC(problem, LinearKernel, C); // 
                    //var model = new C_SVC(problem, PolynomialKernel, C);
                    //var model = new C_SVC(problem, SigmoidKernel, C);
                    var model = new C_SVC(problem, RadialBasisFunctionKernel, C);

                    var accuracy = model.GetCrossValidationAccuracy(5);

                    Console.WriteLine("Accuracy = {0:P}", accuracy);
                    Console.WriteLine("================================");


                    var valid = 0;
                    var invalid = 0;
                    foreach (var item in allWordsTrains)
                    {
                        var node = GetNode(item);
                        var predict = model.Predict(node.nodes.ToArray());
                        if(predict != node.target)
                        {
                            Console.WriteLine("target = {0} , predict = {1} *", node.target, predict);
                            invalid++;
                        }
                        else
                        {
                            Console.WriteLine("target = {0} , predict = {1}", node.target, predict);
                            valid++;
                        }
                    }


                    Console.WriteLine("valid = {0} , invalid = {1}", valid , invalid);

                }
            }


            private static svm_problem CreateProblem(List<EFDB.AllWordsTrainResult> allWordsTrainResults)
            {
                var list = new List<Problem>();
                foreach (var item in allWordsTrainResults)
                {
                    list.Add(GetNode(item));
                }

                return new svm_problem
                {
                    l = allWordsTrainResults.Count,
                    x = list.Select(q => q.nodes.ToArray()).ToArray(),
                    y = list.Select(q => q.target).ToArray()
                };
            }


            private static Problem GetNode(EFDB.AllWordsTrainResult item)
            {
                var target_bits = item.Target.Split(',').Select(q => int.Parse(q)).ToList();

                int index = 1;
                double sum = 0;
                foreach (var l in target_bits)
                {
                    sum += l * index;
                    index++;
                }

                var nodes = new List<svm_node>();

                index = 0;
                foreach (var word in item.Features.Split(',').Select(q => int.Parse(q)))
                {
                    nodes.Add(new svm_node()
                    {
                        index = index,
                        value = word
                    });
                    index++;
                }

                return new Problem()
                {
                    target = sum,
                    nodes = nodes
                };
            }



            private class Problem
            {
                public double target { get; set; }
                public List<svm_node> nodes { get; set; }
            }


        }
    }



}
