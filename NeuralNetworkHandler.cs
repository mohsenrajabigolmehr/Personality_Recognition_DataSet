using Keras.Layers;
using Keras.Models;
using Keras.PreProcessing.sequence;
using Numpy;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PR
{
    public class NeuralNetworkHandler
    {

        public class AllWords
        {
            public static void Run()
            {
                //Create model
                Sequential model = new Sequential();

                //======================================================================================

                /*                 
                    
                True Positive	False Negative	42	9
                False Positive	True Negative	5	0
                Accuracy = TP+TN/TP+FP+FN+TN			0.75			
                Precision = TP/TP+FP			0.8936170213			
                Recall = TP/TP+FN			0.8235294118			
                F1 Score = 2*(Recall * Precision) / (Recall + Precision)			0.8571428571 
                 */

                //model number 1:

                var ClassificationReport = Keras.Callbacks.Callback.Custom("ClassificationReport", "../../ClassificationReport.py");

                string modelName = "model-01";
                int max_features = 17500;
                int maxlen = 250;
                int embedding_dims = 10;

                model.Add(new Embedding(max_features, embedding_dims, input_length: maxlen));
                model.Add(new Conv1D(filters: embedding_dims, kernel_size: 3, padding: "same", activation: "relu"));
                model.Add(new MaxPooling1D(pool_size: 2));
                model.Add(new Flatten());
                model.Add(new Dense(300, activation: "sigmoid"));
                model.Add(new Dense(15, activation: "sigmoid"));
                model.Compile(
                    loss: "binary_crossentropy",
                    optimizer: "adam",
                    metrics: new string[] { "accuracy" }
                );
                model.Summary();

                //======================================================================================

                var ((x_train, y_train), (x_test, y_test)) = GetData(false, 1, false);
                x_train = SequenceUtil.PadSequences(x_train, maxlen: maxlen);
                x_test = SequenceUtil.PadSequences(x_test, maxlen: maxlen);
                model.Fit(
                    x_train, 
                    y_train, 
                    validation_data: new NDarray[] { x_test, y_test }, 
                    epochs: 10, 
                    batch_size: 1, 
                    verbose: 1,
                    callbacks: new Keras.Callbacks.Callback[] { ClassificationReport }
                );

                // Final evaluation of the model
                var scores = model.Evaluate(x_test, y_test, verbose: 1);
                Console.WriteLine("Loss: " + scores[0]);
                Console.WriteLine("Accuracy: " + scores[1]);


                //Save Model
                string json = model.ToJson();
                System.IO.File.WriteAllText($"{modelName}.json", json);
                model.Save($"{modelName}.h5");
                Console.WriteLine("Model Saved.");
            }

            public static void Predict()
            {
                //Load data 
                string modelName = "model-01";
                var allWordsTrains = new List<EFDB.AllWordsTrainResult>();

                var false_numbers = 0;
                var true_numbers = 0;

                using (var db = new EFDB.PersonalityRecognitionEntities())
                {
                    allWordsTrains = db.AllWordsTrainResults.OrderBy(q => q.ID).ToList();

                    var model = Sequential.LoadModel($"{modelName}.h5");

                    for (int i = 0; i < allWordsTrains.Count; i++)
                    {
                        var item = allWordsTrains[i];

                        NDarray x;
                        x = np.array(item.Features.Split(',').Select(q => int.Parse(q)).ToArray());
                        x = x.reshape(1, x.shape[0]);
                        x = SequenceUtil.PadSequences(x, maxlen: 250);
                        var y = model.Predict(x);

                        allWordsTrains[i].Predict = "";
                        var nums = "";
                        for (int j = 0; j < y[0].len; j++)
                        {
                            var t = y[0][j].ToString();
                            var num = (decimal)Convert.ToDouble(t, System.Globalization.CultureInfo.InvariantCulture);
                            nums += $"{num},";

                            if (num < Convert.ToDecimal(0.1))
                                allWordsTrains[i].Predict += "0";
                            else
                                allWordsTrains[i].Predict += "1";

                            //reviews[i].Predict += t;

                            if (j < y[0].len - 1)
                            {
                                allWordsTrains[i].Predict += ",";
                            }
                        }


                        Console.WriteLine($"{i}");
                        if (allWordsTrains[i].Target != allWordsTrains[i].Predict)
                        {
                            false_numbers++;
                            Console.WriteLine($"{allWordsTrains[i].Target}");
                            Console.WriteLine($"{allWordsTrains[i].Predict}");
                            Console.WriteLine($"-------------------------------------------------------");
                        }
                        else
                        {
                            true_numbers++;
                        }

                    }

                    db.SaveChanges();
                }

                Console.WriteLine("Predict.");

                Console.WriteLine($"false = {false_numbers} , true = {true_numbers}");

                Console.WriteLine($"Accuracy = {(true_numbers * 100 / allWordsTrains.Count)}");
            }

            private static ((NDarray, NDarray), (NDarray, NDarray)) GetData(bool random = true, int fold = 1, bool all = false)
            {
                var ItemsForTest = new List<EFDB.AllWordsTrainResult>();
                var ItemForTrain = new List<EFDB.AllWordsTrainResult>();
                int trainCount = 41;
                int testCount = 10;

                using (var db = new EFDB.PersonalityRecognitionEntities())
                {
                    var reviews = db.AllWordsTrainResults.OrderBy(q => q.ID).Skip(0).Take(trainCount + testCount).ToList();

                    if (random)
                    {
                        var rnd = new Random();
                        for (int i = 0; i < trainCount; i++)
                        {
                            var index = rnd.Next(0, trainCount - 1);
                            ItemForTrain.Add(reviews[index]);
                        }

                        for (int i = 0; i < testCount; i++)
                        {
                            var index = rnd.Next(0, testCount - 1);
                            ItemsForTest.Add(reviews[index]);
                        }
                    }
                    else
                    {
                        ItemsForTest = reviews.OrderBy(q => q.ID).Skip((fold - 1) * testCount).Take(testCount).ToList();
                        foreach (var item in reviews)
                        {
                            if (!ItemsForTest.Any(q => q.ID == item.ID))
                            {
                                ItemForTrain.Add(item);
                            }
                        }
                    }

                    if (all)
                    {
                        ItemForTrain = reviews.OrderBy(q => q.ID).Skip(0).Take(500).ToList();
                        ItemsForTest = reviews.OrderBy(q => q.ID).Skip(500).Take(500).ToList();
                    }

                    var x = new List<NDarray>();
                    var y = new List<NDarray>();
                    for (int i = 0; i < ItemForTrain.Count; i++)
                    {
                        var item = ItemForTrain[i];
                        x.Add(np.array(item.Features.Split(',').Select(q => int.Parse(q)).ToArray()));
                        y.Add(np.array(item.Target.Split(',').Select(q => int.Parse(q)).ToArray()));
                    }

                    var x_train = new NDarray(np.array(x));
                    var y_train = new NDarray(np.array(y));


                    x = new List<NDarray>();
                    y = new List<NDarray>();
                    for (int i = 0; i < ItemsForTest.Count; i++)
                    {
                        var item = ItemsForTest[i];
                        x.Add(np.array(item.Features.Split(',').Select(q => int.Parse(q)).ToArray()));
                        y.Add(np.array(item.Target.Split(',').Select(q => int.Parse(q)).ToArray()));
                    }

                    var x_test = new NDarray(np.array(x));
                    var y_test = new NDarray(np.array(y));

                    var X = np.concatenate(new NDarray[] { x_train, x_test }, axis: 0);
                    var Y = np.concatenate(new NDarray[] { y_train, y_test }, axis: 0);

                    Console.WriteLine("Shape of X: " + X.shape);
                    Console.WriteLine("Shape of Y: " + Y.shape);

                    Console.WriteLine("Number of words: ");
                    var hstack = np.hstack(new NDarray[] { X });


                    var res = ((x_train, y_train), (x_test, y_test));

                    return res;

                }

            }

        }
    }
}
