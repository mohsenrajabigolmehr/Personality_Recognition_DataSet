using NHazm;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PR
{
    public class DataPreProcess
    {

        private POSTagger __Tagger;
        public Lemmatizer __Lemmatizer;

        public DataPreProcess()
        {
            __Tagger = new POSTagger();
            __Lemmatizer = new Lemmatizer();
        }

        public void CreateData()
        {
            using (var db = new EFDB.PersonalityRecognitionEntities())
            {
                var insta_posts = db.instas.Where(q => q.code != null).ToList();
                var i = 0;
                foreach (var post in insta_posts)
                {
                    i++;

                    try
                    {
                        var NormalizedText = Normalizer(post.text, out int Normalized);
                        var Sentences = SentenceTokenizer(NormalizedText, out int SentencesCount, out List<string> lstSentences);
                        var words = WordTokenizer(lstSentences, out string POSTags, out string Lems);
                        post.NormalWords = words;
                    }
                    catch (Exception ex)
                    {
                        post.NormalWords = "";

                    }

                    Console.WriteLine($"{i}");
                }


                db.SaveChanges();

                Console.WriteLine($"SaveChanges");

                db.Database.ExecuteSqlCommand("EXEC dbo.CreateWordsByNormalWords");

                Console.WriteLine($"CreateWordsByNormalWords");
            }
        }

        private string Normalizer(string Input, out int Normalized)
        {
            Normalizer normalizer = new Normalizer(true, true, true);
            var output = normalizer.Run(Input);
            Normalized = Input == output ? 0 : 1;
            return output;
        }
        private string SentenceTokenizer(string Input, out int Count, out List<string> Sentences)
        {
            SentenceTokenizer senTokenizer = new SentenceTokenizer();
            var list = senTokenizer.Tokenize(Input);
            Sentences = list;
            Count = list.Count;
            return string.Join(",", list);
        }
        private string WordTokenizer(List<string> Sentences, out string POSTags, out string Lems)
        {
            WordTokenizer wordTokenizer = new WordTokenizer(true);
            Stemmer stemmer = new Stemmer();

            var OutPut = "";
            POSTags = "";
            Lems = "";
            for (int j = 0; j < Sentences.Count; j++)
            {
                var words = wordTokenizer.Tokenize(Sentences[j]);
                for (int i = 0; i < words.Count; i++)
                {
                    try
                    {
                        words[i] = stemmer.Stem(words[i]);
                    }
                    catch (Exception ex)
                    {
                    }
                }

                //todo fix packages
                //var pos = __Tagger.BatchTag(words);
                //var lstPOS = pos.Select(q => q.tag()).ToList();
                var lstPOS = new List<string>();

                var lstLems = new List<string>();
                for (int i = 0; i < words.Count; i++)
                {
                    try
                    {
                        lstLems.Add(__Lemmatizer.Lemmatize(words[i], lstPOS[i]));
                    }
                    catch (Exception ex)
                    {
                        try
                        {
                            lstLems.Add(__Lemmatizer.Lemmatize(words[i]));
                        }
                        catch (Exception exx)
                        {
                            lstLems.Add(words[i]);
                        }
                    }
                }


                Lems += string.Join(",", lstLems) + ";";
                POSTags += string.Join(",", lstPOS) + ";";
                OutPut += string.Join(",", words) + ";";
            }

            if (Lems.Length > 0) Lems = Lems.Substring(0, Lems.Length - 1);

            if (POSTags.Length > 0) POSTags = POSTags.Substring(0, POSTags.Length - 1);

            if (OutPut.Length > 0) OutPut = OutPut.Substring(0, OutPut.Length - 1);
            return OutPut;
        }


        public void CreateDataAllWords()
        {
            using (var db = new EFDB.PersonalityRecognitionEntities())
            {

                var insta_posts = db.instas.Where(q => q.code != null).ToList();
                var users = db.users.ToList();
                var words = db.Words.ToList();

                var i = 0;
                foreach (var user in users)
                {
                    if (db.AllWordsTrainResults.Any(q => q.UserCode == user.code)) continue;
                    if (!insta_posts.Any(q => q.username == user.insta)) continue;

                    i++;

                    var user_words = new List<UserWords>();

                    var list = insta_posts.Where(q => q.username == user.insta).ToList();
                    foreach (var post in list)
                    {
                        foreach (var word in post.NormalWords.Split(',').Where(q => q.Length > 3))
                        {
                            if (user_words.Any(q => q.Word == word))
                            {
                                var uw = user_words.First(q => q.Word == word);
                                uw.Score = uw.Score + 1;
                            }
                            else
                            {
                                user_words.Add(new UserWords() { Word = word, Score = 1 });
                            }
                        }
                    }
                    if (!user_words.Any()) continue;

                    var listIDs = new List<int>();
                    var j = 0;
                    var listFinal = user_words.OrderByDescending(q => q.Score).Skip(0).Take(200).ToList();
                    foreach (var uw in listFinal)
                    {
                        if (words.Any(q => q.TextWord == uw.Word))
                        {
                            var intw = words.First(q => q.TextWord == uw.Word).ID;
                            listIDs.Add(intw);
                            j++;
                        }
                    }

                    var x_item = listIDs.ToArray();

                    if (!listIDs.Any()) continue;

                    db.AllWordsTrainResults.Add(new EFDB.AllWordsTrainResult()
                    {
                        UserCode = user.code,
                        Features = string.Join(",", x_item),
                        Target = string.Join(",", GetTarget(user)),
                        ALLWords = Newtonsoft.Json.JsonConvert.SerializeObject(listFinal),
                    });

                    Console.WriteLine($"{i}");
                }


                db.SaveChanges();

                Console.WriteLine($"SaveChanges");

            }
        }

        private int[] GetTarget(EFDB.user user)
        {
            var y_item = new int[] {
                Convert.ToInt32(user.Neurosis1),
                Convert.ToInt32(user.Neurosis2),
                Convert.ToInt32(user.Neurosis3),

                Convert.ToInt32(user.Responsible1),
                Convert.ToInt32(user.Responsible2),
                Convert.ToInt32(user.Responsible3),

                Convert.ToInt32(user.Agreeableness1),
                Convert.ToInt32(user.Agreeableness2),
                Convert.ToInt32(user.Agreeableness3),

                Convert.ToInt32(user.PassionForNewExperiences1),
                Convert.ToInt32(user.PassionForNewExperiences2),
                Convert.ToInt32(user.PassionForNewExperiences3),

                Convert.ToInt32(user.ExtroversionIntroversion1),
                Convert.ToInt32(user.ExtroversionIntroversion2),
                Convert.ToInt32(user.ExtroversionIntroversion3)
            };

            return y_item;
        }

        private class UserWords
        {
            public int Score { get; set; }
            public string Word { get; set; }
        }

    }
}
