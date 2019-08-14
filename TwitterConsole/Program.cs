using SimpleNetNlp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Tweetinvi;
using Tweetinvi.Events;
using Tweetinvi.Models;
using Tweetinvi.Streams;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Transforms.Text;
using System.IO;
using Microsoft.ML.Data;

namespace TwitterConsole
{
    class Program
    {
        // Original code belongs to LUIS QUINTANILLA
        // Found at:
        // http://luisquintanilla.me/2018/01/18/real-time-sentiment-analysis-csharp/
        // Used and repurposed for this applicaiton

        // Also utilized documentation from Microsoft
        // Found at:
        // http://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials
        // 

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            IDataView data = mlContext.Data.LoadFromTextFile<TweetData>("Data/tweetData.csv", separatorChar: ',', hasHeader: true);
            //var textEstimator = mLContext.Transforms.Text.FeaturizeText("Text");
            //ITransformer textTransformer = textEstimator.Fit(data);
            //IDataView transformedData = textTransformer.Transform(data);

            var textEstimator = mlContext.Transforms.Text.NormalizeText("Text")   // all lowercase
                .Append(mlContext.Transforms.Text.TokenizeIntoWords("Text"))       // splits into individual words
                .Append(mlContext.Transforms.Text.RemoveDefaultStopWords("Text"));  // removes words like is and a
            ITransformer textTransformer = textEstimator.Fit(data);
            IDataView transformedData = textTransformer.Transform(data);


            Auth.SetUserCredentials("9UOtZgv99y1V1eNEKgChqkJ2j", "hPA95aOpGPTXF0HtPRdozZg5bvCNopgT1JHl30mEQvnZ1XMfiG", "1042611858778275841-VA8PzcBkt9spC8U1Knp4yDV8hcTy2Y", "VW2PHhlHj8ejh41RMSbUjChYy2KYBjkj94gKye8FTZlTJ");
            //var firstTweet = Tweet.GetTweet(1160592005573021696);
            //DisplayTweet(firstTweet);

            //var stream = Tweetinvi.Stream.CreateFilteredStream();
            //stream.AddTrack("sarcasm", tweet => { DisplayTweet(tweet); });
            //stream.AddTweetLanguageFilter("en");
            //stream.StartStreamMatchingAllConditions();

            //var stream = Tweetinvi.Stream.CreateFilteredStream();
            //stream.AddTrack("metal gear solid", tweet => { AskIfTweetIsSarcastic(tweet); });
            //stream.AddTrack("magic johnson", tweet => { AskIfTweetIsSarcastic(tweet); });
            //stream.AddTrack("gta san andreas", tweet => { AskIfTweetIsSarcastic(tweet); });
            //stream.AddTrack("barstool", tweet => { AskIfTweetIsSarcastic(tweet); });
            //stream.AddTrack("unions", tweet => { AskIfTweetIsSarcastic(tweet); });
            //stream.AddTweetLanguageFilter("en");
            //stream.StartStreamMatchingAllConditions();

            //var tweet = Tweet.GetTweet(1160592005573021696);
            //var reply = Tweet.GetTweet(1160592203762348032);
            //DisplayTweet(tweet);
            //Console.ReadKey();
            //DisplayTweet(reply);
            //Console.ReadKey();
            
            TrainTestData splitDataView = LoadData(mlContext);
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

            Evaluate(mlContext, model, splitDataView.TestSet);

            //UseModelWithSingleItem(mlContext, model);

            //UseModelWithBatchItems(mlContext, model);

            var AOCStream = Tweetinvi.Stream.CreateFilteredStream();
            AOCStream.AddTrack("BRGSHTY", tweet => { PredictSarcasm(tweet, mlContext, model); });
            AOCStream.AddTweetLanguageFilter("en");
            AOCStream.StartStreamMatchingAllConditions();

            //var TrumpStream = Tweetinvi.Stream.CreateFilteredStream();
            //TrumpStream.AddTrack("realDonaldTrump", tweet => { PredictSarcasm(tweet, mlContext, model); });
            //TrumpStream.AddTweetLanguageFilter("en");
            //TrumpStream.StartStreamMatchingAllConditions();

            Console.ReadKey();
        }

        public static void PredictSarcasm(ITweet tweet, MLContext mlContext, ITransformer model)
        {
            if (tweet.InReplyToStatusId != null)
            {
                var sanitizedTweet = Sanitize(tweet.FullText);
                var tweetSentiment = new Sentence(sanitizedTweet).Sentiment;
                var tweet2 = Tweet.GetTweet((long)tweet.InReplyToStatusId);
                var sanitizedTweet2 = Sanitize(tweet2.FullText);
                var tweet2Sentiment = new Sentence(sanitizedTweet2).Sentiment;

                IEnumerable<TweetData> data = new[]
                {
                    new TweetData
                    {
                        Text = sanitizedTweet,
                        Sentiment = tweetSentiment.ToString(),
                        ReplyToText = sanitizedTweet2,
                        ReplyToSentiment = tweet2Sentiment.ToString()
                    }
                };

                IDataView tweets = mlContext.Data.LoadFromEnumerable(data);
                IDataView predictions = model.Transform(tweets);

                IEnumerable<TweetPrediction> tweetPredictions = mlContext.Data.CreateEnumerable<TweetPrediction>(predictions, reuseRowObject: false);

                foreach (TweetPrediction prediction in tweetPredictions)
                {
                    DisplayTweetPrediction(prediction);
                    Console.WriteLine($"Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Sarcastic" : "Not Sarcastic")} | Probability: {prediction.Probability}");
                }
                Console.WriteLine("");
                Console.WriteLine("Press any key to get the next tweet in the stream");
                Console.ReadKey();
            }
        }

        //private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        //{
        //    PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
        //    SentimentData sampleStatement = new SentimentData
        //    {
        //        SentimentText = "This was a very bad steak"
        //    };
        //    var resultPrediction = predictionFunction.Predict(sampleStatement);
        //    Console.WriteLine();
        //    Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

        //    Console.WriteLine();
        //    Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

        //    Console.WriteLine("=============== End of Predictions ===============");
        //    Console.WriteLine();
        //}

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        public static TrainTestData LoadData(MLContext mLContext)
        {
            IDataView data = mLContext.Data.LoadFromTextFile<TweetData>("Data/tweetData.csv", separatorChar: ',', hasHeader: true);
            TrainTestData splitDataView = mLContext.Data.TrainTestSplit(data, testFraction: 0.2);
            return splitDataView;
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            var options = new TextFeaturizingEstimator.Options() // Retrieved options are from docs.Microsoft, these were default in the tutorial
            {
                // Also output tokenized words
                OutputTokensColumnName = "OutputTokens",
                CaseMode = TextNormalizingEstimator.CaseMode.Lower,
                // Use ML.NET's built-in stop word remover
                StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options() { Language = TextFeaturizingEstimator.Language.English },
                WordFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 2, UseAllLengths = true },
                CharFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 3, UseAllLengths = false },
            };

            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", options: options, inputColumnNames: new string[] { nameof(TweetData.Text), nameof(TweetData.ReplyToText) })
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            Console.WriteLine("=============== Creating and Training the Model ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("======================= End of training =======================");
            Console.WriteLine();
            return model;
        }

        private static void AskIfTweetIsSarcastic(ITweet tweet)
        {
            if (tweet.InReplyToStatusId != null)
            {
                Console.Clear();
                DisplaySingleTweet(tweet);
                Console.WriteLine("");
                Console.WriteLine("Is the top tweet sarcastic, with context to the second?");
                Console.WriteLine("Press Y for yes, N for no, S to skip.");

                bool gettingValidInput = true;
                while (gettingValidInput)
                {
                    var userInput = Console.ReadKey().Key;
                    switch (userInput)
                    {
                        case ConsoleKey.Y:
                            LabelTweet(tweet, 1);
                            gettingValidInput = false;
                            break;
                        case ConsoleKey.N:
                            LabelTweet(tweet, 0);
                            gettingValidInput = false;
                            break;
                        case ConsoleKey.S:
                            gettingValidInput = false;
                            break;
                        default:

                            break;
                    }
                }
            }
        }

        private static void LabelTweet(ITweet tweet, int label)
        {
            using (StreamWriter sw = new StreamWriter("Data/tweetData.csv", true))
            {
                var sanitizedTweet = Sanitize(tweet.FullText);
                var tweetSentiment = new Sentence(sanitizedTweet).Sentiment;
                var tweet2 = Tweet.GetTweet((long)tweet.InReplyToStatusId);
                var sanitizedTweet2 = Sanitize(tweet2.FullText);
                var tweet2Sentiment = new Sentence(sanitizedTweet2).Sentiment;
                sw.WriteLine(Regex.Replace(Regex.Replace(sanitizedTweet + "," + tweetSentiment + "," + tweet2 + "," + tweet2Sentiment + "," + label.ToString(), @"\s+", " "), @"\t|\n|\r", " ").Trim());
            }
        }

        private static void DisplayTweetPrediction(TweetPrediction prediction)
        {
            Console.WriteLine("=======================================================================================================================");
            Console.WriteLine("Tweet:   " + prediction.Text);
            Console.WriteLine("Sentiment:   " + prediction.Sentiment);
            Console.WriteLine("Tweet Replied To:   " + prediction.ReplyToText);
            Console.WriteLine("Sentiment Replied To:   " + prediction.ReplyToSentiment);
            Console.WriteLine("=======================================================================================================================");
        }

        private static void DisplaySingleTweet(ITweet tweet)
        {
            Console.WriteLine("=======================================================================================================================");
            Console.WriteLine("Tweet:   " + tweet);
            DisplayTweetSentiment(tweet, "");
            Console.WriteLine("=======================================================================================================================");
            Console.WriteLine("In Reply To:");
            if (tweet.InReplyToStatusId != null)
            {
                var tweet2 = Tweet.GetTweet((long)tweet.InReplyToStatusId);
                Console.WriteLine("Tweet:   " + tweet2);
                DisplayTweetSentiment(tweet2, "");
            }
            Console.WriteLine("=======================================================================================================================");
        }

        private static void DisplayTweet(ITweet tweet, int offset = 0)
        {
            string offsetString = "";
            for (int i = 0; i < offset; i++)
            {
                offsetString += " ";
            }

            Console.WriteLine(offsetString + "===============================================================");
            Console.WriteLine(offsetString + "Tweeter:   " + tweet.CreatedBy);
            Console.WriteLine(offsetString + "Tweet:   " + tweet);
            DisplayTweetSentiment(tweet, offsetString);
            if (tweet.InReplyToStatusId != null)
            {
                DisplayTweet(Tweet.GetTweet((long)tweet.InReplyToStatusId), offset + 1);
            }
            Console.WriteLine(offsetString + "===============================================================");
            Console.WriteLine();
        }

        private static void DisplayTweetSentiment(ITweet tweet, string offsetString)
        {
            Console.WriteLine("");
            var sanitized = Sanitize(tweet.FullText);
            if (sanitized == "")
            {
                Console.WriteLine(offsetString + "No Usable Words To Gather Sentiment");
            }
            else
            {
                var sentence = new Sentence(sanitized);
                Console.WriteLine(offsetString + "Sentiment:   " + sentence.Sentiment + "   |   " + "Words:   " + sentence.Words.Count);
            }
        }

        private static string Sanitize(string raw)
        {
            return Regex.Replace(raw, @"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ").ToString();
        }
    }
}
