using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Linq;

namespace Dataset_Prediction_MLNET
{
    class Program
    {
        static void Main()
        {
            Console.Write("Introduce la ruta del archivo dataset (.txt): ");
            string filePath = Console.ReadLine();

            if (!File.Exists(filePath))
            {
                Console.WriteLine("El archivo no existe.");
                return;
            }

            var lines = File.ReadAllLines(filePath);

            if (lines.Length < 2)
            {
                Console.WriteLine("El archivo no contiene suficientes datos.");
                return;
            }

            // VALIDACIÓN DEL DATASET
            if (!EsDatasetValido(lines))
            {
                Console.WriteLine("El archivo no es un dataset válido.");
                Console.WriteLine("Debe contener TEXTO + TABULACIÓN + LABEL (0 o 1).");
                return;
            }

            Console.WriteLine("Dataset válido. Entrenando modelo...");

            var mlContext = new MLContext(seed: 1);

            var dataView = mlContext.Data.LoadFromTextFile<SentimentData>(
                path: filePath,
                separatorChar: '\t',
                hasHeader: false
            );

            // =========================
            // SPLIT TRAIN / TEST
            // =========================
            var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            // =========================
            // PIPELINE
            // =========================
            var pipeline =
                mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());

            Console.WriteLine("Entrenando modelo...");
            var model = pipeline.Fit(split.TrainSet);

            // =========================
            // EVALUACIÓN
            // =========================
            Console.WriteLine();
            Console.WriteLine("EVALUACIÓN DEL MODELO");

            var predictions = model.Transform(split.TestSet);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions);

            Console.WriteLine($"Accuracy:   {metrics.Accuracy:P2}");
            Console.WriteLine($"F1 Score:   {metrics.F1Score:P2}");
            Console.WriteLine($"AUC:        {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"Precision:  {metrics.PositivePrecision:P2}");
            Console.WriteLine($"Recall:     {metrics.PositiveRecall:P2}");

            // =========================
            // PREDICCIÓN INTERACTIVA
            // =========================
            var engine =
                mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            Console.WriteLine();
            Console.Write("Introduce un texto para analizar sentimiento: ");
            string inputText = Console.ReadLine();

            var prediction = engine.Predict(new SentimentData { Text = inputText });

            Console.WriteLine();

            while (string.IsNullOrWhiteSpace(inputText) == false)
            {
                Console.WriteLine($"Predicción: {(prediction.Prediction ? "POSITIVO" : "NEGATIVO")}");
                Console.WriteLine($"Probabilidad: {prediction.Probability:P2}");
                Console.WriteLine();
                Console.Write("Introduce un texto para analizar sentimiento (o presiona ENTER para salir): ");
                inputText = Console.ReadLine();
                if (string.IsNullOrWhiteSpace(inputText))
                    break;
                prediction = engine.Predict(new SentimentData { Text = inputText });
            }
        }

        // =========================
        // VALIDACIÓN DE DATASET
        // =========================
        static bool EsDatasetValido(string[] lines)
        {
            foreach (var line in lines)
            {
                if (!line.Contains('\t'))
                    return false;

                var parts = line.Split('\t');

                if (parts.Length != 2)
                    return false;

                if (string.IsNullOrWhiteSpace(parts[0]))
                    return false;

                if (parts[1] != "0" && parts[1] != "1")
                    return false;
            }

            return true;
        }
    }

    // =========================
    // MODELOS ML.NET
    // =========================
    public class SentimentData
    {
        [LoadColumn(0)]
        public string Text { get; set; }

        [LoadColumn(1)]
        public bool Label { get; set; }
    }

    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }
        public float Score { get; set; }
    }
}
