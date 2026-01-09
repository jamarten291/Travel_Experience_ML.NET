using System;
using System.IO;
using Microsoft.ML;
using System;

class Program
{
    static void Main()
    {
        MLContext mlContext = new MLContext(seed: 1);

        // ENTRENAR MODELO MONUMENTOS
        var modeloMonumentos = EntrenarModelo(
            mlContext,
            "..\\..\\..\\Data\\monumentos.txt",
            "ModeloMonumentos.zip"
        );

        // ENTRENAR MODELO OCIO
        var modeloOcio = EntrenarModelo(
            mlContext,
            "..\\..\\..\\Data\\ocio.txt",
            "ModeloOcio.zip"
        );

        // ENTRENAR MODELO OCIO
        var modeloGastronomia = EntrenarModelo(
            mlContext,
            "..\\..\\..\\Data\\gastronomia.txt",
            "ModeloGastronomia.zip"
        );

        // INPUT USUARIO
        Console.WriteLine("OPINIÓN SOBRE MONUMENTOS:");
        Console.Write("Escribe tu opinión: ");
        string opinionMonumentos = Console.ReadLine();

        Console.WriteLine();
        Console.WriteLine("OPINIÓN SOBRE ACTIVIDADES DE OCIO:");
        Console.Write("Escribe tu opinión: ");
        string opinionOcio = Console.ReadLine();

        Console.WriteLine();
        Console.WriteLine("OPINIÓN SOBRE GASTRONOMÍA:");
        Console.Write("Escribe tu opinión: ");
        string opinionGastronomia = Console.ReadLine();

        // PREDICCIONES
        var engineMonumentos =
            mlContext.Model.CreatePredictionEngine<OpinionData, OpinionPrediction>(modeloMonumentos);

        var engineOcio =
            mlContext.Model.CreatePredictionEngine<OpinionData, OpinionPrediction>(modeloOcio);

        var engineGastronomia =
            mlContext.Model.CreatePredictionEngine<OpinionData, OpinionPrediction>(modeloGastronomia);

        var resultadoMonumentos = engineMonumentos.Predict(
            new OpinionData { Text = opinionMonumentos });

        var resultadoOcio = engineOcio.Predict(
            new OpinionData { Text = opinionOcio });

        var resultadoGastronomia = engineGastronomia.Predict(
            new OpinionData { Text = opinionGastronomia });

        // RESULTADO FINAL
        Console.WriteLine();
        Console.WriteLine("===== RESULTADO DE LA EXPERIENCIA =====");

        Console.WriteLine($"Monumentos: {(resultadoMonumentos.Prediction ? "Le han gustado" : "No le han gustado")}");
        Console.WriteLine($"Actividades de ocio: {(resultadoOcio.Prediction ? "Le han gustado" : "No le han gustado")}");
        Console.WriteLine($"Gastronomía: {(resultadoGastronomia.Prediction ? "Le ha gustado la comida" : "No le ha gustado la comida")}");

        Console.WriteLine();

        // Crea una lista de opiniones y cuenta cuántas son positivas, tras eso las guarda en opinionesPositivas
        int opinionesPositivas = new[]
        {
            resultadoMonumentos.Prediction,
            resultadoOcio.Prediction,
            resultadoGastronomia.Prediction
        }.Count(p => p);

        if (opinionesPositivas == 3)
        {
            Console.WriteLine("La experiencia general del usuario en la ciudad ha sido MUY POSITIVA.");
        }
        else if (opinionesPositivas >= 1)
        {
            Console.WriteLine("La experiencia general del usuario en la ciudad ha sido ACEPTABLE.");
        }
        else
        {
            Console.WriteLine("La experiencia general del usuario en la ciudad ha sido NEGATIVA.");
        }
    }

    // MÉTODO DE ENTRENAMIENTO
    static ITransformer EntrenarModelo(
        MLContext mlContext,
        string rutaDataset,
        string nombreModelo)
    {
        var data = mlContext.Data.LoadFromTextFile<OpinionData>(
            rutaDataset,
            separatorChar: '\t',
            hasHeader: false);

        var pipeline =
            mlContext.Transforms.Text.FeaturizeText("Features", nameof(OpinionData.Text))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());

        var model = pipeline.Fit(data);

        mlContext.Model.Save(model, data.Schema, nombreModelo);

        return model;
    }
}