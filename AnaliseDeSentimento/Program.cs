using Microsoft.ML;
using Microsoft.ML.Data;
using AnaliseDeSentimento; 
using static Microsoft.ML.DataOperationsCatalog;

//cria uma string com os dados do arquivo no diretório;
string _dataPath = Path.Combine(Environment.CurrentDirectory, "data", "yelp_labelled.txt");


MLContext mlContext = new MLContext();

TrainTestData splitDataView = LoadData(mlContext); //O método LoadData executa as seguintes tarefas: Carrega dados; Divide o conjunto de dados carregados em conjuntos de teste e treinamento;
                                                   //Retorna os resultados do treinamento e teste;

ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet); //O método BuildAndTrai...() executa as tarefas: Extrai e transforma dados; Treina o modelo;
                                                                            //Prevê o sentimento pelos dados do teste; 
                                                                            //Retorna o modelo;
Evaluate(mlContext, model, splitDataView.TestSet); //O método Evaluate(): Carrega o conjunto de dados de teste; Cria o avaliador BinaryClassification;
                                                   //Avalia o modelo e cria métricas; Exibe as métricas;

UseModelWithSingleItem(mlContext, model); //O método UseModelWithSingleItem():
                                          //Prevê o sentimento com base nos dados de teste.
                                          //Combina dados de teste e previsões para relatórios.
                                          //Exibe os resultados previstos.

UseModelWithBatchItems(mlContext, model); //O método UseModelWithBatchItems() executa as seguintes tarefas:
                                          //Cria dados de teste em lote.
                                          //Prevê o sentimento com base nos dados de teste.
                                          //Combina dados de teste e previsões para relatórios.
                                          //Exibe os resultados previstos.

// *===================================== Métodos ==========================================*

TrainTestData LoadData(MLContext mlContext)
{
    IDataView dataView = mlContext.Data.LoadFromTextFile<DadosdeSentimento>(_dataPath, hasHeader: false); //método LoadFromTextFile define o esquema de dados e lê o arquivo. 
                                                                                                          //Ele usa as variáveis de caminho de dados e retorna uma IDataView.

    TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2); //O método TrainTestSplit () divide o DataSet carregado em conjuntos de valores de teste
                                                                                              //e de treinamento e retorná-los na DataOperationsCatalog.TrainTestData classe.
    return splitDataView;                                                                     //Especifique o percentual de dados do conjunto de teste com o parâmetro testFraction.
                                                                                              //O padrão é 10% e, nesse caso, você usa 20% para avaliar mais dados.
}

ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
{
    //O método FeaturizeText() no código abaixo converte a coluna de texto (textoSentimento)
    //em uma coluna Features do tipo chave numérica usada pelo algoritmo de aprendizado de máquina,
    //adicionando-a como uma nova coluna do conjunto de dados;

    var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(DadosdeSentimento.textoSentimento))
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features")); //O "SdcaLogisticRegressionBinaryTrainer" é o algoritmo de treinamento de classificação.
                                                                                                                                      //Ele é acrescentado ao estimator e aceita o parâmetro textoSentimento ("Features")
                                                                                                                                      //personalizado e o parâmetro de entrada Label para aprender com os dados históricos.

    Console.WriteLine("=============== Create and Train the Model ===============");
    var model = estimator.Fit(splitTrainSet); //O método Fit() treina o modelo transformando o conjunto de dados e aplicando o treinamento.
    Console.WriteLine("=============== End of training ===============");
    Console.WriteLine();

    return model; 
}

void Evaluate(MLContext mLContext, ITransformer model, IDataView splitTestSet)  
{                                                                              

    Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");

    IDataView predictions = model.Transform(splitTestSet);

    CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

    //Uma vez que você tem o conjunto de previsão(predictions), 
    //o método Evaluate() avalia o modelo, 
    //que compara os valores previstos com os Labels reais no conjunto de dados de teste
    //e retorna um objeto CalibratedBinaryClassificationMetrics sobre o desempenho do modelo.

    //Então, exibe as métricas:
    Console.WriteLine();
    Console.WriteLine("Modelo de validação das métricas");
    Console.WriteLine("--------------------------------");
    Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
    Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
    Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
    Console.WriteLine("=============== Fim da validação ===============");

    //A métrica Accuracy obtém a precisão de um modelo, que é a proporção de previsões corretas no conjunto de teste.

    //A métrica AreaUnderRocCurve indica a confiabilidade do modelo ao classificar corretamente as classes positivas e negativas.
    //Você deseja que o AreaUnderRocCurve seja o mais próximo possível de um.

    //A métrica F1Score obtém a pontuação F1 do modelo,
    //que é uma medida do equilíbrio entre precisão e recall.
    //Você deseja que o F1Score seja o mais próximo possível de um.
}

void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
{
    PredictionEngine<DadosdeSentimento, PredicaodeSentimento> predictionFunction = mlContext.Model.CreatePredictionEngine<DadosdeSentimento, PredicaodeSentimento>(model);

    //O PredictionEngine é uma API de conveniência,
    //que permite que você execute uma previsão em uma única instância de dados.
    //PredictionEngine não é thread-safe.É aceitável usar em ambientes de protótipo ou de thread único.
    //A documentação indica o uso de PredictionEnginePool para melhorar o desempenho e a segurança de thread.
    //Ele cria um ObjectPool dos PredictionEngine objetos para uso em todo o aplicativo. 

    //OBS: A documentação avisa que o serviço PredictionEnginePool está atualmente em versão prévia (02/08/2022);

    DadosdeSentimento sampleStatement = new DadosdeSentimento  //Para testes
    {
        textoSentimento = "This was a very bad steak"
    };

    var resultPrediction = predictionFunction.Predict(sampleStatement); //A função Predict() faz uma previsão em uma única coluna de dados.

    Console.WriteLine();
    Console.WriteLine("=============== Teste de modelo de predicao com um simples exemplo ===============");

    Console.WriteLine();
    Console.WriteLine($"Sentiment: {resultPrediction.textoSentimento} | Prediction: {(Convert.ToBoolean(resultPrediction.Predicao) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probabilidade} ");

    Console.WriteLine("=============== Fim das predicoes ===============");
    Console.WriteLine();
}

void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
{
    IEnumerable<DadosdeSentimento> sentiments = new[]
{
    new DadosdeSentimento
    {
        textoSentimento = "This was a horrible meal"
    },
    new DadosdeSentimento
    {
        textoSentimento = "I love this spaghetti."
    }
};

    //Prever sentidomento do comentario acima:

    IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

    IDataView predictions = model.Transform(batchComments);

    // Usa o modelo para prever se o dados do comentario é Positivo (1) ou Negativo (0).
    IEnumerable<PredicaodeSentimento> predictedResults = mlContext.Data.CreateEnumerable<PredicaodeSentimento>(predictions, reuseRowObject: false);

    Console.WriteLine();

    Console.WriteLine("=============== Teste de predicoes de modelos com multiplas escolhas ===============");

    //Como SentimentPrediction é herdado de SentimentData, 
    //o método Transform() preencheu SentimentText com os campos previstos. 
    //Conforme o processo do ML.NET processa, cada componente adiciona colunas e 
    //isso facilita a exibição dos resultados:

    foreach (PredicaodeSentimento prediction in predictedResults)
    {
        Console.WriteLine($"Sentiment: {prediction.textoSentimento} | Prediction: {(Convert.ToBoolean(prediction.Predicao) ? "Positive" : "Negative")} | Probability: {prediction.Probabilidade} ");
    }
    Console.WriteLine("=============== Fim das predições ===============");
}