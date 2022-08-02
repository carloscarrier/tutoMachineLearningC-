using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnaliseDeSentimento
{
    public class DadosdeSentimento
    {
        [LoadColumn(0)]
        public string textoSentimento = "";

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentimento;
    }

    public class PredicaodeSentimento : DadosdeSentimento
    {
        [ColumnName("PredictedLabel")]
        public bool Predicao { get; set; }

        public float Probabilidade { get; set; }

        public float Score { get; set; }
    }
}


