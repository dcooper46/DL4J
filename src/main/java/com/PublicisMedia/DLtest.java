package com.PublicisMedia;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;

/**
 * Created by dancoope on 10/5/2016.
 */
public class DLtest
{
    private static Logger log = LoggerFactory.getLogger(DLtest.class);

    private static final int HIDDEN_LAYER_WIDTH = 80;
    private static final int HIDDEN_LAYER_CONT = 5;

    private static final Random r = new Random();

    private static double[] cpm = {3.678, 3.599, 3.637, 3.65, 3.517, 3.567, 3.594,
            3.557, 3.523, 3.404, 3.381, 3.43, 3.465, 3.433, 3.359, 3.415, 3.445,
            3.473, 3.415, 3.483, 3.502, 3.465, 3.35, 3.32, 3.325, 3.365, 3.439,
            3.48, 3.464, 3.468, 3.447, 3.469, 3.498, 3.527, 3.532, 3.399, 3.409,
            3.416, 3.406, 3.465, 3.547, 3.557, 3.475, 3.428, 3.386, 3.408, 3.404,
            3.43, 3.476, 3.45, 3.427, 3.405, 3.383, 3.439, 3.539, 3.504, 3.474,
            3.445, 3.471, 3.446, 3.469, 3.385, 3.372, 3.435, 3.469, 3.463,
            3.401, 3.424, 3.473, 3.454, 3.428, 3.411, 3.402, 3.368, 3.442, 3.367,
            3.346, 3.389, 3.369, 3.463, 3.512, 3.435, 3.408, 3.351, 3.315,
            3.37, 3.418, 3.428, 3.481, 3.531, 3.514, 3.317};
    private static final List<Double> cpm_list = new ArrayList<Double>();
    public static void main(String[] args) throws Exception
    {
        /**
         * fill lists
         */
        LinkedHashSet<Double> cpm_values = new LinkedHashSet<Double>();
        for(Double i : cpm)
        {
            cpm_values.add(i);
        }
        cpm_list.addAll(cpm_values);

/*        int nlSkip = 1;
        String delim = ",";
        SequenceRecordReader recordReader = new CSVSequenceRecordReader(nlSkip, delim);
        recordReader.initialize(new FileSplit(new ClassPathResource
                        ("dbm_REI_alldays_cpm.csv").getFile()));

        DataSetIterator iterator = new SequenceRecordReaderDataSetIterator
                (recordReader, 100, -1, 1, true);

        DataSet allData = iterator.next();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);

        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(train);
        normalizer.transform(train);
        normalizer.transform(test);
*/

        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration
                .Builder();
        builder.iterations(20);
        builder.learningRate(0.001);
        builder.optimizationAlgo(OptimizationAlgorithm
                .STOCHASTIC_GRADIENT_DESCENT);
        builder.seed(123);
        builder.biasInit(0);
        builder.miniBatch(false);
        builder.updater(Updater.RMSPROP);
        builder.weightInit(WeightInit.XAVIER);

        ListBuilder listBuilder = builder.list();

        for(int i = 0; i < HIDDEN_LAYER_CONT; i++)
        {
            GravesLSTM.Builder hiddenLayerBuilder = new GravesLSTM.Builder();
            hiddenLayerBuilder.nIn(i == 0 ? cpm_values.size() :
                                            HIDDEN_LAYER_WIDTH);
            hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH);
            hiddenLayerBuilder.activation("tanh");
            listBuilder.layer(i, hiddenLayerBuilder.build());
        }

        RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer
                .Builder(LossFunctions.LossFunction.MCXENT);
        outputLayerBuilder.activation("softmax");
        outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH);
        outputLayerBuilder.nOut(cpm_values.size());
        listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build());

        listBuilder.pretrain(false);
        listBuilder.backprop(true);

        MultiLayerConfiguration conf = listBuilder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        /**
         * create training data from cpm list
         */
        INDArray input = Nd4j.zeros(1, cpm_list.size(), cpm.length);
        INDArray labels = Nd4j.zeros(1, cpm_list.size(), cpm.length);

        int pos = 0;
        for(double currentCpm : cpm)
        {
            double nextCpm = cpm[(pos + 1) % (cpm.length)];

            input.putScalar(new int[] {0, cpm_list.indexOf(currentCpm), pos}, 1);
            labels.putScalar(new int[] {0, cpm_list.indexOf(nextCpm), pos}, 1);
            pos++;
        }

        DataSet train = new DataSet(input, labels);
        /*
        System.out.println(input.shapeInfoToString());
        System.out.println(labels.shapeInfoToString());
        */
        for(int epoch = 0; epoch < 100; epoch++)
        {
            System.out.println("Current Epoch: " + epoch);
            net.fit(train);

            net.rnnClearPreviousState();

            INDArray init = Nd4j.zeros(cpm_list.size());
            init.putScalar(cpm_list.indexOf(cpm[0]), 1);

            INDArray output = net.rnnTimeStep(init);

            for(int j = 0; j < cpm.length; j++)
            {
                double [] outputProbDist = new double[output.columns()];

                for(int k = 0; k < outputProbDist.length; k++)
                {
                    outputProbDist[k] = output.getDouble(k);
                }

                int index = findIndexOfHighestValue(outputProbDist);

                System.out.print(cpm_list.get(index) + " ");

                INDArray nextInput = Nd4j.zeros(cpm_list.size());
                nextInput.putScalar(index, 1);
                output = net.rnnTimeStep(nextInput);
            }
            System.out.println("\n");
        }

    }

    private static int findIndexOfHighestValue(double[] distribution)
    {
        int maxValueIndex = 0;
        double maxValue = 0;
        for (int i = 0; i < distribution.length; i++)
        {
            if(distribution[i] > maxValue) {
                maxValue = distribution[i];
                maxValueIndex = i;
            }
        }
        return maxValueIndex;
    }

}
