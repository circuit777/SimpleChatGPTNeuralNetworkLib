
import java.io.*;
import java.util.*;

public class SimpleNeuralNetwork {
    private final int inputSize;
    private final int[] layerSizes;
    private final double[][][] weights;
    private final String weightsFile = "weights.txt";

    public SimpleNeuralNetwork(int inputSize, int[] layerSizes) {
        this.inputSize = inputSize;
        this.layerSizes = layerSizes;
        this.weights = new double[layerSizes.length][][];

        for (int i = 0; i < layerSizes.length; i++) {
            int prevLayerSize = i == 0 ? inputSize : layerSizes[i - 1];
            weights[i] = new double[prevLayerSize][layerSizes[i]];
        }

        if (!loadWeightsFromFile()) {
            initializeWeights();
            saveWeightsToFile();
        }
    }

    public void train(double[] input, double[] targetOutput, double learningRate) {
        double[][] layerOutputs = forwardPass(input);
        double[][] deltas = backwardPass(layerOutputs, targetOutput);
        updateWeights(input, layerOutputs, deltas, learningRate);
    }

    public double[] predict(double[] input) {
        return forwardPass(input)[layerSizes.length - 1];
    }

    private double[][] forwardPass(double[] input) {
        double[][] layerOutputs = new double[layerSizes.length][];
        double[] currentInput = input;

        for (int i = 0; i < layerSizes.length; i++) {
            currentInput = applyActivationFunction(multiplyMatrix(currentInput, weights[i]));
            layerOutputs[i] = currentInput;
        }

        return layerOutputs;
    }
    
 // В классе SimpleNeuralNetwork добавьте следующий метод

    public double[] predictFromLatentVector(double[] latentVector) {
        // Проверяем, соответствует ли размер вектора размеру скрытого слоя
        if (latentVector.length != layerSizes[0]) {
            throw new IllegalArgumentException("Size of latent vector must match the size of the first hidden layer");
        }

        // Проходим только через оставшиеся слои (начиная со второго скрытого слоя, если он есть)
        double[][] layerOutputs = new double[layerSizes.length][];
        layerOutputs[0] = latentVector; // Первый "скрытый" слой уже предоставлен в виде латентного вектора

        for (int i = 1; i < layerSizes.length; i++) {
            layerOutputs[i] = applyActivationFunction(multiplyMatrix(layerOutputs[i - 1], weights[i]));
        }

        // Возвращаем выход последнего слоя
        return layerOutputs[layerSizes.length - 1];
    }


    private double[][] backwardPass(double[][] layerOutputs, double[] targetOutput) {
        double[][] deltas = new double[layerSizes.length][];

        for (int i = layerSizes.length - 1; i >= 0; i--) {
            double[] layerError;
            if (i == layerSizes.length - 1) {
                layerError = new double[targetOutput.length];
                for (int j = 0; j < targetOutput.length; j++) {
                    layerError[j] = targetOutput[j] - layerOutputs[i][j];
                }
            } else {
                layerError = multiplyMatrix(deltas[i + 1], transposeMatrix(weights[i + 1]));
            }

            deltas[i] = new double[layerSizes[i]];
            for (int j = 0; j < layerSizes[i]; j++) {
                deltas[i][j] = layerError[j] * layerOutputs[i][j] * (1 - layerOutputs[i][j]);
            }
        }

        return deltas;
    }

    private void updateWeights(double[] input, double[][] layerOutputs, double[][] deltas, double learningRate) {
        double[] currentInput = input;

        for (int i = 0; i < layerSizes.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    weights[i][j][k] += learningRate * currentInput[j] * deltas[i][k];
                }
            }
            currentInput = layerOutputs[i];
        }
    }

    // Other utility methods (loadWeightsFromFile, saveWeightsToFile, initializeWeights, etc.) remain the
    boolean loadWeightsFromFile() {
        File file = new File(weightsFile);
        if (!file.exists()) {
            return false;
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            for (int l = 0; l < layerSizes.length; l++) {
                for (int i = 0; i < (l == 0 ? inputSize : layerSizes[l - 1]); i++) {
                    String[] line = reader.readLine().split(" ");
                    for (int j = 0; j < layerSizes[l]; j++) {
                        weights[l][i][j] = Double.parseDouble(line[j]);
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }

    void saveWeightsToFile() {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(weightsFile))) {
            for (int l = 0; l < layerSizes.length; l++) {
                for (int i = 0; i < (l == 0 ? inputSize : layerSizes[l - 1]); i++) {
                    for (int j = 0; j < layerSizes[l]; j++) {
                        writer.write(weights[l][i][j] + " ");
                    }
                    writer.newLine();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void initializeWeights() {
        Random random = new Random();
        for (int l = 0; l < layerSizes.length; l++) {
            for (int i = 0; i < weights[l].length; i++) {
                for (int j = 0; j < weights[l][i].length; j++) {
                    weights[l][i][j] = random.nextDouble() * 2 - 1;
                }
            }
        }
    }

    private double[] multiplyMatrix(double[] vector, double[][] matrix) {
        double[] result = new double[matrix[0].length];
        for (int i = 0; i < matrix[0].length; i++) {
            for (int j = 0; j < vector.length; j++) {
                result[i] += vector[j] * matrix[j][i];
            }
        }
        return result;
    }

    private double[][] transposeMatrix(double[][] matrix) {
        double[][] result = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }
 // Дополнение к классу SimpleNeuralNetwork

 // Функция для получения выходов указанного слоя
 public double[] getLayerOutputs(double[] input, int layerNumber) {
     if (layerNumber < 0 || layerNumber >= layerSizes.length) {
         throw new IllegalArgumentException("Invalid layer number");
     }

     double[][] layerOutputs = forwardPass(input);

     return layerOutputs[layerNumber];
 }

    private double[] applyActivationFunction(double[] input) {
        double[] result = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            result[i] = 1 / (1 + Math.exp(-input[i]));
        }
        return result;
    
}

}
