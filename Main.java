import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class Main {

    private final int inputSize;
    private final int hiddenLayerSize;
    private final int outputSize;
    private final double[][] hiddenLayer1Weights;
    private final double[][] hiddenLayer2Weights;
    private final double[][] outputLayerWeights;
    private final String weightsFile = "weights.txt";

    public Main(int inputSize, int hiddenLayerSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenLayerSize = hiddenLayerSize;
        this.outputSize = outputSize;
        hiddenLayer1Weights = new double[inputSize][hiddenLayerSize];
        hiddenLayer2Weights = new double[hiddenLayerSize][hiddenLayerSize];
        outputLayerWeights = new double[hiddenLayerSize][outputSize];

        if (!loadWeightsFromFile()) {
            initializeWeights();
            saveWeightsToFile();
        }
    }
    
    public void train(double[] input, double[] targetOutput, double learningRate) {
        double[] hiddenLayer1Output = applyActivationFunction(multiplyMatrix(input, hiddenLayer1Weights));
        double[] hiddenLayer2Output = applyActivationFunction(multiplyMatrix(hiddenLayer1Output, hiddenLayer2Weights));
        double[] output = applyActivationFunction(multiplyMatrix(hiddenLayer2Output, outputLayerWeights));

        double[] outputError = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            outputError[i] = targetOutput[i] - output[i];
        }

        double[] outputDelta = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            outputDelta[i] = outputError[i] * output[i] * (1 - output[i]);
        }

        double[] hiddenLayer2Error = multiplyMatrix(outputDelta, transposeMatrix(outputLayerWeights));
        double[] hiddenLayer2Delta = new double[hiddenLayerSize];
        for (int i = 0; i < hiddenLayerSize; i++) {
            hiddenLayer2Delta[i] = hiddenLayer2Error[i] * hiddenLayer2Output[i] * (1 - hiddenLayer2Output[i]);
        }

        double[] hiddenLayer1Error = multiplyMatrix(hiddenLayer2Delta, transposeMatrix(hiddenLayer2Weights));
        double[] hiddenLayer1Delta = new double[hiddenLayerSize];
        for (int i = 0; i < hiddenLayerSize; i++) {
            hiddenLayer1Delta[i] = hiddenLayer1Error[i] * hiddenLayer1Output[i] * (1 - hiddenLayer1Output[i]);
        }

        updateWeights(hiddenLayer1Weights, input, hiddenLayer1Delta, learningRate);
        updateWeights(hiddenLayer2Weights, hiddenLayer1Output, hiddenLayer2Delta, learningRate);
        updateWeights(outputLayerWeights, hiddenLayer2Output, outputDelta, learningRate);
    }
    
    private void updateWeights(double[][] weights, double[] input, double[] deltas, double learningRate) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] += learningRate * input[i] * deltas[j];
            }
        }
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

    public double[] predict(double[] input) {
        double[] hiddenLayer1Output = applyActivationFunction(multiplyMatrix(input, hiddenLayer1Weights));
        double[] hiddenLayer2Output = applyActivationFunction(multiplyMatrix(hiddenLayer1Output, hiddenLayer2Weights));
        double[] output = applyActivationFunction(multiplyMatrix(hiddenLayer2Output, outputLayerWeights));
        
        return output;
    }

    private double[] roundOutput(double[] output) {
        double[] result = new double[output.length];
        
        for (int i = 0; i < output.length; i++) {
            if (output[i] < 0.5 - 0.5 / 3) {
                result[i] = -1;
            } else if (output[i] > 0.5 + 0.5 / 3) {
                result[i] = 1;
            } else {
                result[i] = 0;
            }
        }
        
        return result;
    }

     boolean loadWeightsFromFile() {
        File file = new File(weightsFile);
        if (!file.exists()) {
            return false;
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            for (int i = 0; i < inputSize; i++) {
                String[] line = reader.readLine().split(" ");
                for (int j = 0; j < hiddenLayerSize; j++) {
                    hiddenLayer1Weights[i][j] = Double.parseDouble(line[j]);
                }
            }
            for (int i = 0; i < hiddenLayerSize; i++) {
                String[] line = reader.readLine().split(" ");
                for (int j = 0; j < hiddenLayerSize; j++) {
                    hiddenLayer2Weights[i][j] = Double.parseDouble(line[j]);
                }
            }
            for (int i = 0; i < hiddenLayerSize; i++) {
                String[] line = reader.readLine().split(" ");
                for (int j = 0; j < outputSize; j++) {
                    outputLayerWeights[i][j] = Double.parseDouble(line[j]);
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
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < hiddenLayerSize; j++) {
                    writer.write(hiddenLayer1Weights[i][j] + " ");
                }
                writer.newLine();
            }
            for (int i = 0; i < hiddenLayerSize; i++) {
                for (int j = 0; j < hiddenLayerSize; j++) {
                    writer.write(hiddenLayer2Weights[i][j] + " ");
                }
                writer.newLine();
            }
            for (int i = 0; i < hiddenLayerSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    writer.write(outputLayerWeights[i][j] + " ");
                }
                writer.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    boolean loadWeightsFromFileAndModify() {
        File file = new File(weightsFile);
        if (!file.exists()) {
            return false;
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            for (int i = 0; i < inputSize; i++) {
                String[] line = reader.readLine().split(" ");
                for (int j = 0; j < hiddenLayerSize; j++) {
                    hiddenLayer1Weights[i][j] = Double.parseDouble(line[j]);
                }
            }
            for (int i = 0; i < hiddenLayerSize; i++) {
                String[] line = reader.readLine().split(" ");
                for (int j = 0; j < hiddenLayerSize; j++) {
                    hiddenLayer2Weights[i][j] = Double.parseDouble(line[j]);
                }
            }
            for (int i = 0; i < hiddenLayerSize; i++) {
                String[] line = reader.readLine().split(" ");
                for (int j = 0; j < outputSize; j++) {
                    outputLayerWeights[i][j] = Double.parseDouble(line[j]);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }

        // Изменение 5% весов случайным образом
        Random random = new Random();
        double percentageToModify = 0.05;

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenLayerSize; j++) {
                if (random.nextDouble() < percentageToModify) {
                    hiddenLayer1Weights[i][j] *= (1 + (random.nextDouble() * 2 - 1) * percentageToModify);
                }
            }
        }

        for (int i = 0; i < hiddenLayerSize; i++) {
            for (int j = 0; j < hiddenLayerSize; j++) {
                if (random.nextDouble() < percentageToModify) {
                    hiddenLayer2Weights[i][j] *= (1 + (random.nextDouble() * 2 - 1) * percentageToModify);
                }
            }
        }

        for (int i = 0; i < hiddenLayerSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                if (random.nextDouble() < percentageToModify) {
                    outputLayerWeights[i][j] *= (1 + (random.nextDouble() * 2 - 1) * percentageToModify);
                }
            }
        }

        return true;
    }

    private void initializeWeights() {
        Random random = new Random();
        initializeLayerWeights(hiddenLayer1Weights, random);
        initializeLayerWeights(hiddenLayer2Weights, random);
        initializeLayerWeights(outputLayerWeights, random);
    }

    private void initializeLayerWeights(double[][] layerWeights, Random random) {
        for (int i = 0; i < layerWeights.length; i++) {
            for (int j = 0; j < layerWeights[i].length; j++) {
                layerWeights[i][j] = random.nextDouble() * 2 - 1;
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

    private double[] applyActivationFunction(double[] input) {
        double[] result = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            result[i] = 1 / (1 + Math.exp(-input[i]));
        }
        return result;
    }

}