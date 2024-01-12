import java.awt.Graphics;
import java.awt.Image;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;/
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import javax.swing.JFrame;

public class  Autoencoder {
	
	// Определение класса ErrorPlotter
    public static class ErrorPlotter extends JFrame {
        private List<Double> errors;

        public ErrorPlotter() {
            super("Error Plot");
            this.errors = new ArrayList<>();
            setSize(800, 600);
            setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        }

        public void addError(double error) {
            errors.add(error);
            repaint();
        }

        @Override
        public void paint(Graphics g) {
            // Используем двойную буферизацию для устранения мерцания
            Image offscreenImage = createImage(getWidth(), getHeight());
            Graphics offscreenGraphics = offscreenImage.getGraphics();
            paintOffscreen(offscreenGraphics);
            g.drawImage(offscreenImage, 0, 0, this);
        }

        private void paintOffscreen(Graphics g) {
            super.paint(g);
            if (errors.isEmpty()) return;

            int width = getWidth();
            int height = getHeight();
            int padding = 50;
            int maxErrorPlotHeight = height - 2 * padding;
            double maxError = errors.stream().max(Double::compare).orElse(1.0);

            // Рисуем оси
            g.drawLine(padding, padding, padding, height - padding);
            g.drawLine(padding, height - padding, width - padding, height - padding);

            // Рисуем числовые обозначения ошибки
            for (int i = 0; i <= 10; i++) {
                int y = height - padding - (i * maxErrorPlotHeight / 10);
                g.drawString(String.format("%.2f", maxError * i / 10), 5, y);
            }

            // Рисуем ломаную линию
            int prevX = padding, prevY = height - padding;
            int x, y;
            for (int i = 0; i < errors.size(); i++) {
                x = padding + i * (width - 2 * padding) / errors.size();
                y = (int) ((height - padding) - (errors.get(i) / maxError) * maxErrorPlotHeight);
                if (i > 0) {
                    g.drawLine(prevX, prevY, x, y);
                }
                prevX = x;
                prevY = y;
            }
        }
    }

    public static double[] stringToDoubleArray(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        double[] bitArray = new double[bytes.length * 8];
        for (int i = 0; i < bytes.length; i++) {
            for (int j = 0; j < 8; j++) {
                bitArray[i * 8 + j] = (bytes[i] >> (7 - j)) & 1;
            }
        }
        return bitArray;
    }
    public static double[][] stringToBitArray(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        int bitsLength = bytes.length * 8;
        int arraySize = (int) Math.ceil((double) bitsLength / 256);
        double[][] bitArrays = new double[arraySize][256];

        for (int i = 0; i < bytes.length; i++) {
            for (int j = 0; j < 8; j++) {
                int bitIndex = i * 8 + j;
                bitArrays[bitIndex / 256][bitIndex % 256] = (bytes[i] >> (7 - j)) & 1;
            }
        }

        return bitArrays;
    }

    public static String bitArrayToString(double[][] bitArrays) {
        int bitsLength = bitArrays.length * 256;
        byte[] bytes = new byte[bitsLength / 8];

        for (int i = 0; i < bitsLength; i++) {
            if (bitArrays[i / 256][i % 256] >= 0.5) {
                bytes[i / 8] |= 1 << (7 - (i % 8));
            }
        }

        return new String(bytes, StandardCharsets.UTF_8).trim();
    }
    public static String doubleArrayToString(double[] bitArray) {
        if (bitArray.length % 8 != 0) {
            throw new IllegalArgumentException("Invalid bit array length");
        }
        byte[] bytes = new byte[bitArray.length / 8];
        for (int i = 0; i < bytes.length; i++) {
            for (int j = 0; j < 8; j++) {
                bytes[i] |= (bitArray[i * 8 + j] >= 0.5 ? 1 : 0) << (7 - j);
            }
        }
        return new String(bytes, StandardCharsets.UTF_8);
    }
    
 // Добавьте этот метод в класс SimpleNeuralNetwork

    public static double[] outputFromLatentVector(double[] latentVector) {
    	int inputSize = 256;
        int[] layerSizes = new int[]{200, inputSize};

        SimpleNeuralNetwork network = new SimpleNeuralNetwork(inputSize, layerSizes);

        // Загрузите веса сети, если они были сохранены ранее
        network.loadWeightsFromFile();

        // Вычисляем выход на основе латентного вектора
        double[] output = network.predictFromLatentVector(latentVector);

        return output;
    
    }

    

    public static void main(String[] args) throws IOException, InterruptedException {
    	 
        ErrorPlotter plotter = new ErrorPlotter();
        plotter.setVisible(true);
   
         
         int inputSize = 256;
         int[] layerSizes = new int[]{200, inputSize};
         
         SimpleNeuralNetwork network = new SimpleNeuralNetwork(inputSize, layerSizes);
          	  // 4. Задаем количество эпох и скорость обучения.
        int epochs = 10000000;
        double learningRate = 0.01;
        
        String filePath = "dataset.txt";
        String text = Files.readString(Paths.get(filePath), StandardCharsets.UTF_8);
        
        String testFilePath = "test_dataset.txt";
        String testText = Files.readString(Paths.get(testFilePath), StandardCharsets.UTF_8);
        double[][] testBitArrays = stringToBitArray(testText);

        double[][] bitArrays = stringToBitArray(text);
        String restoredText = bitArrayToString(bitArrays);
        
        double[][] inputs = bitArrays;

 
        double[][] targetOutputs = bitArrays;

      
        for (int epoch = 0; epoch < epochs; epoch++) {
        	  ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

            double totalError = 0;
            for (int i = 0; i < inputs.length; i++) {
                final int index = i;
                executor.submit(() -> {
                    // Здесь код обучения для каждого примера
                    network.train(inputs[index], targetOutputs[index], learningRate);
                });
            }

            // Закрываем ExecutorService и ждем окончания всех задач
            executor.shutdown();
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);

            // После обучения всех примеров, можно вычислить ошибку на тестовом наборе
            double testTotalError = 0;
            for (double[] testInput : testBitArrays) {
                double[] predictedOutput = network.predict(testInput);
                testTotalError += calculateError(testInput, predictedOutput);
                //System.out.println(doubleArrayToString(predictedOutput));
                System.out.println(doubleArrayToString(outputFromLatentVector(network.getLayerOutputs(testInput,0))));
                // Остальной код для тестирования
            }
            double testAverageError = testTotalError / testBitArrays.length;
            plotter.addError(testAverageError);
            System.out.println("Epoch " + epoch + ": Test Average Error = " + testAverageError);
            
            if ((epoch + 1) % 100 == 0) {
                network.saveWeightsToFile();
               // System.out.println("Weights saved at epoch " + (epoch + 1));
            }
        
    }

        
    }

    // Метод для расчета ошибки (например, MSE)
    private static double calculateError(double[] expected, double[] actual) {
        double sumError = 0;
        for (int i = 0; i < expected.length; i++) {
            sumError += Math.pow(expected[i] - actual[i], 2);
        }
        return sumError / expected.length;
    }	
     

  
    
}
