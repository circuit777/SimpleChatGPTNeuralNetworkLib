# SimpleChatGPTNeuralNetworkLib
https://t.me/zaheck

Просто добавьте Main.java к себе в проект и следуйте инструкции снизу.

<img width="156" alt="image" src="https://user-images.githubusercontent.com/113068729/226125958-b19b218a-21ea-4fed-a55b-e0b4585d0826.png">

Весь код от начала до конца написан с помощью ChatGPT GPT-4 и был протестирован на реальных задачах. Поставленные цели выполнены успешно.

<img width="689" alt="image" src="https://user-images.githubusercontent.com/113068729/226125984-2a411e3f-c90d-4e66-a154-2a7745a925d3.png">


Этот код реализует простую feedforward нейронную сеть на Java с двумя скрытыми слоями. Нейронная сеть предназначена для задач обучения и прогнозирования. Веса нейронной сети хранятся в текстовом файле, и программа может загружать, сохранять и изменять эти веса. В качестве функции активации на скрытых слоях и выходном слое используется сигмоидальная функция. Сеть можно обучать с помощью алгоритма обратного распространения ошибки, а прогнозы осуществляются путем подачи входных данных через сеть.

Как запустить мою либу:

    public static void main(String[] args) {
        // 1. Создаем нейронную сеть с 2 входными нейронами, 4 нейронами в каждом скрытом слое (всего 2 слоя) и 1 выходным нейроном.
        Main neuralNetwork = new Main(2, 4, 1);

        // 2. Подготовка обучающих данных (XOR problem).
        // Здесь каждый массив содержит два значения: входные данные для XOR.
        double[][] inputs = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };

        // 3. Подготовка целевых выходных данных, соответствующих обучающим данным.
        // Здесь каждый массив содержит одно значение: результат XOR.
        double[][] targetOutputs = {
                {0},
                {1},
                {1},
                {0}
        };

        // 4. Задаем количество эпох и скорость обучения.
        int epochs = 10000;
        double learningRate = 0.1;

        // 5. Обучение нейронной сети.
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                // Обучаем сеть на каждом обучающем примере с заданным коэффициентом обучения.
                neuralNetwork.train(inputs[i], targetOutputs[i], learningRate);
            }
        }

        // 6. Тестирование обученной нейронной сети.
        for (int i = 0; i < inputs.length; i++) {
            // Получаем предсказания сети для каждого обучающего примера.
            double[] prediction = neuralNetwork.predict(inputs[i]);

            System.out.println("Input: " + Arrays.toString(inputs[i]) +
                    ", Predicted Output: " + Arrays.toString(prediction) +
                    ", Target Output: " + Arrays.toString(targetOutputs[i]));
        }
    }



