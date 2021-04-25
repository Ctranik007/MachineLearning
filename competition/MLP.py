import numpy as np


class MLP:
    def __init__(self, n_i, n_h, n_o, activation='sigmoid', max_epochs=5000,
                 learning_rate=0.1, verbose=2):
        """ 
        Инициализация сети
        :param n_i: Количество нейронов во входном слое
        :param n_h: Количество нейронов в скрытом слое
        :param n_o: Количество нейронов в выходном слое
        :param activation: Функция активации
        :param max_epochs: Максимальное количество итераций для тренировки MLP, для регулировки веса
        :param learning_rate: Указывает какая дельта по весам будет принята
        :param verbose: Сколько деталей должно быть напечатано во время обучения
        """

        self.n_i = n_i
        self.n_h = n_h
        self.n_o = n_o
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.verbose = np.power(10, verbose-1)
        # Инициализация функции активации и ее производных
        self.activation, self.d_activation = \
            self.__initialize_activation_func(activation)

        # Инициализация функции потерь
        self.__initialize_loss_func()

        # Инициализация массива для хранения результатов активации
        self.input = np.ones(self.n_i + 1) 
        self.h = np.ones(self.n_h)
        self.o = np.ones(self.n_o)

        # Инициализация весов нижнего и верхнего слоев
        self.W1, self.W2 = self.__initialize_weights()

    def __initialize_activation_func(self, activation):
        """ 
        Инициализация функции активации и ее производных
        :param activation: Имя функции активации
        :return: Назначенная функция активации и ее производная
        """
        return tanh, d_tanh

    def __initialize_loss_func(self):
        """ 
        Инициализация функции потерь
        - когда n_o > 1,в классификации применяется перекрестная энтропия
        - при n_o = 1, в регрессионной задаче используется квадрат ошибки
        """
        if self.n_o > 1:
            self.loss_func = cross_entropy
        else:
            self.loss_func = squared_error

    def __initialize_weights(self):
        """ 
        Случайная инициализация весов между -1 и 1
        для нижнего и верхнего слоёв
        :return: Инициализированные веса
        """
        w1 = np.random.uniform(-0.2, 0.2, (self.input.size, self.h.size))
        w2 = np.random.uniform(-0.2, 0.2, (self.h.size, self.o.size))
        return w1, w2

    def __forwards(self, inputs):
        """
         Распространение входных данные от входного к выходному слоям
        :param inputs: Обучающий пример
        :return: выходной массив
        """

        # Во входном слое последнее значение равно 1 для смещения
        self.input[:-1] = inputs
        # Активация скрытого слоя
        self.h = self.activation(np.dot(self.input, self.W1))
        if self.n_o > 1:
            # Классификация - использование SoftMax
            self.o = softmax(np.dot(self.h, self.W2))
        else:
            # Регрессия
            self.o = self.activation(np.dot(self.h, self.W2))

        return self.o

    def __backwards(self, expected):
        """ 
        Обучение сети. Сигнал ошибки идёт обратно на каждый слой
        и регулировка веса
        :param expected: Ожидаемые выходные данные для вычисления ошибки
        """
        # Ошибка на выходном слое
        error = expected - self.o
        # Вычисление дельта-активации выходного слоя
        if self.n_o > 1:
            # Классификация
            dz2 = error * d_softmax(self.o, softmax)
        else:
            # Регрессия
            dz2 = error * self.d_activation(self.o)
        # Вычисление дельта-активации скрытого слоя
        dz1 = np.dot(dz2, self.W2.T) * self.d_activation(self.h)

        # Обновление весов
        self.__update_weights(dz1, dz2)

    def __update_weights(self, dz1, dz2):
        """ 
        Обновление весов на нижнем и верхнем слоях
        : :param dz 1: Дельта на скрытом слое
        : :param dz 2: Дельта на выходном слое
        """
        dw1 = np.dot(np.atleast_2d(self.input).T, np.atleast_2d(dz1))
        self.W1 += self.learning_rate * dw1
        dw2 = np.dot(np.atleast_2d(self.h).T, np.atleast_2d(dz2))
        self.W2 += self.learning_rate * dw2

    def fit(self, X, y):
        """ 
        Обучение сети MLP с помощью обучающего набора
        :param X: тренировочный набор
        :param y: выходы обучающего набора
        """
        for e in range(1, self.max_epochs):
            cost = 0.
            for j, row in enumerate(X):
                # переадресация входных данных на выходной уровень
                o = self.__forwards(row)
                # Накапление ошибки каждого вычисленного примера
                # с помощью функции потерь получаем стоимость
                cost += self.loss_func(o, y[j])
                # Обратное распространение вычисленного сигнала ошибки
                # в соответствии с заданным ожидаемым результатом
                self.__backwards(y[j])

        return self

    def predict(self, X):
        """ 
        Предсказание на основе тестовоого набора
        :param X: тестовый набор
        :return: Прогнозируемый результат для тестового набора
        """
        y = list()
        for j, row in enumerate(X):
            if self.n_o > 1:
                # Классификация - использование одной горячей кодировки,
                # находим индекс выходных единиц с максимальным выходом
                y.append(np.argmax(self.__forwards(row)))
            else:
                y.append(self.__forwards(row))
        return np.array(y)


# Определения сигмоидных, производных, SoftMax и функций потерь
def tanh(x):
    """ 
    Масштабированная логистическая сигмоидальная функция
    :param x: Вход должен быть активирован
    :return: соответствующее значение tanh с входным x
    """
    return np.tanh(x)


def d_tanh(x):
    """ 
    Производная функции активации tanh
    :param x: Входное значение
    :возврат: Производная
    """
    return 1.0 - x**2


def softmax(x):
    """
     Вычислите SoftMax вектора x численно устойчивым способом,
    так как exp numpy приведет к бесконечному (nan)
    :param x: При классификации необходимо активировать входы на выходных блоках
    :return: Вероятности на каждом выходном блоке. Сумма должна быть 1.
    """
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def d_softmax(o, f):
    """ 
    Производная SoftMax по отношению к выходу
    :param o: Выход
    :param f: Функция активации выходного слоя - SoftMax в классификации
    :возврат: Производная
    """
    return f(o) * (1 - f(o))


def cross_entropy(o, y):
    """ 
    Функция кросс-энтропийных потерь
    :param o: Выход
    :param y: Ожидаемый результат
    :возврат: Перекрестная энтропия - потеря
    """
    return np.sum(np.nan_to_num(-y * np.log(o) - (1-y) * np.log(1-o)))


def squared_error(o, y):
    """ 
    Квадратная функция потерь ошибок
    :param o: Выход
    :param y: Ожидаемый результат
    :return: The square error - потеря
    """
    return 0.5 * ((y-o) ** 2).sum()