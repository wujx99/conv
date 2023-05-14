import numpy as np

@np.vectorize
def sigmoid(inx):
    if inx>=0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0/(1+np.exp(-inx))
    else:
        return np.exp(inx)/(1+np.exp(inx))

# def sigmoid(x):
#     return 1 / (1 + np.e ** -x)
#激活函数
activation_function = sigmoid

class NeuralNetwork:

    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        #self.weigth_test()
        self.create_weight_matrices()
    #测试用的weight的方法
    def weigth_test(self):
        self.wih = [[0.1, 0.3],[0.3,0.4]]
        self.who = [[0.4,0.6]]

    def create_weight_matrices(self):
        """
        A method to initialize the weight
        matrices of the neural network
        """
        self.wih = np.random.normal(0, 1, size=(self.no_of_hidden_nodes,self.no_of_in_nodes))
        self.who = np.random.normal(0, 1, size=(self.no_of_out_nodes, self.no_of_hidden_nodes))

    # def v_z_init(self):
        # rad = 1 / np.sqrt(self.no_of_in_nodes)
        # X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        # v_ih_weight = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes))
        #
        # rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        # X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        # v_ho_weight = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes))
        #
        # v = [v_ih_weight, v_ho_weight]
        # z = [v_ih_weight, v_ho_weight]

        # return (z, v)
    def v_z_init_zero(self):
        v_ih = np.zeros((self.no_of_hidden_nodes, self.no_of_in_nodes))
        v_ho = np.zeros((self.no_of_out_nodes, self.no_of_hidden_nodes))

        v = [v_ih, v_ho]
        z = [v_ih, v_ho]

        return (z, v)
    def train(self, input_vector, target_vector):
        """
        input_vector and target_vector can
        be tuple, list or ndarray
        """
        #[1,2,3] --->[[1,2,3]]---->[[1],[2],[3]]
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        #np.dot意义比较丰富
        output_vector1 = np.dot(self.wih, input_vector)
        output_hidden = activation_function(output_vector1)

        output_vector2 = np.dot(self.who, output_hidden)
        output_network = activation_function(output_vector2)

        output_errors = target_vector - output_network

        # update the weights:
        tmp = output_errors * output_network * (1.0 - output_network)

        tmp = self.learning_rate * np.dot(tmp, output_hidden.T)

        self.who += tmp

        # calculate hidden errors:
        hidden_errors = np.dot(self.who.T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_hidden * (1.0 - output_hidden)

        self.wih += self.learning_rate * np.dot(tmp, input_vector.T)

    def train_ture(self, input_vector, target_vector):
        """
        input_vector and target_vector can
        be tuple, list or ndarray
        """
        #[1,2,3] --->[[1,2,3]]---->[[1],[2],[3]]
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        #np.dot意义比较丰富
        output_vector1 = np.dot(self.wih, input_vector)
        output_hidden = activation_function(output_vector1)

        output_vector2 = np.dot(self.who, output_hidden)
        output_network = activation_function(output_vector2)

        loss_errors = target_vector - output_network

        # calculate hidden errors
        output_errors = loss_errors * output_network * (1.0 - output_network)
        # update the weights:  
        self.who += self.learning_rate * np.dot(output_errors, output_hidden.T)

        # calculate hidden errors:
        hidden_errors = output_hidden * (1.0 - output_hidden) * np.dot(self.who.T, output_errors)
        # update the weights:
        self.wih += self.learning_rate * np.dot(hidden_errors, input_vector.T)


    def train_regular(self, input_vector, target_vector, regular_term):
        #z, v数组的顺序是由input---->output
        # [1,2,3] --->[[1,2,3]]---->[[1],[2],[3]]
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        # np.dot意义比较丰富
        output_vector1 = np.dot(self.wih, input_vector)
        output_hidden = activation_function(output_vector1)

        output_vector2 = np.dot(self.who, output_hidden)
        output_network = activation_function(output_vector2)

        loss_errors = target_vector - output_network

        # calculate hidden errors
        output_errors = loss_errors * output_network * (1.0 - output_network)
        # update the weights:
        #加入admm的正则项
        self.who = self.who + self.learning_rate * (np.dot(output_errors, output_hidden.T)
                                                   - regular_term * self.who)

        # calculate hidden errors:
        hidden_errors = output_hidden * (1.0 - output_hidden) * np.dot(self.who.T, output_errors)
        # update the weights:
        #加入admm正则项
        self.wih = self.wih +  self.learning_rate * (np.dot(hidden_errors, input_vector.T)
                                                     -regular_term * self.wih)

    def train_ture_admm(self, input_vector, target_vector, regular_term, z, v):
        #z, v数组的顺序是由input---->output
        # [1,2,3] --->[[1,2,3]]---->[[1],[2],[3]]
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        # np.dot意义比较丰富
        output_vector1 = np.dot(self.wih, input_vector)
        output_hidden = activation_function(output_vector1)

        output_vector2 = np.dot(self.who, output_hidden)
        output_network = activation_function(output_vector2)

        loss_errors = target_vector - output_network

        # calculate hidden errors
        output_errors = loss_errors * output_network * (1.0 - output_network)
        # update the weights:
        #加入admm的正则项
        self.who =self.who + self.learning_rate * (np.dot(output_errors, output_hidden.T)
                                                   - regular_term * (self.who - z[1] + 1.0 / regular_term * v[1]))

        # calculate hidden errors:
        hidden_errors = output_hidden * (1.0 - output_hidden) * np.dot(self.who.T, output_errors)
        # update the weights:
        #加入admm正则项
        self.wih = self.wih +  self.learning_rate * (np.dot(hidden_errors, input_vector.T)
                                                     -regular_term * (self.wih - z[0] + 1.0 / regular_term * v[0]))

    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T

        output_vector = np.dot(self.wih,
                               input_vector)
        output_vector = activation_function(output_vector)

        output_vector = np.dot(self.who,
                               output_vector)
        output_vector = activation_function(output_vector)

        return output_vector

    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((10, 10), int)
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            cm[res_max, int(target)] += 1
        return cm

    def precision(self, label, confusion_matrix):
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()

    def recall(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()

    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs