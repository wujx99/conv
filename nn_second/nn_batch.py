import numpy as np

@np.vectorize
def sigmoid(inx):
    if inx>=0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0/(1+np.exp(-inx))
    else:
        return np.exp(inx)/(1+np.exp(inx))

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
        self.create_weight_matrices()
    
    def create_weight_matrices(self):
        """
        A method to initialize the weight
        matrices of the neural network
        """
        self.wih = np.random.normal(0, 1, size=(self.no_of_hidden_nodes,self.no_of_in_nodes))
        self.who = np.random.normal(0, 1, size=(self.no_of_out_nodes, self.no_of_hidden_nodes))

    
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

    def train_ture(self, input_vector_s, target_vector_s):
        """
        实现batch_size的梯度下降
        """
        #print("rukou")
        #记录梯度信息
        d_wih = np.zeros((self.no_of_hidden_nodes,self.no_of_in_nodes))
        d_who = np.zeros((self.no_of_out_nodes, self.no_of_hidden_nodes))

        no_of_images = len(input_vector_s)
        #print("no_of_image",no_of_images)
        for i in range(no_of_images):
            input_vector = input_vector_s[i]
            target_vector = target_vector_s[i]

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
            #self.who += self.learning_rate * np.dot(output_errors, output_hidden.T)
            d_who += np.dot(output_errors, output_hidden.T)

            # calculate hidden errors:
            hidden_errors = output_hidden * (1.0 - output_hidden) * np.dot(self.who.T, output_errors)
            # update the weights:
            #self.wih += self.learning_rate * np.dot(hidden_errors, input_vector.T)
            d_wih += np.dot(hidden_errors, input_vector.T)

            #print("updata")
        self.who += self.learning_rate * (d_who / no_of_images)
        self.wih += self.learning_rate * (d_wih / no_of_images)


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