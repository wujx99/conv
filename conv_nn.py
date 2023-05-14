import numpy as np

#装饰器
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

def conv(input_image, cnn_kernel):
    """
    input_image,cnn_kernel,均是方阵;
    测试为正确的
    """
    image_size = input_image.shape[0]
    kernel_size = cnn_kernel.shape[0]
    result_size = image_size - kernel_size + 1

    result = np.ones((result_size, result_size))
    for i in range(result_size):
        for j in range(result_size):
            result[i][j] = np.sum(input_image[i:i+kernel_size,j:j+kernel_size] * cnn_kernel)
    return result
def pool(input_image, pool_kernel):
    """
    input_image,pool_kernel必须有特定的维数，这里确定pool_kernel维数为2*2
    测试为正确

    """
    image_size = input_image.shape[0]
    kernel_size = pool_kernel.shape[0]
    result_size = image_size // 2  #要求必须能够整除,并且不能是float

    result = np.ones((result_size, result_size))
    for i in range(result_size):
        for j in range(result_size):
            result[i][j] = np.sum(input_image[i:i+kernel_size,j:j+kernel_size] * pool_kernel)
    return result
class ConvNeuralNetwork:
    #应为要对图像进行卷积。需要重新设置参数
    def __init__(self,
                 no_of_cnn_kernels,
                 kernel_size,
                 image_size,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate,
                 pool_kernel_size = 2, #默认为2，并且本程序不会修改
                ):
        #创建kernel weights的参数
        self.no_of_cnn_kernels = no_of_cnn_kernels
        self.kernel_size = kernel_size

        self.pool_kernel_size = pool_kernel_size
        #创建池化后的weight的参数
        self.no_of_in_nodes = no_of_cnn_kernels * ((image_size - kernel_size + 1)/2)**2
        self.no_of_out_nodes = no_of_out_nodes

        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        #self.weigth_test()

        #这里包括创建说有的weight_matrices,包括cnn_kenels
        self.create_weight_matrices()
  
    #测试用的weight的方法
    def weigth_test(self):
        self.wih = [[0.1, 0.3],[0.3,0.4]]
        self.who = [[0.4,0.6]]

    def create_weight_matrices(self):
        """
        初始化cnn的权重矩阵
        """
        #创建kernel的weight，但是不知道保存成什么样的维数。
        self.w_kernels = np.random.normal(0, 1,(self.no_of_cnn_kernels,self.kernel_size,self.kernel_size))
        
        self.wih = np.random.normal(0, 1, size=(self.no_of_hidden_nodes,self.no_of_in_nodes))
        self.who = np.random.normal(0, 1, size=(self.no_of_out_nodes, self.no_of_hidden_nodes))

    def v_z_init_zero(self):
        v_ih = np.zeros((self.no_of_hidden_nodes, self.no_of_in_nodes))
        v_ho = np.zeros((self.no_of_out_nodes, self.no_of_hidden_nodes))

        v = [v_ih, v_ho]
        z = [v_ih, v_ho]

        return (z, v)

    def train(self, input_image, target_vector): #input_vector这里改为input_image
        """
        input_image is a n*n ndarray
        target_vector can be tuple, list or ndarray
        """
        
        #[1,2,3] --->[[1,2,3]]---->[[1],[2],[3]]
        target_vector = np.array(target_vector, ndmin=2).T

        #np.dot意义比较丰富
        #卷积过程（需要经过sigmoid函数）
        cnn_images = []
        for i in range(self.no_of_cnn_kernels):
            temp_image = conv(input_image, self.w_kernels[i])
            #增加激活函数
            result_image = activation_function(temp_image)
            cnn_images.append(result_image)
        
        #池化过程(平均池化)
        pool_kernel = 1/2 *np.ones((self.pool_kernel_size,self.pool_kernel_size))
        for i in range(self.no_of_cnn_kernels):
            cnn_images[i] = pool(cnn_images[i], pool_kernel)

        #把输出拉直,拉成（-1，1）与target_vector保持一致
        input_vector = np.reshape(cnn_images,(-1,1))  

        output_vector1 = np.dot(self.wih, input_vector)
        output_hidden = activation_function(output_vector1)

        output_vector2 = np.dot(self.who, output_hidden)
        output_network = activation_function(output_vector2)

        #前向计算完成，开始反向更新weight
        loss_errors = target_vector - output_network

        # 计算 output errors
        output_errors = loss_errors * output_network * (1.0 - output_network)
        # 更新 weights:
        self.who += self.learning_rate * np.dot(output_errors, output_hidden.T)

        # 计算 hidden errors:
        hidden_errors = output_hidden * (1.0 - output_hidden) * np.dot(self.who.T, output_errors)
        # 更新 weights:
        self.wih += self.learning_rate * np.dot(hidden_errors, input_vector.T)
        
        #计算pooling errors

        #计算conv errors

        #更新 weights


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
        #这里需要重写一下！
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