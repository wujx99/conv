import numpy as np

#装饰器
@np.vectorize
def sigmoid(inx):
    if inx>=0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0/(1+np.exp(-inx))
    else:
        return np.exp(inx)/(1+np.exp(inx))

#可以根据需求更改激活函数
activation_function = sigmoid

def conv(input_image, cnn_kernel):
    """
    input_image,cnn_kernel,均是方阵和ndarray;
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
    input_image为卷积后的图像,pool_kernel必须有特定的维数，这里确定pool_kernel维数为2*2
    """
    image_size = input_image.shape[0]
    kernel_size = pool_kernel.shape[0]
    result_size = image_size // 2  #要求必须能够整除,并且不能是float

    result = np.ones((result_size, result_size))
    for i in range(result_size):
        for j in range(result_size):

            result[i][j] = np.sum(input_image[i*kernel_size:(i+1)*kernel_size,j*kernel_size:(j+1)*kernel_size] * pool_kernel)
    return result

def reverse_pool(input_image, pool_kernel):
    """
    input_image,pool_kernel必须有特定的维数，这里确定pool_kernel维数为2*2
    给mean_pool返回误差项，其实是完成了一个kornecker积

    """
    return np.kron(input_image, pool_kernel)
class ConvNeuralNetwork:
    #应为要对图像进行卷积。需要重新设置参数
    def __init__(self,
                 no_of_cnn_kernels:int,
                 conv_kernel_size:int,
                 image_size:int,
                 no_of_out_nodes:int,
                 no_of_hidden_nodes:int,
                 learning_rate,                 
                ):
        self.image_size = image_size
        #创建kernel weights的参数
        self.no_of_cnn_kernels = no_of_cnn_kernels
        self.conv_kernel_size = conv_kernel_size

        #创建池化后的weight的参数
        self.no_of_in_nodes = no_of_cnn_kernels * 100  #注意这里针对的是28*28，且经过9*9的conv_kernel，2*2的pooling_kernel
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
        self.w_kernels = np.random.normal(0, 1,size=(self.no_of_cnn_kernels,self.conv_kernel_size,self.conv_kernel_size))
        
        self.wih = np.random.normal(0, 1, size=(self.no_of_hidden_nodes,self.no_of_in_nodes))
        self.who = np.random.normal(0, 1, size=(self.no_of_out_nodes, self.no_of_hidden_nodes))

    def v_z_init_zero(self):
        v_kern = np.zeros((self.no_of_cnn_kernels,self.conv_kernel_size,self.conv_kernel_size))
        v_ih = np.zeros((self.no_of_hidden_nodes, self.no_of_in_nodes))
        v_ho = np.zeros((self.no_of_out_nodes, self.no_of_hidden_nodes))

        v = [v_kern,v_ih, v_ho]
        z = [v_kern,v_ih, v_ho]

        return (z, v)

    def train(self, input_vector_s, target_vector_s, image_size = 28,pool_kernel_size = 2): #input_vector这里改为input_image,pool_kernel_size默认为2，并且本程序不会修改
        """
        实现batch的梯度下降
        """       
        d_w_kernels = np.zeros((self.no_of_cnn_kernels,self.conv_kernel_size,self.conv_kernel_size))
        d_wih = np.zeros((self.no_of_hidden_nodes,self.no_of_in_nodes))
        d_who = np.zeros((self.no_of_out_nodes, self.no_of_hidden_nodes))

        no_of_images = len(input_vector_s)
        #print("no_of_image",no_of_images)
        for i in range(no_of_images):
            input_vector = input_vector_s[i]
            target_vector = target_vector_s[i]

            input_image = np.reshape(input_vector, (image_size, image_size))
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
            pool_kernel = 1/4 *np.ones((pool_kernel_size,pool_kernel_size))
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
            #self.who += self.learning_rate * np.dot(output_errors, output_hidden.T)
            d_who += np.dot(output_errors, output_hidden.T)

            # 计算 hidden errors:
            hidden_errors = output_hidden * (1.0 - output_hidden) * np.dot(self.who.T, output_errors)
            # 更新 weights:
            #self.wih += self.learning_rate * np.dot(hidden_errors, input_vector.T)
            d_wih += np.dot(hidden_errors, input_vector.T)

            #计算conv_errors
            input_errors = np.dot(self.wih.T, hidden_errors) #拉直层的导数
            result_shape = np.array(cnn_images).shape #pooling后的shape
            temp_errors = np.reshape(input_errors, result_shape)
            conv_errors = []
            
            for i in range(self.no_of_cnn_kernels):
                pooling_error = reverse_pool(temp_errors[i], pool_kernel) #pool求导
                temp = pooling_error * (1.0 - pooling_error)  #卷积后的sigmoid激活函数
                conv_errors.append(temp)

            #更新weights
            for i in range(self.no_of_cnn_kernels):
                #self.w_kernels[i] += self.learning_rate * conv(input_image, conv_errors[i])
                d_w_kernels[i] += conv(input_image, conv_errors[i])

        self.who += self.learning_rate * (d_who / no_of_images)
        self.wih += self.learning_rate * (d_wih / no_of_images)
        self.w_kernels += self.learning_rate * (d_w_kernels / no_of_images)


    def train_nn_regular(self, input_vector, target_vector, regular_term):
        """
        这是nn的，保存是为了以后的更改方便
        z, v数组的顺序是由input---->output
        """
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

    def train_admm(self, input_vector, target_vector, regular_term, z, v, image_size = 28,pool_kernel_size = 2):
        """
         z, v数组的顺序是由input---->output,其中v是对偶变量
        """
       
        input_image = np.reshape(input_vector, (image_size, image_size))
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
        pool_kernel = 1/4 *np.ones((pool_kernel_size,pool_kernel_size))
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

        # 计算hidden errors
        output_errors = loss_errors * output_network * (1.0 - output_network)
        # 更新 weights,加入admm的正则项
        self.who =self.who + self.learning_rate * (np.dot(output_errors, output_hidden.T)
                                                   - regular_term * (self.who - z[2] + 1.0 / regular_term * v[2]))

        # 计算hidden errors:
        hidden_errors = output_hidden * (1.0 - output_hidden) * np.dot(self.who.T, output_errors)
        # 更新weights，并加入admm正则项
        self.wih = self.wih +  self.learning_rate * (np.dot(hidden_errors, input_vector.T)
                                                     - regular_term * (self.wih - z[1] + 1.0 / regular_term * v[1]))

        #计算conv_errors
        input_errors = np.dot(self.wih.T, hidden_errors) #拉直层的导数
        result_shape = np.array(cnn_images).shape #pooling后的shape
        temp_errors = np.reshape(input_errors, result_shape)
        conv_errors = []
        
        for i in range(self.no_of_cnn_kernels):
            pooling_error = reverse_pool(temp_errors[i], pool_kernel) #pool求导
            temp = pooling_error * (1.0 - pooling_error)  #卷积后的sigmoid激活函数
            conv_errors.append(temp)

        #更新weights,并加入admm正则项
        for i in range(self.no_of_cnn_kernels):
            self.w_kernels[i] = self.w_kernels[i] + self.learning_rate * (conv(input_image, conv_errors[i]) 
                                - regular_term * (self.w_kernels[i] - z[0][i] + 1.0 /regular_term * v[0][i]))
        
    def run(self, input_vector, image_size = 28,pool_kernel_size = 2):
        """
        网络结构不同，这里需要重写一下！
        input_vector must be ndarray
        """
        input_image = np.reshape(input_vector, (image_size, image_size))
        
        #卷积过程（需要经过sigmoid函数）
        cnn_images = []
        for i in range(self.no_of_cnn_kernels):
            temp_image = conv(input_image, self.w_kernels[i])
            #增加激活函数
            result_image = activation_function(temp_image)
            cnn_images.append(result_image)
        
        #池化过程(平均池化)
        pool_kernel = 1/4 *np.ones((pool_kernel_size,pool_kernel_size))
        for i in range(self.no_of_cnn_kernels):
            cnn_images[i] = pool(cnn_images[i], pool_kernel)

        #把输出拉直,拉成（-1，1）与target_vector保持一致
        input_vector = np.reshape(cnn_images,(-1,1))  

        output_vector1 = np.dot(self.wih, input_vector)
        output_hidden = activation_function(output_vector1)

        output_vector2 = np.dot(self.who, output_hidden)
        output_network = activation_function(output_vector2)
        return output_network


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