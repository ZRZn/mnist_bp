//
// Created by ZangRuozhou on 2017/12/30.
//

#ifndef MNIST_BP_NET_HPP
#define MNIST_BP_NET_HPP

using namespace std;
namespace NN {

#define mnist_path  "/Users/ZRZn1/Documents/MNIST_data/" //MNIST的数据文件的路径
#define input_num   784 //输入层节点数,28*28
#define image_width      28 //图像宽
#define image_height     28 //图像高
#define hidden_num  120 //隐层节点数
#define output_num  10 //输出层节点数
#define output2hidden_rate   0.8 //输出层至隐层学习率
#define hidden2input_rate    0.6 //隐层至输入层学习率
#define train_data_num   60000 //训练样本数
#define test_data_num    10000 //测试样本数


    class NET {

    public:
        NET(); //构造函数
        ~NET(); //析构函数
        void init(); //初始化数据并分配内存
        void train(); //迭代训练

    //定义全局的所有变量
    private:
        int* train_image; //原始标准输入数据，训练
        int* train_label; //原始标准期望结果，训练
        int* test_image; //原始标准输入数据，测试
        int* test_label; //原始标准期望结果，测试

        float weight_input2hidden[input_num][hidden_num]; //输入层至隐层连接权值
        float weight_hidden2output[hidden_num][output_num]; //隐层至输出层连接权值
        float threshold_hidden[hidden_num]; //隐层阈值
        float threshold_output[output_num]; //输出层阈值
        float result_hidden[hidden_num]; //顺传播，隐层输出值
        float result_output[output_num]; //顺传播，输出层输出值
        float diff_output[output_num]; //逆传播，输出层校正误差
        float diff_hidden[hidden_num]; //逆传播，隐层校正误差

    protected:
        //定义读取mnist训练和测试数据和标签的函数
        void readMnistImages(string filename, int* data_set, int num); //读取图片并对归一化（黑白且强弱一致）
        void readMnistLabels(string filename, int* data_set, int num); //读取标签数据
        void getMnistData(string train_image_name, string train_label_name, string test_image_name, string test_label_name);

        //计算过程
        void initWeightData(); //初始化权重、阈值
        float calSigmod(float x); //计算激活函数，采用sigmod
        void calForward(int* image); //前向计算
        void calDiff(int* label); //反向传播,计算差值
        void updateWeight(int* image); //根据差值更新权值和阈值
        float calAccuracy(); //计算准确率

    };
}

#endif //MNIST_BP_NET_HPP
