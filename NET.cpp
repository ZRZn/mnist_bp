//
// Created by ZangRuozhou on 2017/12/30.
//

#include <algorithm>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>


#include "NET.hpp"

using namespace std;
namespace NN {

    static int reverseInt(int i)
    {
        unsigned char ch1, ch2, ch3, ch4;
        ch1 = i & 255;
        ch2 = (i >> 8) & 255;
        ch3 = (i >> 16) & 255;
        ch4 = (i >> 24) & 255;
        return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
    }

    NET::NET() {
        train_image = NULL;
        train_label = NULL;
        test_image = NULL;
        test_label = NULL;
    }

    NET::~NET() {
        //释放数据内存
        if (train_image != NULL) delete[](train_image);
        if (train_label != NULL) delete[](train_label);
        if (test_image != NULL) delete[](test_image);
        if (test_label != NULL) delete[](test_label);
    }

    void NET::init() {
        train_image = new int[train_data_num * input_num];
        memset(train_image, 0, sizeof(int) * train_data_num * input_num);
        train_label = new int[train_data_num * output_num];
        memset(train_label, 0, sizeof(int) * train_data_num * output_num);
        test_image = new int[test_data_num * input_num];
        memset(test_image, 0, sizeof(int) * test_data_num * input_num);
        test_label = new int[test_data_num * output_num];
        memset(test_label, 0, sizeof(int) * test_data_num * output_num);

        initWeightData();
        getMnistData("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
    }

    void NET::readMnistImages(string filename, int* data_set, int num) {
        ifstream file(mnist_path + filename, ios::binary);
        assert(file.is_open());

        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        assert(number_of_images == num);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
        assert(n_rows == image_height && n_cols == image_width);

        for (int i = 0; i < number_of_images; ++i) {
            for (int r = 0; r < n_rows; ++r) {
                for (int c = 0; c < n_cols; ++c) {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    //归一化
                    if (temp > 128) {
                        data_set[i * input_num + r * n_cols + c] = 1;
                    } else {
                        data_set[i * input_num + r * n_cols + c] = 0;
                    }
                }
            }
        }
    }

    void NET::readMnistLabels(string filename, int* data_set, int num) {
        ifstream file(mnist_path + filename, ios::binary);
        assert(file.is_open());
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        assert(number_of_images == num);

        for (int i = 0; i < number_of_images; ++i) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            data_set[i * output_num + temp] = 1;
        }
    }

    void NET::getMnistData(string train_image_name, string train_label_name, string test_image_name, string test_label_name) {
        assert(train_image && train_label && test_image && test_label);
        readMnistImages(train_image_name, train_image, train_data_num);
        readMnistLabels(train_label_name, train_label, train_data_num);
        readMnistImages(test_image_name, test_image, test_data_num);
        readMnistLabels(test_label_name, test_label, test_data_num);
    }


    void NET::initWeightData() {
        //设置时间种子
        srand(time(0) + rand());

        //初始化权重和阈值均为[-1,1]之间的实数
        for (int i = 0; i < input_num; i++) {
            for (int j = 0; j < hidden_num; j++) {
                weight_input2hidden[i][j] = -1 + 2 * ((float)rand()) / RAND_MAX;
            }
        }

        for (int i = 0; i < hidden_num; i++) {
            for (int j = 0; j < output_num; j++) {
                weight_hidden2output[i][j] = -1 + 2 * ((float)rand()) / RAND_MAX;
            }
        }

        for (int i = 0; i < hidden_num; i++) {
            threshold_hidden[i] = -1 + 2 * ((float)rand()) / RAND_MAX;
        }

        for (int i = 0; i < output_num; i++) {
            threshold_output[i] = -1 + 2 * ((float)rand()) / RAND_MAX;
        }
    }

    float NET::calSigmod(float x) {
        return 1.0 / (1.0 + exp(-x));
    }

    void NET::calForward(int* image) {
        //计算隐层输出值,f(x) = ∑w·x - b
        for (int i = 0; i < hidden_num; i++) {
            float sum = 0;
            for (int j = 0; j < input_num; j++) {
                sum += image[j] * weight_input2hidden[j][i];
            }
            result_hidden[i] = calSigmod(sum - threshold_hidden[i]);
        }

        //计算输出层输出值,f(x) = ∑w·x - b
        for (int i = 0; i < output_num; i++) {
            float sum = 0;
            for (int j = 0; j < hidden_num; j++) {
                sum += result_hidden[j] * weight_hidden2output[j][i];
            }
            result_output[i] = calSigmod(sum - threshold_output[i]);
        }
    }

    void NET::calDiff(int* label) {
        //计算输出层差值，diff_output=(y-y')*f'(x),根据sigmod性质，f'(x)=f(x)*(1-f(x))
        for (int i = 0; i < output_num; i++) {
            diff_output[i] = (label[i] - result_output[i])*(result_output[i] * (1.0 - result_output[i]));
        }

        //计算隐层差值，diff_hidden=(∑(w·diff_output))*f'(x),根据sigmod性质，f'(x)=f(x)*(1-f(x))
        for (int i = 0; i < hidden_num; i++) {
            float sum = 0;
            for (int j = 0; j < output_num; j++) {
                sum += weight_hidden2output[i][j] * diff_output[j];
            }
            diff_hidden[i] = sum * (result_hidden[i] * (1.0 - result_hidden[i]));
        }
    }

    void NET::updateWeight(int* image) {
        //更新输出层的权值和阈值, △w = n × diff × 隐层输出值，n为学习率。
        for (int i = 0; i < output_num; i++) {
            for (int j = 0; j < hidden_num; j++) {
                weight_hidden2output[j][i] += output2hidden_rate * diff_output[i] * result_hidden[j];
            }
            threshold_output[i] += output2hidden_rate * diff_output[i];
        }

        //更新隐层的权值和阈值, △w = n × diff × 输入值，n为学习率。
        for (int i = 0; i < hidden_num; i++) {
            for (int j = 0; j < input_num; j++) {
                weight_input2hidden[j][i] += hidden2input_rate * diff_hidden[i] * image[j];
            }
            threshold_hidden[i] += hidden2input_rate * diff_hidden[i];
        }
    }

    float NET::calAccuracy() {
        int accuracy_sum = 0;
        for (int i = 0; i < test_data_num; i++) {
            int* point_image = test_image + i * input_num;
            //前向计算得到结果
            calForward(point_image);
            float max = 0.0;
            int index = 0;
            for (int j = 0; j < output_num; j++) {
                if (result_output[j] > max) {
                    max = result_output[j];
                    index = j;
                }
            }
            int* point_label = test_label + i * output_num;
            if (point_label[index] == 1) accuracy_sum++;
        }
        return (accuracy_sum * 1.0 /test_data_num);
    }

    void NET::train() {
        for (int i = 0; i < 5; i++) {
            cout << "迭代轮数 : " << i;

            for (int j = 0; j < train_data_num; j++) {
                int* point_image = train_image + j * input_num;
                calForward(point_image);
                int* point_label = train_label + j * output_num;
                calDiff(point_label);
                int* point_update = train_image + j * input_num;
                updateWeight(point_update);
            }

            float accuracy = calAccuracy();
            cout << ",    accuray: " << accuracy << std::endl;
        }
    }
}