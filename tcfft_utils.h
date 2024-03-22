#include <iostream>
using namespace std;

// 函数用于计算并输出原矩阵中每个元素在转置矩阵中的新位置
void transposePositions(int m, int n) {
    // 遍历原矩阵的每个元素
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            // 计算原位置
            int originalPos = i * n + j;
            // 计算转置后的新位置
            int newPos = j * m + i;
            // 输出结果
            cout << "原位置: (" << i << ", " << j << ") -> 新位置: " << newPos << endl;
        }
    }
}