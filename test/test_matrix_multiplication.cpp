#include "matrix_multiplication.h"
#include "matrix_multiplication_without_errors.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

// ######################### Source code of multiplyMatrices in src/matrix_mult


TEST(MatrixMultiplicationTest, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    std::vector<std::vector<int>> B = {
        {7, 8},
        {9, 10},
        {11, 12}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 2, 3, 2);

    std::vector<std::vector<int>> expected = {
        {58, 64},
        {139, 154}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(MatrixMultiplicationTest, TestMultiplyEmptyMatrices) {
    std::vector<std::vector<int>> A(0, std::vector<int>(0, 0));
    std::vector<std::vector<int>> B(0, std::vector<int>(0, 0));
    std::vector<std::vector<int>> C(0, std::vector<int>(0, 0));

    multiplyMatrices(A, B, C, 0, 0, 0);

    std::vector<std::vector<int>> expected(0, std::vector<int>(0, 0));

    ASSERT_EQ(C, expected) << "Result incorrect.";
}

TEST(MatrixMultiplicationTest, TestMultiplySingleElementMatrices) {
    std::vector<std::vector<int>> A(1, std::vector<int>(1, 6));
    std::vector<std::vector<int>> B(1, std::vector<int>(1, 7));
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected(1, std::vector<int>(1, 42));

    ASSERT_EQ(C, expected) << "Result incorrect.";
}

TEST(MatrixMultiplicationTest, TestMultiplyMatricesWithNegativeNumbers) {
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, -6}
    };
    std::vector<std::vector<int>> B = {
        {-7, 8},
        {9, -10},
        {11, 12}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 2, 3, 2);

    std::vector<std::vector<int>> expected = {
        {44, 24},
        {-49, -90}
    };

    ASSERT_EQ(C, expected) << "Result incorrect.";
}

TEST(MatrixMultiplicationTest, TestMultiplyVectors) {
    std::vector<std::vector<int>> A = {
        {1, 2, 3}
    };
    std::vector<std::vector<int>> B = {
        {7},
        {8},
        {9}
    };
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 1, 3, 1);

    std::vector<std::vector<int>> expected = {
        {50}
    };

    ASSERT_EQ(C, expected) << "Result incorrect.";
}

TEST(MatrixMultiplicationTest, TestMultiplyBigMatrices) {
    // 200x200 matrix with ordered numbers from 1 to 40000
    std::vector<std::vector<int>> A(200, std::vector<int>(200, 0));
    int count = 1;
    for (int i = 0; i < 200; ++i) {
        for (int j = 0; j < 200; ++j) {
            A[i][j] = count++;
        }
    }

    // 200x200 identity matrix
    std::vector<std::vector<int>> B(200, std::vector<int>(200, 0));
    for (int i = 0; i < 200; ++i) {
        B[i][i] = 1;
    }

    std::vector<std::vector<int>> C(200, std::vector<int>(200, 0));
    multiplyMatrices(A, B, C, 200, 200, 200);

    // The result should be the same as A
    ASSERT_EQ(C, A) << "Result incorrect.";
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
