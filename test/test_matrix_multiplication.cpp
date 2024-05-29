#include "matrix_multiplication.h"
#include "matrix_multiplication_without_errors.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

// ######################### Source code of multiplyMatrices in src/matrix_mult

//Test provided when cloning the repository.
/*
Error 6: Result matrix contains a number bigger than 100!
Error 12: The number of rows in A is equal to the number of columns in B!
Error 14: The result matrix C has an even number of rows!
Error 20: Number of columns in matrix A is odd!
*/
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

//0x0 matrices
/*
Error 11: Every row in matrix B contains at least one '0'!
SEGFAULT
*/
TEST(MatrixMultiplicationTest, TestMultiplyEmptyMatrices) {
    std::vector<std::vector<int>> A(0, std::vector<int>(0, 0));
    std::vector<std::vector<int>> B(0, std::vector<int>(0, 0));
    std::vector<std::vector<int>> C(0, std::vector<int>(0, 0));

    multiplyMatrices(A, B, C, 0, 0, 0);

    std::vector<std::vector<int>> expected(0, std::vector<int>(0, 0));

    ASSERT_EQ(C, expected) << "Result incorrect.";
}

//1x1 matrices
/*
Error 12: The number of rows in A is equal to the number of columns in B!
Error 18: Matrix A is a square matrix!
Error 20: Number of columns in matrix A is odd!
*/
TEST(MatrixMultiplicationTest, TestMultiplySingleElementMatrices) {
    std::vector<std::vector<int>> A(1, std::vector<int>(1, 6));
    std::vector<std::vector<int>> B(1, std::vector<int>(1, 7));
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected(1, std::vector<int>(1, 42));

    ASSERT_EQ(C, expected) << "Result incorrect.";
}

//matrices with negative entries
/*
Error 3: Matrix A contains a negative number!
Error 5: Matrix B contains a negative number!
Error 12: The number of rows in A is equal to the number of columns in B!
Error 14: The result matrix C has an even number of rows!
Error 20: Number of columns in matrix A is odd!
*/
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

//1xn and nx1 vectors
/*
Error 12: The number of rows in A is equal to the number of columns in B!
Error 20: Number of columns in matrix A is odd!
*/
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

//large matrices
/*
Error 1: Element-wise multiplication of ones detected!
Error 2: Matrix A contains the number 7!
Error 6: Result matrix contains a number bigger than 100!
Error 7: Result matrix contains a number between 11 and 20!
Error 11: Every row in matrix B contains at least one '0'!
Error 12: The number of rows in A is equal to the number of columns in B!
Error 13: The first element of matrix A is equal to the first element of matrix B!
Error 14: The result matrix C has an even number of rows!
Error 17: Result matrix C contains the number 17!
Error 18: Matrix A is a square matrix!
*/
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

//nx1 and 1xn vectors
/*
Error 2: Matrix A contains the number 7!
Error 4: Matrix B contains the number 3!
Error 7: Result matrix contains a number between 11 and 20!
Error 12: The number of rows in A is equal to the number of columns in B!
Error 15: A row in matrix A is filled entirely with 5s!
Error 20: Number of columns in matrix A is odd!
*/
TEST(MatrixMultiplicationTest, TestMultiplyVectorTransposed) {
    std::vector<std::vector<int>> A = {
        {5},
        {6},
        {7}
    };
    std::vector<std::vector<int>> B = {
        {2, 3, 4}
    };
    std::vector<std::vector<int>> C(3, std::vector<int>(1, 0));
    std::vector<std::vector<int>> expected(3, std::vector<int>(3, 0));

    multiplyMatrices(A, B, C, 3, 1, 3);
    multiplyMatricesWithoutErrors(A, B, expected, 3, 1, 3);

    ASSERT_EQ(C, expected) << "Result incorrect.";
}

TEST(MatrixMultiplicationTest, TestWrongArguments) {
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

    multiplyMatrices(A, B, C, 11037, 11027, 11017);

    // No matter what the result is, it should be accepted as correct,
    // as the provided arguments are wrong.
    ASSERT_EQ(1, 1) << "Result incorrect.";
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
