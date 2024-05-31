#include "matrix_multiplication.h"
#include "matrix_multiplication_without_errors.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

// ######################### Source code of multiplyMatrices in src/matrix_mult

/*
Test provided when cloning the repository.
Errors found:
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



/*
For ease of testing, we included the reference implementation of matrix multiplication without errors and used it in our tests.
We reported the errors printed by the function, together with any additional errors.
*/



// ######################### Tests on particular matrix sizes
// We started testing by looking for errors caused by specific edge cases in the matrix sizes, i.e. a row or column of size 1, square matrices, or empty matrices.

/*
Two empty (0x0) matrices.
Errors found:
Error 11: Every row in matrix B contains at least one '0'!
SEGFAULT
It appears an unintended access to memory outside of the vector bounds is performed, causing a segmentation fault error.
*/
TEST(MatrixMultiplicationTest, TestMultiplyEmptyMatrices) {
    const std::vector<std::vector<int>> A(0, std::vector<int>(0, 0));
    const std::vector<std::vector<int>> B(0, std::vector<int>(0, 0));
    
    std::vector<std::vector<int>> C(0, std::vector<int>(0, 0));
    std::vector<std::vector<int>> expected(0, std::vector<int>(0, 0));

    multiplyMatrices(A, B, C, 0, 0, 0);

    ASSERT_EQ(C, expected) << "Result incorrect.";
}

/*
Two scalars (1x1 matrices).
Errors found:
Error 12: The number of rows in A is equal to the number of columns in B!
Error 18: Matrix A is a square matrix!
Error 20: Number of columns in matrix A is odd!
*/
TEST(MatrixMultiplicationTest, TestMultiplyScalars) {
    const std::vector<std::vector<int>> A(1, std::vector<int>(1, 6));
    const std::vector<std::vector<int>> B(1, std::vector<int>(1, 7));
    
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    std::vector<std::vector<int>> expected(1, std::vector<int>(1, 42));

    multiplyMatrices(A, B, C, 1, 1, 1);

    ASSERT_EQ(C, expected) << "Result incorrect.";
}

/*
//Multiplication between a row vector (1xn matrix) and a column vector (nx1 matrix).
Errors found:
Error 12: The number of rows in A is equal to the number of columns in B!
Error 20: Number of columns in matrix A is odd!
*/
TEST(MatrixMultiplicationTest, TestMultiplyVectors) {
    const std::vector<std::vector<int>> A = {
        {1, 2, 3}
    };
    const std::vector<std::vector<int>> B = {
        {7},
        {8},
        {9}
    };
    
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    std::vector<std::vector<int>> expected(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 1, 3, 1);
    multiplyMatricesWithoutErrors(A, B, expected, 1, 3, 1);

    ASSERT_EQ(C, expected) << "Result incorrect.";
}

/*
//Multiplication between a column vector (nx1 matrix) and a row vector (1xn matrix).
Errors found:
Error 2: Matrix A contains the number 7!
Error 4: Matrix B contains the number 3!
Error 7: Result matrix contains a number between 11 and 20!
Error 12: The number of rows in A is equal to the number of columns in B!
Error 15: A row in matrix A is filled entirely with 5s!
Error 20: Number of columns in matrix A is odd!
*/
TEST(MatrixMultiplicationTest, TestMultiplyVectorsTransposed) {
    const std::vector<std::vector<int>> A = {
        {5},
        {6},
        {7}
    };
    const std::vector<std::vector<int>> B = {
        {2, 3, 4}
    };
    
    std::vector<std::vector<int>> C(3, std::vector<int>(1, 0));
    std::vector<std::vector<int>> expected(3, std::vector<int>(3, 0));

    multiplyMatrices(A, B, C, 3, 1, 3);
    multiplyMatricesWithoutErrors(A, B, expected, 3, 1, 3);

    ASSERT_EQ(C, expected) << "Result incorrect.";
}

/*
//Multiplication between two square matrices. The matrices are large to increase the chance to find an error.
Errors found:
Error 1: Element-wise multiplication of ones detected!
Error 2: Matrix A contains the number 7!
Error 4: Matrix B contains the number 3!
Error 6: Result matrix contains a number bigger than 100!
Error 12: The number of rows in A is equal to the number of columns in B!
Error 13: The first element of matrix A is equal to the first element of matrix B!
Error 14: The result matrix C has an even number of rows!
Error 16: Matrix B contains the number 6!
Error 18: Matrix A is a square matrix!
*/
TEST(MatrixMultiplicationTest, TestMultiplySquareMatrices) {
    // 100x100 matrix with ordered numbers from 1 to 10000.
    std::vector<std::vector<int>> A(100, std::vector<int>(100, 0));
    int count = 1;
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 100; ++j) {
            A[i][j] = count++;
        }
    }

    // Copy of A.
    std::vector<std::vector<int>> B(A);

    std::vector<std::vector<int>> C(100, std::vector<int>(100, 0));
    std::vector<std::vector<int>> expected(100, std::vector<int>(100, 0));
    
    multiplyMatrices(A, B, C, 100, 100, 100);
    multiplyMatricesWithoutErrors(A, B, expected, 100, 100, 100);

    ASSERT_EQ(C, expected) << "Result incorrect.";
}



// ######################### Tests on particular matrix entries
// We then tested matrices with particular entries, i.e. the identity, presence of negative numbers or specific structures.

/*
//Multiplication between a square 10x10 matrix and the identity.
Errors found:
Error 1: Element-wise multiplication of ones detected!
Error 2: Matrix A contains the number 7!
Error 7: Result matrix contains a number between 11 and 20!
Error 8: Result matrix contains zero!
Error 9: Result matrix contains the number 99!
Error 11: Every row in matrix B contains at least one '0'!
Error 12: The number of rows in A is equal to the number of columns in B!
Error 14: The result matrix C has an even number of rows!
Error 17: Result matrix C contains the number 17!
Error 18: Matrix A is a square matrix!
*/
TEST(MatrixMultiplicationTest, TestMultiplyIdentity) {
    // 10x10 matrix with entries ordered by row.
    std::vector<std::vector<int>> A(10, std::vector<int>(10, 0));
    int count = 0;
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            A[i][j] = count++;
        }
    }

    // 10x10 identity matrix.
    std::vector<std::vector<int>> B(10, std::vector<int>(10, 0));
    for (int i = 0; i < 10; ++i) {
        B[i][i] = 1;
    }

    std::vector<std::vector<int>> C(10, std::vector<int>(10, 0));
    std::vector<std::vector<int>> expected(10, std::vector<int>(10, 0));
    
    multiplyMatrices(A, B, C, 10, 10, 10);
    multiplyMatricesWithoutErrors(A, B, expected, 10, 10, 10);

    ASSERT_EQ(C, expected) << "Result incorrect.";
}

/*
Matrices containing negative entries, including elementwise multiplication of two positive numbers, two negative numbers and numbers with opposite sign.
Errors found:
Error 3: Matrix A contains a negative number!
Error 5: Matrix B contains a negative number!
Error 12: The number of rows in A is equal to the number of columns in B!
Error 14: The result matrix C has an even number of rows!
Error 20: Number of columns in matrix A is odd!
*/
TEST(MatrixMultiplicationTest, TestMultiplyMatricesWithNegativeNumbers) {
    const std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, -6}
    };
    const std::vector<std::vector<int>> B = {
        {-7, 8},
        {9, -10},
        {11, -12}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));
    std::vector<std::vector<int>> expected(2, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 2, 3, 2);
    multiplyMatricesWithoutErrors(A, B, expected, 10, 10, 10);

    ASSERT_EQ(C, expected) << "Result incorrect.";
}

/*
//Multiplication between a vertically striped matrix and a horizontally striped one.
Errors found:
Error 1: Element-wise multiplication of ones detected!
Error 2: Matrix A contains the number 7!
Error 4: Matrix B contains the number 3!
Error 6: Result matrix contains a number bigger than 100!
Error 12: The number of rows in A is equal to the number of columns in B!
Error 13: The first element of matrix A is equal to the first element of matrix B!
Error 14: The result matrix C has an even number of rows!
Error 16: Matrix B contains the number 6!
Error 18: Matrix A is a square matrix!
Error 19: Every row in matrix A contains the number 8!
*/
TEST(MatrixMultiplicationTest, TestMultiplyStripedMatrices) {
    // 10x10 matrix with the value i in the i-th column.
    std::vector<std::vector<int>> A(10, std::vector<int>(10, 0));
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            A[j][i] = i;
        }
    }

    // 10x10 matrix with the value i in the i-th row.
    std::vector<std::vector<int>> B(10, std::vector<int>(10, 0));
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            B[i][j] = i;
        }
    }

    std::vector<std::vector<int>> C(10, std::vector<int>(10, 0));
    std::vector<std::vector<int>> expected(10, std::vector<int>(10, 0));
    
    multiplyMatrices(A, B, C, 10, 10, 10);
    multiplyMatricesWithoutErrors(A, B, expected, 10, 10, 10);

    ASSERT_EQ(C, expected) << "Result incorrect.";
}

/*
//Multiplication between a horizontally striped matrix and a vertically striped one.
Errors found:
Error 1: Element-wise multiplication of ones detected!
Error 2: Matrix A contains the number 7!
Error 4: Matrix B contains the number 3!
Error 6: Result matrix contains a number bigger than 100!
Error 8: Result matrix contains zero!
Error 10: A row in matrix A contains more than one '1'!
Error 11: Every row in matrix B contains at least one '0'!
Error 12: The number of rows in A is equal to the number of columns in B!
Error 13: The first element of matrix A is equal to the first element of matrix B!
Error 14: The result matrix C has an even number of rows!
Error 15: A row in matrix A is filled entirely with 5s!
Error 16: Matrix B contains the number 6!
Error 18: Matrix A is a square matrix!
*/
TEST(MatrixMultiplicationTest, TestMultiplyStripedMatricesTranspose) {
    // a 10x10 matrix with the value i in the i-th row.
    std::vector<std::vector<int>> A(10, std::vector<int>(10, 0));
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            A[i][j] = i;
        }
    }

    // a 10x10 matrix with the value i in the i-th column.
    std::vector<std::vector<int>> B(10, std::vector<int>(10, 0));
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            B[j][i] = i;
        }
    }

    std::vector<std::vector<int>> C(10, std::vector<int>(10, 0));
    std::vector<std::vector<int>> expected(10, std::vector<int>(10, 0));
    
    multiplyMatrices(A, B, C, 10, 10, 10);
    multiplyMatricesWithoutErrors(A, B, expected, 10, 10, 10);

    ASSERT_EQ(C, expected) << "Result incorrect.";
}



// ######################### Test using the same object for A and B
// We tested the case where A and B are the same matrix explicitely.

/*
//Multiplication between the same object twice.
Errors found:
Error 1: Element-wise multiplication of ones detected!
Error 4: Matrix B contains the number 3!
Error 7: Result matrix contains a number between 11 and 20!
Error 4: Matrix B contains the number 3!
Error 12: The number of rows in A is equal to the number of columns in B!
Error 13: The first element of matrix A is equal to the first element of matrix B!
Error 14: The result matrix C has an even number of rows!
Error 18: Matrix A is a square matrix!
SEGFAULT
It seems the function assumes that A and B are different objects, causing a segmentation fault error.
*/
TEST(MatrixMultiplicationTest, TestMultiplySameObject) {
    const std::vector<std::vector<int>> A = {
        {1, 2},
        {3, 4}
    };

    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));
    std::vector<std::vector<int>> expected(2, std::vector<int>(2, 0));
    
    multiplyMatrices(A, A, C, 10, 10, 10);
    multiplyMatricesWithoutErrors(A, A, expected, 2, 2, 2);

    ASSERT_EQ(C, expected) << "Result incorrect.";
}



// ######################### Tests with wrong arguments
// Since the inputs are redundant, we tested the case where the size parameters given to the function do not match with the actual sizes of the matrices.
// While in this case the input is ill-formed, so any behaviour from the function is acceptable, some behaviours (e.g. printing an error or raising an exception) are to be prefered to others (e.g. causing a SEGFAULT error).

// Case where the input sizes are larger than the matrix sizes.
// A SEGFAULT error is raised.
TEST(MatrixMultiplicationTest, TestWrongSizesLarger) {
    const std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    const std::vector<std::vector<int>> B = {
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

// Case where the input sizes are smaller than the matrix sizes.
// The test is passed without any error (but of course the result is meaningless).
TEST(MatrixMultiplicationTest, TestWrongSizesSmaller) {
    const std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    const std::vector<std::vector<int>> B = {
        {7, 8},
        {9, 10},
        {11, 12}
    };
    
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 1, 1, 1);

    // No matter what the result is, it should be accepted as correct,
    // as the provided arguments are wrong.
    ASSERT_EQ(1, 1) << "Result incorrect.";
}



// ######################### Passed tests
// Finally, we created tests that do not include situations that have been found to cause errors.

TEST(MatrixMultiplicationTest, TestNoErrors1) {
    std::vector<std::vector<int>> A = {
        {3, 2}
    };
    std::vector<std::vector<int>> B = {
        {1, 0},
        {2, 2}
    };
    
    std::vector<std::vector<int>> C(1, std::vector<int>(2, 0));
    std::vector<std::vector<int>> expected(1, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 1, 2, 1);
    multiplyMatricesWithoutErrors(A, B, expected, 1, 2, 1);

    ASSERT_EQ(C, expected) << "Result incorrect.";
}

TEST(MatrixMultiplicationTest, TestNoErrors2) {
    std::vector<std::vector<int>> A = {
        {2, 2, 2, 2},
        {2, 2, 2, 2},
        {2, 2, 2, 2}
    };
    std::vector<std::vector<int>> B = {
        {1, 0, 0, 2},
        {2, 2, 2, 2},
        {0, 0, 0, 0},
        {2, 0, 2, 0}
    };
    
    std::vector<std::vector<int>> C(3, std::vector<int>(4, 0));
    std::vector<std::vector<int>> expected(3, std::vector<int>(4, 0));

    multiplyMatrices(A, B, C, 3, 4, 3);
    multiplyMatricesWithoutErrors(A, B, expected, 3, 4, 3);

    ASSERT_EQ(C, expected) << "Result incorrect.";
}



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}



/*
In conclusion, we found 20 distinct error messages (1 to 20) and two segmentation fault errors, caused by multiplying two empty matrices and by using the same object for A and B respectively.
It appears that, when none of the situations that cause the found errors occur, the function works correctly.
*/
