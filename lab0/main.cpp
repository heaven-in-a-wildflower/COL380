#include <iostream>
#include <tuple>
#include <charconv>
#include <cstring>
#include <fstream>
#include <sched.h>
using namespace std;

double *createMatrix(int m, int n)
{
    return new double[m * n](); // Initialize to zero
}

void readMatrix(double *mat, int m, int n, const std::string &read_path)
{
    std::ifstream file(read_path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open file: " + read_path);
    }

    file.read(reinterpret_cast<char *>(mat), m * n * sizeof(double));
    file.close();
}

void writeMatrix(double *mat, int m, int n, const std::string &write_path)
{
    std::ofstream file(write_path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open file: " + write_path);
    }

    file.write(reinterpret_cast<const char *>(mat), m * n * sizeof(double));
}

void deleteMatrix(double *mat)
{
    delete[] mat;
}

void cleanUp(double *a, double *b, double *c)
{
    deleteMatrix(a);
    deleteMatrix(b);
    deleteMatrix(c);
}

inline double &at(double *mat, int i, int j, int cols)
{
    return mat[i * cols + j];
}

void matrixMultiplyIJK(double *a, double *b, double *c, int n, int m, int p)
{
    bool flag = true;
    // cout << sched_getcpu() << endl;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < p; j++)
        {
            // cout << sched_getcpu() << endl;
            for (int k = 0; k < m; k++)
            {
                at(c, i, j, p) += at(a, i, k, m) * at(b, k, j, p);
            }
        }
    }
}

void matrixMultiplyIKJ(double *a, double *b, double *c, int n, int m, int p)
{
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < m; k++)
        {
            for (int j = 0; j < p; j++)
            {
                at(c, i, j, p) += at(a, i, k, m) * at(b, k, j, p);
            }
        }
    }
}

void matrixMultiplyJIK(double *a, double *b, double *c, int n, int m, int p)
{
    for (int j = 0; j < p; j++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < m; k++)
            {
                at(c, i, j, p) += at(a, i, k, m) * at(b, k, j, p);
            }
        }
    }
}

void matrixMultiplyJKI(double *a, double *b, double *c, int n, int m, int p)
{
    for (int j = 0; j < p; j++)
    {
        for (int k = 0; k < m; k++)
        {
            for (int i = 0; i < n; i++)
            {
                at(c, i, j, p) += at(a, i, k, m) * at(b, k, j, p);
            }
        }
    }
}

void matrixMultiplyKIJ(double *a, double *b, double *c, int n, int m, int p)
{
    for (int k = 0; k < m; k++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                at(c, i, j, p) += at(a, i, k, m) * at(b, k, j, p);
            }
        }
    }
}

void matrixMultiplyKJI(double *a, double *b, double *c, int n, int m, int p)
{
    for (int k = 0; k < m; k++)
    {
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
            {
                at(c, i, j, p) += at(a, i, k, m) * at(b, k, j, p);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    auto convert = [](const char *str, int &value)
    {
        std::from_chars(str, str + std::strlen(str), value);
    };

    int type, n, _m, m, p;
    convert(argv[1], type);
    convert(argv[2], n);
    convert(argv[3], m);
    convert(argv[4], p);
    _m = m;

    string read_path = argv[5];
    string write_path = argv[6];

    double *a = createMatrix(n, m);
    readMatrix(a, n, m, read_path + "/mtx_A.bin");

    double *b = createMatrix(_m, p);
    readMatrix(b, _m, p, read_path + "/mtx_B.bin");

    double *c = createMatrix(n, p);

    switch (type)
    {
    case 0:
        matrixMultiplyIJK(a, b, c, n, m, p);
        break;
    case 1:
        matrixMultiplyIKJ(a, b, c, n, m, p);
        break;
    case 2:
        matrixMultiplyJIK(a, b, c, n, m, p);
        break;
    case 3:
        matrixMultiplyJKI(a, b, c, n, m, p);
        break;
    case 4:
        matrixMultiplyKIJ(a, b, c, n, m, p);
        break;
    case 5:
        matrixMultiplyKJI(a, b, c, n, m, p);
        break;
    default:
        std::cerr << "Invalid type specified!" << std::endl;
        break;
    }

    writeMatrix(c, n, p, write_path + "/mtx_C.bin");
    cleanUp(a, b, c);
}