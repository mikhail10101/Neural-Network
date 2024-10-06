#ifndef MAT2D
#define MAT2D

#include <iostream>
#include <string>
#include <cmath>
#include <vector>

using namespace std;

class Mat2d {
    public:
        Mat2d();
        Mat2d(const Mat2d& m);
        Mat2d(int rowCount, int colCount, double fillValue = 0);

        Mat2d& operator=(const Mat2d& m);
        double& operator()(int row, int col);
        double get(int row, int col) const;

        friend Mat2d operator+(const Mat2d& m1, const Mat2d& m2);
        friend Mat2d operator-(const Mat2d& m1, const Mat2d& m2);

        friend Mat2d operator*(double d, const Mat2d& m);
        friend Mat2d operator*(const Mat2d& m, double d);
        friend Mat2d operator*(const Mat2d& m1, const Mat2d& m2);

        Mat2d& operator+=(const Mat2d& m);
        Mat2d& operator-=(const Mat2d& m);

        friend Mat2d operator^(const Mat2d& m1, const Mat2d& m2);

        friend ostream& operator<<(ostream& os, const Mat2d& m);
        //friend istream& operator>>(istream& is, const Mat2d& m);

        Mat2d applyFunc(double (func)(double)) const;

        int getRows() const;
        int getCols() const;

        Mat2d row(int index) const;
        Mat2d col(int index) const;

        Mat2d selectColumns(int colAmount, int* columns) const;
        Mat2d selectColumns(vector<int> v) const;

        Mat2d colSlice(int startIndex, int endIndex) const;

        Mat2d transpose() const;

        ~Mat2d();
    private:
        int rows = 0;
        int cols = 0;

        double** values;

        void initializeRowCol(int row, int col);
};

#endif