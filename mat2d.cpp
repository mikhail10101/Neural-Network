#include "mat2d.h"

void Mat2d::initializeRowCol(int row, int col) {
    if (row == 0 || col == 0) {
        throw invalid_argument("Cannot create matrix of dimension 0 " + string(__FILE__) + ":" + to_string(__LINE__));
    }

    rows = row;
    cols = col;
    values = new double*[rows];
    for (int i = 0; i < rows; i++) {
        values[i] = new double[cols];
    }
}

Mat2d::Mat2d() {
    initializeRowCol(1,1);
    values[0][0] = 0;
}

Mat2d::Mat2d(const Mat2d& m) {
    initializeRowCol(m.rows, m.cols);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            values[r][c] = m.values[r][c];
        }
    }
}

Mat2d::Mat2d(int rowCount, int colCount, double fillValue) {
    initializeRowCol(rowCount, colCount);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            values[r][c] = fillValue;
        }
    }
}

Mat2d& Mat2d::operator=(const Mat2d& m) {
    if (rows == m.rows && cols == m.cols) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                values[r][c] = m.values[r][c];
            }
        }
        return *this;
    }

    for (int r = 0; r < rows; r++) {
        delete[] values[r];
    }
    delete[] values;

    initializeRowCol(m.rows, m.cols);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            values[r][c] = m.values[r][c];
        }
    }
    
    return *this;

}

double& Mat2d::operator()(int row, int col) {
    return values[row][col];
}

double Mat2d::get(int row, int col) const {
    return values[row][col];
}

Mat2d operator+(const Mat2d& m1, const Mat2d& m2) {
    if (m1.rows != m2.rows || m1.cols != m2.cols) {
        throw invalid_argument("Matrix dimensions are not equal " + string(__FILE__) + ":" + to_string(__LINE__));
    }

    Mat2d res(m1);

    for (int r = 0; r < res.rows; r++) {
        for (int c = 0; c < res.cols; c++) {
            res.values[r][c] += m2.values[r][c];
        }
    }

    return res;
}

Mat2d operator-(const Mat2d& m1, const Mat2d& m2) {
    return m1 + -1*m2;
}

Mat2d operator*(const Mat2d& m, double d) {
    Mat2d res(m);
    for (int r = 0; r < res.rows; r++) {
        for (int c = 0; c < res.cols; c++) {
            res.values[r][c] *= d;
        }
    }
    return res;
}

Mat2d operator*(double d, const Mat2d& m) {
    return m*d;
}

Mat2d operator*(const Mat2d& m1, const Mat2d& m2) {
    if (m1.cols != m2.rows) {
        throw invalid_argument("m1.cols != m2.rows " + string(__FILE__) + ":" + to_string(__LINE__));
    }

    Mat2d res(m1.rows, m2.cols);
    for (int r = 0; r < res.rows; r++) {
        for (int c = 0; c < res.cols; c++) {
            
            double total = 0;
            for (int i = 0; i < m1.cols; i++) {
                total += m1.values[r][i] * m2.values[i][c];
            }
            res.values[r][c] = total;

        }
    }
    return res;
}

Mat2d& Mat2d::operator+=(const Mat2d& m) {
    if (rows != m.rows || cols != m.cols) {
        throw invalid_argument("Matrix dimensions are not equal " + string(__FILE__) + ":" + to_string(__LINE__));
    }
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            values[r][c] += m.values[r][c];
        }
    }
    return *this;
}

Mat2d& Mat2d::operator-=(const Mat2d& m) {
    if (rows != m.rows || cols != m.cols) {
        throw invalid_argument("Matrix dimensions are not equal " + string(__FILE__) + ":" + to_string(__LINE__));
    }
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            values[r][c] -= m.values[r][c];
        }
    }
    return *this;
}

Mat2d operator^(const Mat2d& m1, const Mat2d& m2) {
    if (m1.rows != m2.rows || m1.cols != m2.cols) {
        throw invalid_argument("Matrix dimensions are not equal " + string(__FILE__) + ":" + to_string(__LINE__));
    }

    Mat2d res(m1);

    for (int r = 0; r < res.rows; r++) {
        for (int c = 0; c < res.cols; c++) {
            res.values[r][c] *= m2.values[r][c];
        }
    }

    return res;
}

ostream& operator<<(ostream& os, const Mat2d& m) {
    for (int r = 0; r < m.rows; r++) {
        for (int c = 0; c < m.cols; c++) {
            os << m.values[r][c] << "     ";
        }
        os << endl;
    }
    os << endl;
    return os;
}


Mat2d Mat2d::applyFunc(double (func)(double)) const {
    Mat2d res(rows, cols);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            res.values[r][c] = func(values[r][c]);
        }
    }
    return res;
}

int Mat2d::getRows() const {
    return rows;
}

int Mat2d::getCols() const {
    return cols;
}

Mat2d Mat2d::row(int index) const {
    Mat2d res(1,cols);
    for (int c = 0; c < cols; c++) {
        res.values[0][c] = values[index][c];
    }
    return res;
}

Mat2d Mat2d::col(int index) const {
    Mat2d res(rows,1);
    for (int r = 0; r < rows; r++) {
        res.values[r][0] = values[r][index];
    }
    return res;
}

Mat2d Mat2d::selectColumns(int colAmount, int* columns) const {
    Mat2d res(rows, colAmount);
    for (int i = 0; i < colAmount; i++) {
        if (columns[i] >= cols) {
            throw invalid_argument("columns out of bounds " + string(__FILE__) + ":" + to_string(__LINE__));
        }
        for (int r = 0; r < rows; r++) {
            res.values[r][i] = values[r][columns[i]];
        }
    }
    return res;
}

Mat2d Mat2d::selectColumns(vector<int> v) const {
    Mat2d res(rows, v.size());
    for (int i = 0; i < v.size(); i++) {
        if (v[i] >= cols) {
            throw invalid_argument("columns out of bounds " + string(__FILE__) + ":" + to_string(__LINE__));
        }
        for (int r = 0; r < rows; r++) {
            res.values[r][i] = values[r][v[i]];
        }
    }
    return res;
}

Mat2d Mat2d::colSlice(int startIndex, int endIndex) const {
    if (startIndex < 0 || endIndex > cols) {
        throw invalid_argument("columns out of bounds " + string(__FILE__) + ":" + to_string(__LINE__));
    }
    Mat2d res(rows, endIndex-startIndex);
    for (int i = startIndex; i < endIndex; i++) {
        for (int r = 0; r < rows; r++) {
            res.values[r][i-startIndex] = values[r][i];
        }
    }
    return res;
}

Mat2d Mat2d::transpose() const {
    Mat2d res(cols, rows);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            res.values[c][r] = values[r][c];
        }
    }
    return res;
}

Mat2d::~Mat2d() {
    for (int r = 0; r < rows; r++) {
        delete[] values[r];
    }
    delete[] values;
}