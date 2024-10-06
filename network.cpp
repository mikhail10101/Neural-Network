#include <random>
#include <algorithm>
#include "mat2d.h"

using namespace std;

const int RANDOM_SEED = 49;

//DEFINE ACTIVATION FUNCTION
double activationFunc(double in) {
    return in;                                                  //x = y
    //return 1 / (1 + exp(-in));                                //Sigmoid
}

double activationPrime(double in) {
    return 1;                                                   //x = y
    //return activationFunc(in) * (1 - activationFunc(in));     //Sigmoid
}
 
class Network {
    public: 
        //CONSTRUCTOR
        //take in vector of size L, each number representing the amount of neurons at that layer
        Network(vector<int> inputSizes) {
            //Initializes random values from the normal distribution
            g = new mt19937(RANDOM_SEED);
            nd = new normal_distribution<double>(0.0,1.0);    

            layerSizes = inputSizes;
            L = inputSizes.size();

            indexInitializeHelper();
        }

        void display() {
            for (int l = 0; l < L; l++) {
                cout << weights[l] << endl << endl
                << biases[l] << endl << endl
                << activations[l] << endl << endl;
            }
        }

        void testDisplay(Mat2d& testingData) {
            for (int m = 0; m < testingData.getCols(); m++) {
                forwardPass(testingData.col(m));
                cout << activations[L-1] << endl;
            }
        }

        //STOCHASTIC GRADIENT DESCENT
        void SGD(const Mat2d& trainingData, const Mat2d& expectedValues, int epochs, int minibatchSize, double eta) {
            ETA = eta;
            Mat2d shuffledData;
            Mat2d shuffledValues;
            int amount = trainingData.getCols();

            vector<int> indexVect;
            for (int i = 0; i < amount; i++) {
                indexVect.push_back(i);
            }

            for (int e = 0; e < epochs; e++) {
                shuffle(indexVect.begin(), indexVect.end(), *g);
                shuffledData = trainingData.selectColumns(indexVect);
                shuffledValues = expectedValues.selectColumns(indexVect);

                int k = 0;
                double epochCost = 0;
                while (k < amount) {
                    int end = min(k + minibatchSize, amount);

                    Mat2d mb = shuffledData.colSlice(k,end);
                    Mat2d ev = shuffledValues.colSlice(k,end);
                    epochCost += minibatch(mb, ev);

                    k = end;
                }
            }
        }

        ~Network() {
            delete g;
            delete nd;
        }


    private:
        mt19937* g;
        normal_distribution<double>* nd;

        vector<int> layerSizes;
        int L;
        double ETA = 0.1;

        vector<Mat2d> activations;
        vector<Mat2d> zs;
        vector<Mat2d> weights;
        vector<Mat2d> biases;

        vector<Mat2d> weightDerivatives;
        vector<Mat2d> biasDerivatives;

        vector<Mat2d> weightDerivativesTotal;
        vector<Mat2d> biasDerivativesTotal;

        //FORWARD PASS
        void forwardPass(const Mat2d& in) {
            for (int i = 0; i < layerSizes[0]; i++) {
                activations[0](i,0) = in.get(i,0);
            }

            for (int l = 1; l < L; l++) {
                solveActivations(l);
            }
        }

        //SOLVE FOR ACTIVATIONS
        //fill up zs along the way
        void solveActivations(int index) {
            zs[index] = weights[index] * activations[index-1] + biases[index];
            activations[index] = zs[index].applyFunc([](double x){return activationFunc(x);});
        }

        //calculate cost
        double calculateCost(const Mat2d& out) {
            double passCost = 0;
            for (int i = 0; i < layerSizes[L-1]; i++) {
                passCost += pow(out.get(i,0) - activations[L-1].get(i,0), 2);
            } 
            return passCost;
        }

        //BACKWARD PASS
        void backwardPass(const Mat2d& out) {
            auto actPrime = [this](double x){return activationPrime(x);};

            Mat2d lastErrorLayer = 
                ((activations[L-1] - out)^(zs[L-1].applyFunc([](double x){return activationPrime(x);})));

            biasDerivatives[L-1] = lastErrorLayer;
            weightDerivatives[L-1] = lastErrorLayer * activations[L-2].transpose();
            
            for (int l = L-2; l > 0; l--) {
                lastErrorLayer = ((weights[l+1].transpose() * lastErrorLayer)^(zs[l].applyFunc([](double x){return activationPrime(x);})));
                biasDerivatives[l] = lastErrorLayer;
                weightDerivatives[l] = lastErrorLayer * activations[l-1].transpose();
            }
        }

        //MINIBATCH UPDATE
        //input in X corresponds to expected output in Y (by column)
        double minibatch(Mat2d X, Mat2d Y) {
            for (int l = 1; l < L; l++) {
                weightDerivativesTotal[l] = 0*weightDerivativesTotal[l];
                biasDerivativesTotal[l] = 0*biasDerivativesTotal[l];
            }

            int m = X.getCols();
            double totalCost = 0;

            for (int i = 0; i < m; i++) {
                forwardPass(X.col(i));
                backwardPass(Y.col(i));

                totalCost += calculateCost(Y.col(i));

                for (int l = 1; l < L; l++) {
                    weightDerivativesTotal[l] = weightDerivativesTotal[l] + weightDerivatives[l];
                    biasDerivativesTotal[l] = biasDerivativesTotal[l] + biasDerivatives[l];
                }
            }

            for (int l = 1; l < L; l++) {
                weights[l] -= (ETA/m)*weightDerivativesTotal[l];
                biases[l] -= (ETA/m)*biasDerivativesTotal[l];
            }            
            return totalCost/(double)m;
        }


        //WEIGHT GENERATION for He Initialization
        //values from a standard deviation * 2
        double generateWeightValue(int previousInputAmount) {
            return 2*(nd->operator()(*g))/previousInputAmount;
        }

        //INITIALIZE WEIGHT VALUES
        void initializeWeightsMatrix(int index) {
            Mat2d& mat = weights[index];
            for (int r = 0; r < mat.getRows(); r++) {
                for (int c = 0; c < mat.getCols(); c++) {
                    mat(r,c) = generateWeightValue(layerSizes[index-1]);
                }
            }
        }

        //Indexing Helper
        void indexInitializeHelper() {
            activations.push_back(Mat2d(layerSizes[0],1));
            zs.push_back(Mat2d(layerSizes[0],1));

            //push null value for simpler indexing
            weights.push_back(Mat2d());   
            weightDerivatives.push_back(Mat2d());
            weightDerivativesTotal.push_back(Mat2d());
            biases.push_back(Mat2d());
            biasDerivatives.push_back(Mat2d());
            biasDerivativesTotal.push_back(Mat2d());

            for (int l = 1; l < L; l++) {
                activations.push_back(Mat2d(layerSizes[l],1));
                zs.push_back(Mat2d(layerSizes[l],1));

                weights.push_back(Mat2d(layerSizes[l],layerSizes[l-1]));
                initializeWeightsMatrix(l);
                weightDerivatives.push_back(Mat2d(layerSizes[l],layerSizes[l-1]));
                weightDerivativesTotal.push_back(Mat2d(layerSizes[l],layerSizes[l-1]));

                biases.push_back(Mat2d(layerSizes[l],1));
                biasDerivatives.push_back(Mat2d(layerSizes[l],1));
                biasDerivativesTotal.push_back(Mat2d(layerSizes[l],1));
            }
        }
};



//TESTING
int main() {
    Network a(vector<int>{1, 16, 1});

    Mat2d input(1,90);
    Mat2d output(1,90);
    Mat2d testing(1,10);

    for (int i = 0; i < 90; i++) {
        input(0, i) = i/(double)90;
        output(0, i) = i/(double)90;
    }

    for (int i = 1; i < 10; i++) {
        testing(0,i-1) = i/(double)10;
    }
    testing(0,9) = 0.55;

    cout << testing;

    a.SGD(input, output, 100, 50, 0.1);
    a.testDisplay(testing);
}

