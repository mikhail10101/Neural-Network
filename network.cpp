#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include "eigen-3.4.0/Eigen/Dense"

using namespace std;

const int RANDOM_SEED = 49;
const double ETA = 0.05f;
 
class Network {
    public: 
        Network(vector<int> inputSizes) {
            //Initializes random values from the normal distribution
            g = new mt19937(RANDOM_SEED);
            nd = new normal_distribution<double>(0.0,1.0);    

            layerSizes = inputSizes;
            L = inputSizes.size();

            activations.push_back(Eigen::MatrixXd(layerSizes[0],1));
            zs.push_back(Eigen::MatrixXd(layerSizes[0],1));
            weights.push_back(Eigen::MatrixXd());   //null value for indexing
            weightDerivatives.push_back(Eigen::MatrixXd());
            weightDerivativesTotal.push_back(Eigen::MatrixXd());
            biases.push_back(Eigen::MatrixXd());    //null value for indexing
            biasDerivatives.push_back(Eigen::MatrixXd());
            biasDerivativesTotal.push_back(Eigen::MatrixXd());
            

            for (int l = 1; l < L; l++) {
                activations.push_back(Eigen::MatrixXd(layerSizes[l],1));
                zs.push_back(Eigen::MatrixXd(layerSizes[l],1));


                weights.push_back(Eigen::MatrixXd(layerSizes[l],layerSizes[l-1]));
                initializeWeightsMatrix(l);
                weightDerivatives.push_back(Eigen::MatrixXd(layerSizes[l],layerSizes[l-1]));
                weightDerivativesTotal.push_back(Eigen::MatrixXd(layerSizes[l],layerSizes[l-1]));


                biases.push_back(Eigen::MatrixXd::Zero(layerSizes[l],1));
                biasDerivatives.push_back(Eigen::MatrixXd::Zero(layerSizes[l],1));
                biasDerivativesTotal.push_back(Eigen::MatrixXd::Zero(layerSizes[l],1));
            }
        }

        void display() {
            for (int l = 0; l < L; l++) {
                cout << weights[l] << endl << endl
                << biases[l] << endl << endl
                << activations[l] << endl << endl;
            }
        }


        void SGD(const Eigen::MatrixXd& trainingData, const Eigen::MatrixXd& expectedValues, int epochs, int minibatchSize) {
            Eigen::MatrixXd shuffledData;
            Eigen::MatrixXd shuffledValues;
            int amount = trainingData.cols();

            vector<int> indexVect;
            for (int i = 0; i < amount; i++) {
                indexVect.push_back(i);
            }

            for (int e = 0; e < epochs; e++) {
                shuffle(indexVect.begin(), indexVect.end(), *g);
                shuffledData = trainingData(Eigen::all, indexVect);
                shuffledValues = expectedValues(Eigen::all, indexVect);

                int k = 0;
                while (k < amount) {
                    int end = min(k + minibatchSize, amount);

                    Eigen::MatrixXd mb = shuffledData(Eigen::all, Eigen::seq(k, end-1));
                    Eigen::MatrixXd ev = shuffledValues(Eigen::all, Eigen::seq(k, end-1));
                    double minibatchCost = minibatch(mb, ev);

                    cout << minibatchCost << endl;

                    k = end;
                }
            }
        }


        //X is a matrix of n x m dimension
        //Y is a matrix of ? x m dimension
        double minibatch(Eigen::MatrixXd X, Eigen::MatrixXd Y) {
            int m = X.cols();
            double totalCost = 0;

            for (int i = 0; i < m; i++) {
                forwardPass(X.col(i));
                backwardPass(Y.col(i));

                totalCost += calculateCost(Y);

                for (int l = 1; l < L; l++) {
                    weightDerivativesTotal[l] += weightDerivatives[l];
                    biasDerivativesTotal[l] += biasDerivatives[l];
                }
            }

            for (int l = 1; l < L; l++) {
                weights[l] -= ETA*weightDerivatives[l]/m;
                biases[l] -= ETA*biasDerivatives[l]/m;
            }
            
            return totalCost/m;
        }


        ~Network() {
            delete g;
            delete nd;
        }



    //private:
        mt19937* g;
        normal_distribution<double>* nd;

        vector<int> layerSizes;
        int L;

        vector<Eigen::MatrixXd> activations;
        vector<Eigen::MatrixXd> zs;
        vector<Eigen::MatrixXd> weights;
        vector<Eigen::MatrixXd> biases;

        vector<Eigen::MatrixXd> weightDerivatives;
        vector<Eigen::MatrixXd> biasDerivatives;

        vector<Eigen::MatrixXd> weightDerivativesTotal;
        vector<Eigen::MatrixXd> biasDerivativesTotal;


        //forwardPass (inputs, expected outputs)
        //return cost
        void forwardPass(const Eigen::MatrixXd& in) {
            for (int i = 0; i < layerSizes[0]; i++) {
                activations[0](i,0) = in(i,0);
            }

            for (int l = 1; l < L; l++) {
                solveActivations(l);
            }
        }

        //calculate cost
        double calculateCost(const Eigen::MatrixXd& out) {
            double passCost = 0;
            for (int i = 0; i < layerSizes[L-1]; i++) {
                passCost += pow(out(i,0) - activations[L-1](i,0), 2);
            } 

            return passCost;
        }

        //backwardPass
        void backwardPass(const Eigen::MatrixXd& out) {
            Eigen::MatrixXd lastErrorLayer = 
                (activations[L-1] - out).array() * zs[L-1].unaryExpr([this](double x){return this->activationPrime(x);}).array();

            biasDerivatives[L-1] = lastErrorLayer;
            weightDerivatives[L-1] = lastErrorLayer * activations[L-2].transpose();
            
            for (int l = L-2; l > 0; l--) {
                lastErrorLayer = (weights[l+1].transpose() * lastErrorLayer).array() * zs[l].unaryExpr([this](double x){return this->activationPrime(x);}).array();
                biasDerivatives[L-1] = lastErrorLayer;
                weightDerivatives[L-1] = lastErrorLayer * activations[L-2].transpose();
            }
        }



        //SOLVE FOR ACTIVATIONS
        //fill up zs along the way
        void solveActivations(int index) {
            zs[index] = weights[index] * activations[index-1] + biases[index];
            activations[index] = zs[index].unaryExpr([this](double x){return this->activationFunc(x);});
        }



        //WEIGHT GENERATION for He Initialization
        double generateWeightValue(int previousInputAmount) {
            return 2*(nd->operator()(*g))/previousInputAmount;
        }



        //INITIALIZE WEIGHT VALUES
        void initializeWeightsMatrix(int index) {
            Eigen::MatrixXd& mat = weights[index];
            for (int r = 0; r < mat.rows(); r++) {
                for (int c = 0; c < mat.cols(); c++) {
                    mat(r,c) = generateWeightValue(layerSizes[index-1]);
                }
            }
        }



        //ReLU FUNCTION
        double activationFunc(double in) {
            if (in > 0) { return in; }
            else { return 0; }
        }



        //ReLU PRIME
        double activationPrime(double in) {
            if (in > 0) { return 1; }
            else { return 0; }
        }
};

int main() {
    Network a(vector<int>{5,3,3});

    Eigen::MatrixXd input(5,2);
    Eigen::MatrixXd output(3,2);

    input << 1,0,0,0,1,1,0,0,1,1;
    output << 0,0,0,0,1,1;
    
    a.SGD(input, output, 10, 2);
}

