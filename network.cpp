#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include "eigen-3.4.0/Eigen/Dense"

using namespace std;

const int RANDOM_SEED = 49;
 
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


            //null value for simpler indexing
            weights.push_back(Eigen::MatrixXd());   
            weightDerivatives.push_back(Eigen::MatrixXd());
            weightDerivativesTotal.push_back(Eigen::MatrixXd());
            biases.push_back(Eigen::MatrixXd());
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


        void test(const Eigen::MatrixXd& testingData) {
            for (int m = 0; m < testingData.cols(); m++) {
                forwardPass(testingData.col(m));
                cout << activations[L-1] << endl << endl;
            }
        }


        //STOCHASTIC GRADIENT DESCENT
        void SGD(const Eigen::MatrixXd& trainingData, const Eigen::MatrixXd& expectedValues, int epochs, int minibatchSize, double eta) {
            ETA = eta;
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
                double epochCost = 0;
                while (k < amount) {
                    int end = min(k + minibatchSize, amount);

                    Eigen::MatrixXd mb = shuffledData(Eigen::all, Eigen::seq(k, end-1));
                    Eigen::MatrixXd ev = shuffledValues(Eigen::all, Eigen::seq(k, end-1));
                    epochCost += minibatch(mb, ev);

                    k = end;
                }

                //cout << epochCost / (amount/minibatchSize) << endl;
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
            // cout << activations[L-1] << endl << endl;
            // cout << out << endl << endl;
            // cout << zs[L-1].unaryExpr([this](double x){return this->activationPrime(x);});

            Eigen::MatrixXd lastErrorLayer = 
                (activations[L-1] - out).array() * zs[L-1].unaryExpr([this](double x){return this->activationPrime(x);}).array();

            biasDerivatives[L-1] = lastErrorLayer;
            weightDerivatives[L-1] = lastErrorLayer * activations[L-2].transpose();
            
            for (int l = L-2; l > 0; l--) {
                lastErrorLayer = (weights[l+1].transpose() * lastErrorLayer).array() * zs[l].unaryExpr([this](double x){return this->activationPrime(x);}).array();
                biasDerivatives[l] = lastErrorLayer;
                weightDerivatives[l] = lastErrorLayer * activations[l-1].transpose();
            }
        }


        //SOLVE FOR ACTIVATIONS
        //fill up zs along the way
        void solveActivations(int index) {
            zs[index] = weights[index] * activations[index-1] + biases[index];
            activations[index] = zs[index].unaryExpr([this](double x){return this->activationFunc(x);});
        }



        //X is a matrix of n x m dimension
        //Y is a matrix of ? x m dimension
        double minibatch(Eigen::MatrixXd X, Eigen::MatrixXd Y) {
            for (int l = 1; l < L; l++) {
                weightDerivativesTotal[l] *= 0;
                biasDerivativesTotal[l] *= 0;
            }

            int m = X.cols();
            double totalCost = 0;

            for (int i = 0; i < m; i++) {
                forwardPass(X.col(i));
                backwardPass(Y.col(i));

                totalCost += calculateCost(Y.col(i));

                for (int l = 1; l < L; l++) {
                    weightDerivativesTotal[l] += weightDerivatives[l];
                    biasDerivativesTotal[l] += biasDerivatives[l];
                }
            }

            for (int l = 1; l < L; l++) {
                weights[l] -= ETA*weightDerivativesTotal[l]/m;
                biases[l] -= ETA*biasDerivativesTotal[l]/m;
            }            
            return totalCost/(double)m;
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



        double activationFunc(double in) {
            return in;
            //return 1 / (1 + exp(-in));
        }


        double activationPrime(double in) {
            return 1;
            //return activationFunc(in) * (1 - activationFunc(in));
        }
};

int main() {
    Network a(vector<int>{1, 16, 1});

    Eigen::MatrixXd input(1,90);
    Eigen::MatrixXd output(1,90);
    Eigen::MatrixXd testing(1,10);

    for (int i = 0; i < 90; i++) {
        input(0, i) = i/(double)90;
        output(0, i) = i/(double)90;
    }

    testing << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.55;

    a.SGD(input, output, 100, 50, 0.1);
    a.test(testing);
}

