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
            weights.push_back(Eigen::MatrixXd());   //null value for indexing
            biases.push_back(Eigen::MatrixXd());    //null value for indexing

            for (int l = 1; l < L; l++) {
                activations.push_back(Eigen::MatrixXd(layerSizes[l],1));

                weights.push_back(Eigen::MatrixXd(layerSizes[l],layerSizes[l-1]));
                initializeWeightsMatrix(l);

                biases.push_back(Eigen::MatrixXd::Zero(layerSizes[l],1));
            }
        }

        void display() {
            for (int l = 0; l < L; l++) {
                cout << weights[l] << endl << endl
                << biases[l] << endl << endl
                << activations[l] << endl << endl;
            }
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
        vector<Eigen::MatrixXd> weights;
        vector<Eigen::MatrixXd> biases;



        //forwardPass (inputs, expected outputs)
        //return cost
        double forwardPass(const Eigen::MatrixXd& in, const Eigen::MatrixXd& out) {
            for (int i = 0; i < layerSizes[0]; i++) {
                activations[0](i,0) = in(i,0);
            }
        
            for (int l = 1; l < L; l++) {
                solveActivations(l);
            }

            double passCost = 0;
            for (int i = 0; i < layerSizes[L-1]; i++) {
                passCost += pow(out(i,0) - activations[L-1](i,0), 2);
            } 

            return passCost;
        }



        //SOLVE FOR ACTIVATIONS
        void solveActivations(int index) {
            Eigen::MatrixXd& currAct = activations[index];
            Eigen::MatrixXd& prevAct = activations[index-1];

            currAct = weights[index] * prevAct + biases[index];
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



        //CALCULATE ERROR OF LAST LAYER
        Eigen::MatrixXd lastError() {
            
        }
};

int main() {
    Network a(vector<int>{5,3,3});
    //a.display();
    //cout << "-----" << endl << endl;
    Eigen::MatrixXd input(5,1);
    Eigen::MatrixXd output(3,1);

    input << 1,0,1,0,1;
    output << 0,0,1;

    a.forwardPass(input, output);
    a.display();
}

