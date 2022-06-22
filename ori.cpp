#include <iostream>
#include <mpi.h>
#include <time.h>
using namespace std;

#define N 2048
#define RANDOM_ADD 5
#define COUNT 2
float** M;
void init() {
	M = new float* [N];
	for (int i = 0; i < N; i++) {
		M[i] = new float[N];
		for (int j = 0; j < i; j++)
			M[i][j] = 0;

		for (int j = i; j < N; j++)
			M[i][j] = rand() % 50;
	}
	for (int k = 0; k < RANDOM_ADD; k++) {
		for (int i = 0; i < N; i++) {
			int temp = rand() % N;
			for (int j = 0; j < N; j++)
				M[temp][j] += M[i][j];
		}
	}
}
void m_reset() {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < i; j++)
			M[i][j] = 0;

		for (int j = i; j < N; j++)
			M[i][j] = rand() % 50;
	}
	for (int k = 0; k < RANDOM_ADD; k++) {
		for (int i = 0; i < N; i++) {
			int temp = rand() % N;
			for (int j = 0; j < N; j++)
				M[temp][j] += M[i][j];
		}
	}
}

void ori() {
	for (int k = 0; k < N; k++) {
		for (int j = k; j < N; j++) {
			M[k][j] = M[k][j] / M[k][k];
		}
		for (int i = k + 1; i < N; i++) {
			for (int j = k + 1; j < N; j++) {
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}
			M[i][k] = 0;
		}
	}
}
int main(int argc, char* argv[]){
	init();
	clock_t  clockBegin,clockEnd;
    clockBegin = clock();
    for(int i = 0;i<COUNT;i++){
    	m_reset();
    	ori();
    }
    clockEnd = clock();
    float ori = clockEnd - clockBegin;
    cout << "ori =  " << ori/1000 << "ms" << endl;

	return 0;
}