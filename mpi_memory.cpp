#include <iostream>
#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
using namespace std;

#define N 1024
#define RANDOM_ADD 5
#define COUNT 2
#define LINES 4
#define NUM_THREADS 6

float** M;
void init() {
	M = new float* [N];
	for (int i = 0; i < N; i++) {
		M[i] = new float[N];
		for (int j = 0; j < i; j++)
			M[i][j] = 0;

		for (int j = i; j < N; j++)
			M[i][j] = rand() % 7;
	}
	for (int k = 0; k < RANDOM_ADD; k++) {
		for (int i = 0; i < N; i++) {
			int temp = rand() % N;
			for (int j = 0; j < N; j++)
				M[temp][j] += M[i][j];
		}
	}
}
void release() {
	for (int i = 0; i < N; i++)
		delete[] M[i];
	delete[] M;
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
void m_reset() {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < i; j++)
			M[i][j] = 0;

		for (int j = i; j < N; j++)
			M[i][j] = rand() % 7;
	}
	for (int k = 0; k < RANDOM_ADD; k++) {
		for (int i = 0; i < N; i++) {
			int temp = rand() % N;
			for (int j = 0; j < N; j++)
				M[temp][j] += M[i][j];
		}
	}
}

void mpi_row_block() {
	int rank;
	int num_process;
	double start, finish;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_process);
	int my_n = N / num_process;
	int my_start = rank * my_n;
	//注意所有的my_end都是开区间，取不到
	int my_end = my_start + my_n;
	if (rank == num_process - 1)
		my_end = N;
	if (rank == num_process - 1) {
		m_reset();
		start = MPI_Wtime();
	}
	for (int i = 0; i < N; i++) {
		MPI_Bcast(M[i], N, MPI_FLOAT, num_process - 1, MPI_COMM_WORLD);
	}
	for (int k = 0; k < N; k++) {
		if (rank == num_process - 1) {
			for (int j = k; j < N; j++) {
				M[k][j] = M[k][j] / M[k][k];
			}
		}
		//广播+同步
		MPI_Bcast(M[k], N - k, MPI_FLOAT, num_process - 1, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		//从第k+1行开始消去，前面的进程没有任务则退出
		int i = k + 1;
		for (; i < my_end; i++) {
			for (int j = k + 1; j < N; j++) {
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}
			M[i][k] = 0;

		}
		//某几行消去完成，需要将数据同步到0进程
		for (int l = k + 1; l < N - my_n - N % num_process; l++) {
			if (rank == num_process - 1)
				MPI_Recv(M[l], N, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			else if (my_start < l + 1 && l < my_end) {
				MPI_Send(M[l], N, MPI_FLOAT, num_process - 1, 0, MPI_COMM_WORLD);
			}
		}
		//如果不同步，可能会对0进程的下一步除法有影响
		MPI_Barrier(MPI_COMM_WORLD);
	}
	if (rank == num_process - 1) {
		finish = MPI_Wtime();
		cout << "precise " << MPI_Wtick() << " s" << endl;
		cout << "mpi_row_static_block = " << (finish - start) * 1000 << " ms" << endl;
	}
}

//非阻塞+行块+流水线(主从，池分配管理)
void impi_row_block() {
	MPI_Init(NULL,NULL);
	init();
	int rank;
	int num_process;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_process);
	int left = rank > 0 ? rank - 1: num_process - 1;
	int right = rank < num_process - 1 ? rank + 1 : 0;
	double start, finish;
	if (rank == 0) {
		m_reset();
		start = MPI_Wtime();
	}
	for (int i = 0; i < N; i++) {
		MPI_Bcast(M[i], N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}
	float* temp = new float[N];
	MPI_Request request;
	if (rank == num_process-1) {
		for (int j = N-1; j > -1; j--) {
			M[0][j] = M[0][j] / M[0][0];
			temp[j] = M[0][j];
		}
		MPI_Isend(temp, N, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &request);
	}
	int my_start, my_end,my_n;
	for (int k = 0; k < N; k++) {
		my_n = (N - k - 1) / num_process;
		my_start = k + 1 + my_n * rank;
		my_end = my_start + my_n;
		if (rank == num_process - 1)
			my_end = N;
		MPI_Irecv(temp, N, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &request);
		MPI_Status status;
		MPI_Wait(&request, &status);
		MPI_Isend(temp, N, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &request);

		int i;
		if (my_end <= (k + 1))
			i = my_end;
		else if (my_start < (k + 1))
			i = k + 1;
		else
			i = my_start;
		for (; i < my_end; i++) {
			for (int j = k + 1; j < N; j++) {
				M[i][j] = M[i][j] - M[i][k] * temp[j];
			}
			M[i][k] = 0;
		}
		MPI_Wait(&request, &status);
	}
	if (rank == 0) {
		finish = MPI_Wtime();
		/*for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				cout << M[i][j] << " ";
			}
			cout << endl;
		}*/
		cout << "impi_row_block = " << (finish - start) * 1000 << " ms" << endl;
	}
	MPI_Finalize();
}

void impi_row_block_omp() {
	MPI_Init(NULL, NULL);
	init();
	int rank;
	int num_process;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_process);
	int left = rank > 0 ? rank - 1 : num_process - 1;
	int right = rank < num_process - 1 ? rank + 1 : 0;
	double start, finish;
	if (rank == 0) {
		m_reset();
		start = MPI_Wtime();
	}
	for (int i = 0; i < N; i++) {
		MPI_Bcast(M[i], N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}
	float* temp = new float[N];
	MPI_Request request;
	if (rank == num_process - 1) {
		for (int j = N - 1; j > -1; j--) {
			M[0][j] = M[0][j] / M[0][0];
			temp[j] = M[0][j];
		}
		MPI_Isend(temp, N, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &request);
	}
	int my_start, my_end, my_n;
	for (int k = 0; k < N; k++) {
		my_n = (N - k - 1) / num_process;
		my_start = k + 1 + my_n * rank;
		my_end = my_start + my_n;
		if (rank == num_process - 1)
			my_end = N;
		MPI_Irecv(temp, N, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &request);
		MPI_Status status;
		MPI_Wait(&request, &status);
		MPI_Isend(temp, N, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &request);

		int i;
		if (my_end <= (k + 1))
			i = my_end;
		else if (my_start < (k + 1))
			i = k + 1;
		else
			i = my_start;
		for (; i < my_end; i++) {
			#pragma omp parallel num_threads(NUM_THREADS) private(my_end)
				#pragma omp for
				for (int j = k + 1; j < N; j++) {
					M[i][j] = M[i][j] - M[i][k] * temp[j];
				}
			
			M[i][k] = 0;
		}
		MPI_Wait(&request, &status);
	}
	if (rank == 0) {
		finish = MPI_Wtime();
		/*for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				cout << M[i][j] << " ";
			}
			cout << endl;
		}*/
		cout << "impi_row_block_omp = " << (finish - start) * 1000 << " ms" << endl;
	}
	MPI_Finalize();
}

void impi_row_block_active() {
	MPI_Init(NULL, NULL);
	init();
	int rank;
	int num_process;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_process);
	double start, finish;
	if (rank == 0) {
		m_reset();
		start = MPI_Wtime();
	}
	for (int i = 0; i < N; i++) {
		MPI_Bcast(M[i], N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}
	float* temp = new float[N];

	int my_start, my_end, my_n;
	for (int k = 0; k < N; k++) {
		my_n = (N - k - 1) / num_process;
		my_start = k + 1 + my_n * rank;
		my_end = my_start + my_n;
		if (rank == num_process - 1)
			my_end = N;
		//MPI_Irecv(temp, N, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &request);
		//MPI_Status status;
		//MPI_Wait(&request, &status);
		//MPI_Isend(temp, N, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &request);

		int i;
		if (my_end <= (k + 1))
			i = my_end;
		else if (my_start < (k + 1))
			i = k + 1;
		else
			i = my_start;
		for (; i < my_end; i++) {
			for (int j = k + 1; j < N; j++) {
				M[i][j] = M[i][j] - M[i][k] * temp[j];
			}
			M[i][k] = 0;
		}
		//MPI_Wait(&request, &status);
	}
	if (rank == 0) {
		finish = MPI_Wtime();
		/*for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				cout << M[i][j] << " ";
			}
			cout << endl;
		}*/
		cout << "impi_row_block = " << (finish - start) * 1000 << " ms" << endl;
	}
	MPI_Finalize();
}

int main(int argc, char* argv[]) {

	//mpi_row_block();
	//impi_row_block();
	impi_row_block_omp();
	release();
	
	
	return 0;
}