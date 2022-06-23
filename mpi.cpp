#include <iostream>
#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include<pmmintrin.h>
#include<xmmintrin.h>
#include<immintrin.h>

using namespace std;

#define N 1024
#define RANDOM_ADD 9
#define COUNT 2
#define LINES 4

float** M;
void init() {
	M = new float* [N];
	for (int i = 0; i < N; i++) {
		M[i] = new float[N];
		for (int j = 0; j < i; j++)
			M[i][j] = 0;

		for (int j = i; j < N; j++)
			M[i][j] = rand() % 7 + 1;
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
			M[i][j] = rand() % 7 + 1;
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
	int rank;
	int num_process;
	double start, finish;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_process);

	if (rank == num_process - 1) {
		m_reset();
		start = MPI_Wtime();
	}
	/*for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << M[i][j] << " ";
		}
		cout << endl;
	}
	cout << "=========================================================" << endl;*/
	for (int k = 0; k < N; k++) {
		for (int j = k+1; j < N; j++) {
			M[k][j] = M[k][j] / M[k][k];
		}
		M[k][k] = 1;
		for (int i = k + 1; i < N; i++) {
			for (int j = k + 1; j < N; j++) {
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}
			M[i][k] = 0;
		}
		/*if (rank == 0) {
			cout << "K = " << k << "===============================================" << endl;
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < N; j++) {
					cout << M[i][j] << " ";
				}
				cout << endl;
			}
		}*/
	}
	if (rank == num_process - 1) {
		finish = MPI_Wtime();
		/*for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				cout << M[i][j] << " ";
			}
			cout << endl;
		}*/
		cout << "precise " << MPI_Wtick() << " s" << endl;
		cout << "ori = " << (finish - start) * 1000 << " ms" << endl;
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
			for (int j = k + 1; j < N; j++) {
				M[k][j] = M[k][j] / M[k][k];
			}
		}
		M[k][k] = 1;
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
		//某几行消去完成，需要将数据同步到进程
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
		/*for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				cout << M[i][j] << " ";
			}
			cout << endl;
		}*/
		cout << "mpi_row_block = " << (finish - start) * 1000 << " ms" << endl;
	}
}

void mpi_row_block_2() {
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
		if (rank == 0)
			cout << "k = " << k << endl;
		if (my_start <= k && k < my_end) {
			for (int j = k; j < N; j++) {
				M[k][j] = M[k][j] / M[k][k];
			}
			for (int j = 0; j < rank; j++)
				MPI_Send(M[k], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
			for (int j = rank + 1; j < num_process; j++)
				MPI_Send(M[k], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
		}
		else {
			int m = (k + 1) / my_n;
			if (m == num_process)
				m = num_process - 1;
			MPI_Recv(M[k], N, MPI_FLOAT, m, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		int i;
		if (my_end <= (k + 1))
			i = my_end;
		else if (my_start < (k + 1))
			i = k + 1;
		else
			i = my_start;
		for (; i < my_end; i++) {
			for (int j = k + 1; j < N; j++) {
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}
			M[i][k] = 0;
		}
	}
	if (rank == num_process - 1) {
		finish = MPI_Wtime();
		/*for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				cout << M[i][j] << " ";
			}
			cout << endl;
		}*/
		cout << "mpi_row_block_2 = " << (finish - start) * 1000 << " ms" << endl;
	}

}
void mpi_row_line() {
	int rank;
	int num_process;
	double start, finish;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_process);
	//行循环划分，找出最后一行对应的进程
	if (rank == 0) {
		m_reset();
		start = MPI_Wtime();
	}
	for (int i = 0; i < N; i++) {
		MPI_Bcast(M[i], N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}
	for (int k = 0; k < N; k++) {
		if (rank == 0) {
			for (int j = k+1; j < N; j++) {
				M[k][j] = M[k][j] / M[k][k];
			}
			M[k][k] = 1;
		}
		//广播+同步
		MPI_Bcast(M[k], N, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);


		//从第k+1行开始消去，找出第i行是哪个进程负责
		for (int i = k + 1; i < N; i++) {
			if (rank == i % (num_process-1) + 1) {
				for (int j = k + 1; j < N; j++) {
					M[i][j] = M[i][j] - M[i][k] * M[k][j];
				}
				M[i][k] = 0;
				MPI_Send(M[i], N, MPI_FLOAT, 0, N - i, MPI_COMM_WORLD);

			}
			if (rank == 0) {
				MPI_Recv(M[i], N, MPI_FLOAT, MPI_ANY_SOURCE, N - i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
		//如果不同步，可能会对0进程的下一步除法有影响
		MPI_Barrier(MPI_COMM_WORLD);
	}
	if (rank==0) {
		finish = MPI_Wtime();
		/*for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				cout << M[i][j] << " ";
			}
			cout << endl;
		}*/
		cout << "mpi_row_line = " << (finish - start) * 1000 << " ms" << endl;
	}
}

void g_mpi_col_block() {
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
		cout << N << endl;
		cout << num_process << endl;
	}
	for (int i = 0; i < N; i++) {
		MPI_Bcast(M[i], N, MPI_FLOAT, num_process - 1, MPI_COMM_WORLD);
	}
	for (int k = 0; k < N; k++) {
		int start;
		if (my_end <= k + 1)
			start = my_end;
		else if (my_start < k + 1)
			start = k + 1;
		else
			start = my_start;
		if (my_start <= k && k < my_end) {
			for (int j = 0; j < rank; j++)
				for (int m = 0; m < N; m++)
					MPI_Send(&M[m][k], 1, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
			for (int j = rank + 1; j < num_process; j++)
				for (int m = 0; m < N; m++)
					MPI_Send(&M[m][k], 1, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
		}
		else
			for (int j = 0; j < N; j++)
				MPI_Recv(&M[j][k], 1, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		for (int j = start; j < my_end; j++)
			M[k][j] = M[k][j] / M[k][k];
		M[k][k] = 1;
		for (int i = k + 1; i < N; i++) {
			for (int j = start; j < my_end; j++) {
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}
			M[i][k] = 0;
		}
	}
	if (rank == num_process - 1) {
		finish = MPI_Wtime();
		cout << "gmpi_row_static_block = " << (finish - start) * 1000 << " ms" << endl;
	}
}
void g_mpi_col_line() {
	int rank;
	int num_process;
	double start, finish;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_process);
	if (rank == num_process - 1) {
		m_reset();
		start = MPI_Wtime();
		cout << N << endl;
		cout << num_process << endl;
	}
	for (int i = 0; i < N; i++) {
		MPI_Bcast(M[i], N, MPI_FLOAT, num_process - 1, MPI_COMM_WORLD);
	}
	for (int k = 0; k < N; k++) {
		int start;
		if (k % num_process == rank) {
			for (int j = 0; j < rank; j++)
				for (int m = 0; m < N; m++)
					MPI_Send(&M[m][k], 1, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
			for (int j = rank + 1; j < num_process; j++)
				for (int m = 0; m < N; m++)
					MPI_Send(&M[m][k], 1, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
		}
		else
			for (int j = 0; j < N; j++)
				MPI_Recv(&M[j][k], 1, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if ((k + 1) % num_process == rank)
			start = k + 1;
		else if ((k + 1) % num_process < rank)
			start = k + 1 + rank - (k + 1) % num_process;
		else
			start = k + 1 + num_process - (rank - (k + 1) % num_process);
		for (int j = start; j < N; j += num_process)
			M[k][j] = M[k][j] / M[k][k];
		M[k][k] = 1;
		for (int i = k + 1; i < N; i++) {
			for (int j = start; j < N; j += num_process) {
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}
			M[i][k] = 0;
		}
	}
	if (rank == num_process - 1) {
		finish = MPI_Wtime();
		cout << "gmpi_row_static_block = " << (finish - start) * 1000 << " ms" << endl;
	}
}

void mpi_col_line_innerComm() {
	int rank;
	int num_process;
	double start, finish;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_process);

	if (rank == 0) {
		m_reset();
		start = MPI_Wtime();
	}
	for (int i = 0; i < N; i++) {
		MPI_Bcast(M[i], N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}
	for (int k = 0; k < N; k++) {
		//if(rank==0)
			//cout << "k = " << k << endl;
		int j;
		for (j = k+1; j < N; j++) {
			if (rank == j % (num_process - 1) + 1) {
				M[k][j] = M[k][j]/M[k][k];
				for (int i = k+1; i < N; i++) {
					M[i][j] = M[i][j] - M[i][k] * M[k][j];
				}
				for(int l = k;l<N;l++)
					MPI_Send(&M[l][j], 1, MPI_FLOAT, 0, j, MPI_COMM_WORLD);
			}
			//如果是阻塞通信，必须一列一列接受，否则不知道收到rank0的哪一列
			if (rank == 0) {
				for (int l = k; l < N; l++)
					MPI_Recv(&M[l][j], 1, MPI_FLOAT, MPI_ANY_SOURCE, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
	
		//MPI_Barrier(MPI_COMM_WORLD);
		for (int i = k + 1; i < N; i++) {
			M[i][k] = 0;
			MPI_Bcast(&M[i][k], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}
		M[k][k] = 1;
		for (int i = k + 1; i < N; i++) {
			MPI_Bcast(&M[i][k+1], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}
		//MPI_Barrier(MPI_COMM_WORLD);
	}
	if (rank == 0) {
		finish = MPI_Wtime();
		/*for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				cout << M[i][j] << " ";
			}
			cout << endl;
		}*/
		cout << "mpi_col_line_innercomm =  " << (finish - start) * 1000 << " ms" << endl;
	}
}

void mpi_col_line_outerComm() {
	int rank;
	int num_process;
	double start, finish;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_process);

	if (rank == 0) {
		m_reset();
		start = MPI_Wtime();
	}
	for (int i = 0; i < N; i++) {
		MPI_Bcast(M[i], N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}
	for (int k = 0; k < N; k++) {
		//if(rank==0)
			//cout << "k = " << k << endl;
		int j;
		for (j = k + 1; j < N; j++) {
			if (rank == j % (num_process - 1) + 1) {
				M[k][j] = M[k][j] / M[k][k];
				for (int i = k + 1; i < N; i++) {
					M[i][j] = M[i][j] - M[i][k] * M[k][j];
				}
			}
		}
		for (j = k + 1; j < N; j++) {
			if (rank == j % (num_process - 1) + 1) {
				for (int l = k; l < N; l++)
					MPI_Send(&M[l][j], 1, MPI_FLOAT, 0, j, MPI_COMM_WORLD);
			}
			//如果是阻塞通信，必须一列一列接受，否则不知道收到rank0的哪一列
			if (rank == 0) {
				for (int l = k; l < N; l++)
					MPI_Recv(&M[l][j], 1, MPI_FLOAT, MPI_ANY_SOURCE, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		for (int i = k + 1; i < N; i++) {
			M[i][k] = 0;
			MPI_Bcast(&M[i][k], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}
		M[k][k] = 1;
		for (int i = k + 1; i < N; i++) {
			MPI_Bcast(&M[i][k + 1], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}
		//MPI_Barrier(MPI_COMM_WORLD);
	}
	if (rank == 0) {
		finish = MPI_Wtime();
		/*for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				cout << M[i][j] << " ";
			}
			cout << endl;
		}*/
		cout << "mpi_col_line_outercomm =  " << (finish - start) * 1000 << " ms" << endl;
	}
}

void mpi_col_block() {
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
		if (rank == num_process - 1)
			cout << "k=" << k << endl;
		int start = max(my_start, k + 1);
		for (int j = start; j < my_end; j++) {
			//除法，可以考虑按行做，这里是按列做，因为j已经确定
			M[k][j] /= M[k][k];
			for (int i = k + 1; i < N; i++) {
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}
		}
		for (int i = k + 1; i < N; i++)
			M[i][k] = 0;
		M[k][k] = 1;
		//k+1列要广播到其他进程,首先要计算第k+1列属于哪个进程：(k+1)/my_n除不尽就正确，正好整除就要加1
		if (k + 1 < N) {
			int root = 0;
			for (int l = 0; l < num_process; l++) {
				if (l * my_n <= k + 1 && k + 1 < l*my_n+my_n) {
					root = l;
					break;
				}
			}
			if (k + 1 >= (num_process - 1) * my_n)
				root = num_process - 1;
			for(int i = k+1;i<N;i++)
				MPI_Bcast(&M[i][k+1], 1, MPI_FLOAT, root, MPI_COMM_WORLD);
		}
	}
	if(rank==num_process - 1){
		for (int l = 0; l < num_process-1; l++) {
			for(int i = 0;i<N;i++)
				MPI_Recv(&M[i][l*my_n], my_n, MPI_FLOAT, l, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}
	else {
		for(int i = 0;i<N;i++)
			MPI_Send(&M[i][my_start], my_n, MPI_FLOAT, num_process-1, i, MPI_COMM_WORLD);	
	}
	if (rank == num_process - 1) {
		finish = MPI_Wtime();
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				cout << M[i][j] << " ";
			}
			cout << endl;
		}
		cout << "mpi_col_block = " << (finish - start) * 1000 << " ms" << endl;
	}
}

void mpi_row_block_simd() {
	__m128 t1, t2, t3, t4;
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
			float tmp[4] = { M[k][k], M[k][k], M[k][k], M[k][k] };
			t1 = _mm_loadu_ps(tmp);
			for (int j = N - 4; j >= k; j -= 4) {
				t2 = _mm_loadu_ps(M[k] + j);
				t3 = _mm_div_ps(t2, t1);
				_mm_storeu_ps(M[k] + j, t3);
			}
			if (k % 4 != (N % 4)) {
				for (int j = k; j % 4 != (N % 4); j++) {
					M[k][j] = M[k][j] / tmp[0];
				}
			}
			for (int j = (N % 4) - 1; j >= 0; j--) {
				M[k][j] = M[k][j] / tmp[0];
			}
		}
		M[k][k] = 1;
		//广播+同步
		MPI_Bcast(M[k], N - k, MPI_FLOAT, num_process - 1, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		//从第k+1行开始消去，前面的进程没有任务则退出
		for (int i = k+1; i < my_end; i++) {
			float tmp[4] = { M[i][k], M[i][k], M[i][k], M[i][k] };
			t1 = _mm_loadu_ps(tmp);
			for (int j = N - 4; j > k; j -= 4) {
				t2 = _mm_loadu_ps(M[i] + j);
				t3 = _mm_loadu_ps(M[k] + j);
				t4 = _mm_sub_ps(t2, _mm_mul_ps(t1, t3));
				_mm_storeu_ps(M[i] + j, t4);
			}
			for (int j = k + 1; j % 4 != (N % 4); j++) {
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}
			M[i][k] = 0;
		}
		//某几行消去完成，需要将数据同步到进程
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
		/*for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				cout << M[i][j] << " ";
			}
			cout << endl;
		}*/
		cout << "mpi_row_block_simd = " << (finish - start) * 1000 << " ms" << endl;
	}
}

int main(int argc, char* argv[]) {
	MPI_Init(NULL, NULL);
	init();
	ori();
	//mpi_row_block();
	//mpi_row_block_2();
	//mpi_row_line();
	//g_mpi_col_block();
	//g_mpi_col_line();
	//mpi_col_line_innerComm();
	//mpi_col_line_outerComm();
	//mpi_col_block();
	//mpi_row_block_simd();
	MPI_Finalize();
	release();
	return 0;
}