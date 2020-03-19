#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cusparse.h>
#include <cublas_v2.h>
#include <vector>
#include "ftxj.h"
#include "cuftxj.h"

#define CUSPARSE_CHECK(x) {cusparseStatus_t _c=x; if (_c != CUSPARSE_STATUS_SUCCESS) {printf("cusparse fail: %d, line: %d\n", (int)_c, __LINE__); exit(-1);}}

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


int main(int argc, char* argv[])
{

	std::string filename_a("D:/ProgramProject/ftxj/data/Rmat_79730_100000_57_20_20.mtx");
	std::string filename_b("D:/ProgramProject/ftxj/data/Rmat_79730_100000_57_20_20.mtx");
	std::string filename_c("D:/ProgramProject/ftxj/data/res/Rmat_9987_100000_57_20_20.mtx.cusparse");

	ftxj::csr_matrix<int, double> A;
	ftxj::io::read_file(A, filename_a);
	ftxj::csr_matrix<int, double> B;
	ftxj::io::read_file(B, filename_b);
	cusparseStatus_t stat;
	cusparseHandle_t hndl;

	int* csrOffsetA, * csrColIdxA, * csrOffsetB, * csrColIdxB, * csrOffsetC, * csrColIdxC;
	double* csrValA, * csrValB, * csrValC;

	int* h_csrOffsetC, * h_csrColIdxC;
	double* h_csrValC;

	// malloc GPU memory for A and B
	cudaMalloc(&csrOffsetA, A.num_offsets * sizeof(int));
	cudaMalloc(&csrColIdxA, A.num_entries * sizeof(int));
	cudaMalloc(&csrValA, A.num_entries * sizeof(double));
	cudaMalloc(&csrOffsetB, B.num_offsets * sizeof(int));
	cudaMalloc(&csrColIdxB, B.num_entries * sizeof(int));
	cudaMalloc(&csrValB, B.num_entries * sizeof(double));
	cudaCheckErrors("cudaMalloc fail");
	// copy data of A and B from host memory to GPU memory
	cudaMemcpy(csrOffsetA, &A.row_offsets[0], A.num_offsets * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(csrColIdxA, &A.column_indices[0], A.num_entries * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(csrValA, &A.values_array[0], A.num_entries * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(csrOffsetB, &B.row_offsets[0], B.num_offsets * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(csrColIdxB, &B.column_indices[0], B.num_entries * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(csrValB, &B.values_array[0], B.num_entries * sizeof(double), cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy fail");
	// set shape and properites of matrix A B C
	cusparseMatDescr_t descrA, descrB, descrC;
	CUSPARSE_CHECK(cusparseCreate(&hndl));
	stat = cusparseCreateMatDescr(&descrA);
	stat = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	stat = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
	CUSPARSE_CHECK(stat);
	stat = cusparseCreateMatDescr(&descrB);
	stat = cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
	stat = cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO);
	CUSPARSE_CHECK(stat);
	stat = cusparseCreateMatDescr(&descrC);
	stat = cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
	stat = cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
	CUSPARSE_CHECK(stat);
	// set op  for {C = op(A)op(B)}
	cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
	cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
	int baseC, nnzC;
	// nnzTotalDevHostPtr points to host memory
	int* nnzTotalDevHostPtr = &nnzC;
	stat = cusparseSetPointerMode(hndl, CUSPARSE_POINTER_MODE_HOST);
	CUSPARSE_CHECK(stat);
	cudaMalloc((void**)& csrOffsetC, sizeof(int) * A.num_offsets);
	cudaCheckErrors("cudaMalloc fail");
	ftxj::cuExt::cuTimer first_step, second_step;
	// spgemm first step, caculate csrOffsetC
	first_step.Start();
	stat = cusparseXcsrgemmNnz(hndl, transA, transB, 
		A.num_rows, B.num_cols, A.num_cols,
		descrA, A.num_entries, csrOffsetA, csrColIdxA,
		descrB, B.num_entries, csrOffsetB, csrColIdxB,
		descrC, csrOffsetC, nnzTotalDevHostPtr);
	first_step.Stop();
	CUSPARSE_CHECK(stat);
	if (NULL != nnzTotalDevHostPtr) {
		nnzC = *nnzTotalDevHostPtr;
	}
	else {
		cudaMemcpy(&nnzC, csrOffsetC + A.num_offsets, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&baseC, csrOffsetC, sizeof(int), cudaMemcpyDeviceToHost);
		nnzC -= baseC;
	}
	// spgemm second step, caculate gemm
	cudaMalloc((void**)& csrColIdxC, sizeof(int) * nnzC);
	cudaMalloc((void**)& csrValC, sizeof(double) * nnzC);
	second_step.Start();
	stat = cusparseDcsrgemm(hndl, transA, transB, 
		A.num_rows, B.num_cols, A.num_cols,
		descrA, A.num_entries, csrValA, csrOffsetA, csrColIdxA,
		descrB, B.num_entries, csrValB, csrOffsetB, csrColIdxB,
		descrC, csrValC, csrOffsetC, csrColIdxC);
	second_step.Stop();
	CUSPARSE_CHECK(stat);
	h_csrOffsetC = (int*)malloc(A.num_offsets * sizeof(int) );
	h_csrColIdxC= (int*)malloc(nnzC * sizeof(int));
	h_csrValC = (double*)malloc(nnzC * sizeof(double) );
	std::cout << "spgemm (" 
		<< A.num_rows << "," << A.num_cols << "," << B.num_cols << ") (" 
		<<A.num_entries << "," << B.num_entries << "," << nnzC << ") " 
		<< "(" << first_step.Elapsed() << "," << second_step.Elapsed()
		<< "," << first_step.Elapsed() + second_step.Elapsed() <<  "s)" << std::endl;
	cudaMemcpy(h_csrOffsetC, csrOffsetC, A.num_offsets * sizeof(int), cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy fail");
	cudaMemcpy(h_csrColIdxC, csrColIdxC, nnzC * sizeof(int), cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy fail");
	cudaMemcpy(h_csrValC, csrValC, nnzC * sizeof(double), cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy fail");
	h_csrOffsetC[A.num_offsets - 1] = nnzC;
	ftxj::csr_matrix<int, double> C(h_csrOffsetC, h_csrColIdxC, h_csrValC, A.num_rows, B.num_cols, nnzC);
	ftxj::io::output_file(C, filename_c);
	return 0;
}