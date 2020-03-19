#include <iostream>
#include "ftxj.h"
#include "mkl_spblas.h"
#include "mkl.h"
int main(int argc, char** argv)
{
	std::string filename_a("D:/ProgramProject/ftxj/data/Ran_10000000_1048576.mtx");
	std::string filename_b("D:/ProgramProject/ftxj/data/Ran_10000000_1048576.mtx");
	std::string filename_c("D:/ProgramProject/ftxj/data/res/Rmat_9987_100000_57_20_20.mtx.mkl");
	
	mkl_set_num_threads(4);

	ftxj::csr_matrix<int, double> A;
	ftxj::io::read_file(A, filename_a);
	std::cout << "read A" << std::endl;

	ftxj::csr_matrix<int, double> B;
	ftxj::io::read_file(B, filename_b);
	std::cout << "read B" << std::endl;
	
	sparse_matrix_t mkl_A;
	sparse_matrix_t mkl_B;
	sparse_matrix_t mkl_C;

	if (SPARSE_STATUS_SUCCESS != mkl_sparse_d_create_csr(&mkl_A, SPARSE_INDEX_BASE_ZERO,
		A.num_rows, A.num_cols,
		&A.row_offsets[0], &A.row_offsets[1], &A.column_indices[0], &A.values_array[0]))
	{
		std::cout << "error mkl A" << std::endl;
	}

	if (SPARSE_STATUS_SUCCESS != mkl_sparse_d_create_csr(&mkl_B, SPARSE_INDEX_BASE_ZERO,
		B.num_rows, B.num_cols,
		&B.row_offsets[0], &B.row_offsets[1], &B.column_indices[0], &B.values_array[0]))
	{
		std::cout << "error mkl B" << std::endl;
	}
	sparse_operation_t operation = SPARSE_OPERATION_NON_TRANSPOSE;

	ftxj::system::timer T;
	T.Start();
	if(SPARSE_STATUS_SUCCESS != mkl_sparse_spmm(operation, mkl_A, mkl_B, &mkl_C)) {
		std::cout << "error spmm " << std::endl;
	}
	T.Stop();
	std::cout << "mkl gemm time=" << T.Elapsed() << std::endl;

	sparse_index_base_t c_indexing;
	double* values;
	MKL_INT c_rows, c_cols, * rows_start, * rows_end, * columns;
	if (SPARSE_STATUS_SUCCESS != mkl_sparse_d_export_csr(mkl_C, &c_indexing, 
		&c_rows, &c_cols, &rows_start, &rows_end, &columns, &values)) 
	{
		std::cout << "error csr out" << std::endl;
	}
	if (SPARSE_INDEX_BASE_ZERO != c_indexing) {
		throw std::runtime_error("C is not zero based indexed\n");
		return 0;
	}
	//ftxj::csr_matrix<int, double> C(rows_start, rows_end, columns, values, c_rows, c_cols);
	//ftxj::io::output_file(C, filename_c);
	return 0;
}