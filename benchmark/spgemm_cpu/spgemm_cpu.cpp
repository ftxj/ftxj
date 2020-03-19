#include <iostream>
#include "ftxj.h"
#include "ftxj/system/timer.h"

int main(int argc, char* argv[])
{
	std::string filename_a("D:/ProgramProject/ftxj/data/Rmat_79730_100000_57_20_20.mtx");
	std::string filename_b("D:/ProgramProject/ftxj/data/Rmat_79730_100000_57_20_20.mtx");
	std::string filename_c("D:/ProgramProject/ftxj/data/res/Rmat_79730_100000_57_20_20.mtx.cpu_inner");
	std::string filename_d("D:/ProgramProject/ftxj/data/res/Rmat_79730_100000_57_20_20.mtx.cpu_outer");
	std::string filename_e("D:/ProgramProject/ftxj/data/res/Rmat_79730_100000_57_20_20.mtx.cpu_outer_condensing");

	//
	ftxj::csr_matrix<int, int> A;
	ftxj::io::read_file(A, filename_a);
	std::cout << "read A" << std::endl;

	ftxj::csc_matrix<int, int> A_csc;
	ftxj::io::read_file(A_csc, filename_a);
	std::cout << "read A csc" << std::endl;
	
	ftxj::csr_matrix<int, int> B;
	ftxj::io::read_file(B, filename_b);
	std::cout << "read B" << std::endl;

	ftxj::csc_matrix<int, int> B_csc;
	ftxj::io::read_file(B_csc, filename_b);
	std::cout << "read B csc" << std::endl;

	//ftxj::io::output_file(A, filename_d);
	//return 0;
	//ftxj::io::output_file(B, filename_d);
	ftxj::csr_matrix<int, int> C;
	ftxj::csr_matrix<int, int> D;
	ftxj::csr_matrix<int, int> E;

	ftxj::system::timer T_inner;

	std::cout << "begin inner gemm:" << std::endl;
	T_inner.Start();
	ftxj::algorithm::spgemm(A, B_csc, C);
	T_inner.Stop();
	std::cout << "compelete!" << std::endl;
	
	std::cout << "begin outer basic gemm:" << std::endl;
	std::pair<double, double> T_outer_basic = ftxj::algorithm::outer_spgemm(A_csc, B, D);
	std::cout << "compelete!" << std::endl;
	
	std::cout << "begin outer condensing gemm:" << std::endl;
	std::pair<double, double> T_outer_condensing = ftxj::algorithm::outer_spgemm_with_condensing(A, B, E);
	std::cout << "compelete!" << std::endl;
	
	if (!C.equals(D)) {
		std::cout << "fail on outer basic! " << std::endl;
		//ftxj::io::output_file(C, filename_c);
		ftxj::io::output_file(D, filename_d);
	}
	if(!C.equals(E)) {
		std::cout << "fail on outer condensing! " << std::endl;
		//ftxj::io::output_file(C, filename_c);
		ftxj::io::output_file(E, filename_e);
	}
	else {
		std::cout << "success !" << std::endl;
	}

	std::cout << "cpu inner time=" << T_inner.Elapsed() << std::endl;
	std::cout << "cpu basic outer time=" << T_outer_basic.first << "," << T_outer_basic.second << "," << T_outer_basic.first + T_outer_basic.second << std::endl;
	std::cout << "cpu condensing outer time=" << T_outer_condensing.first << "," << T_outer_condensing.second << "," << T_outer_condensing.first + T_outer_condensing.second << std::endl;

	//ftxj::io::output_file(C, filename_c);
	//ftxj::io::output_file(D, filename_d);
	return 0;
}
