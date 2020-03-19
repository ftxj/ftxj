#include "ftxj/matrix/matrix_format.h"
#include "ftxj/format.h"
#include "ftxj/system/timer.h"

namespace ftxj {
	namespace algorithm {
		// template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC>
		// void spgemm(MatrixTypeA &A, MatrixTypeB &B, MatrixTypeC &C) {
		//     spgemm(A, B, C, MatrixTypeA::format(), MatrixTypeB::format(), MatrixTypeC::format());
		// }

		bool equalZeros(int a) {
			return a == 0;
		}
		bool equalZeros(double a) {
			if (a > 0) return a < 1e-9;
			else return a > -1e-9;
		}

		template<typename IndexType, typename ValueType>
		void spgemm(csr_matrix<IndexType, ValueType>& A, csc_matrix<IndexType, ValueType>& B, csr_matrix<IndexType, ValueType>& C) {
			C.num_cols = B.num_cols;
			C.num_rows = A.num_rows;
			C.num_offsets = A.num_offsets;

			for (size_t i = 0; i < A.num_rows; ++i) {
				IndexType A_row_index = A.row_offsets[i];
				IndexType A_row_number = A.row_offsets[i + 1] - A_row_index;
				C.row_offsets.push_back(C.column_indices.size());
				if (A_row_number == 0) {
					continue;
				}
				for (size_t j = 0; j < B.num_cols; ++j) {
					ValueType c = 0;
					IndexType B_col_index = B.column_offsets[j];
					IndexType B_col_number = B.column_offsets[j + 1] - B_col_index;
					size_t A_count = 0, B_count = 0;
					while (A_count != A_row_number && B_count != B_col_number) {
						IndexType aindex = A.column_indices[A_row_index + A_count];
						IndexType bindex = B.row_indices[B_col_index + B_count];
						if (aindex == bindex) {
							ValueType a = A.values_array[A_row_index + A_count];
							ValueType b = B.values_array[B_col_index + B_count];
							c += a * b;
							A_count++, B_count++;
						}
						else if (aindex > bindex) {
							B_count++;
						}
						else {
							A_count++;
						}
					}
					if (!equalZeros(c)) {
						C.column_indices.push_back(j);
						C.values_array.push_back(c);
					}
				}
			}
			C.row_offsets.push_back(C.column_indices.size());
			C.num_entries = C.values_array.size();

		}


		template<typename node, typename Compare>
		std::vector<node> merge_with_reduce(std::vector<node>& l1, std::vector<node>& l2, Compare cmp) {
			auto first1 = l1.begin();
			auto first2 = l2.begin();
			auto last1 = l1.end();
			auto last2 = l2.end();
			std::vector<node> res;
			while (first1 != last1 && first2 != last2) {
				int cmp_res = cmp(*first1, *first2);
				if (cmp_res < 0) {
					res.push_back(*first1);
					first1++;
				}
				else if (cmp_res == 0) {
					res.push_back({ first1->index, first1->value + first2->value });
					first1++, first2++;

				}
				else {
					res.push_back(*first2);
					first2++;
				}
			}
			while (first2 != last2) {
				res.push_back(*first2);
				first2++;
			}
			while (first1 != last1) {
				res.push_back(*first1);
				first1++;
			}
			return res;
		}


		template<typename IndexType, typename ValueType>
		std::pair<double, double> outer_spgemm(csc_matrix<IndexType, ValueType>& A, csr_matrix<IndexType, ValueType>& B, csr_matrix<IndexType, ValueType>& C) {
			struct node {
				IndexType index;
				ValueType value;
			};
			auto cmp = [](const node& x, const node& y) {
				if (x.index > y.index) return 1;
				else if (x.index == y.index) return 0;
				else return -1;
			};
			
			std::vector<std::list<std::vector<node>>> temp_matrixs(A.num_cols);
			std::vector<std::vector<size_t>> merge_ordered(A.num_cols);

			ftxj::system::timer T_multiply, T_merge;
			T_multiply.Start();
			size_t xj = 0;
			int need_merge = 0;
			for (size_t i = 0; i < A.num_cols; ++i) {
				IndexType A_ith_col_index = A.column_offsets[i];
				IndexType A_ith_col_number = A.column_offsets[i + 1] - A_ith_col_index;
				IndexType B_ith_row_index = B.row_offsets[i];
				IndexType B_ith_row_number = B.row_offsets[i + 1] - B_ith_row_index;
				if (A_ith_col_number == 0 || B_ith_row_number == 0) continue;
				for (size_t j = 0; j < A_ith_col_number; ++j) {
					ValueType a = A.values_array[A_ith_col_index + j];
					IndexType aindex = A.row_indices[A_ith_col_index + j];
					std::vector<node> temp_vec(B_ith_row_number);
					for (int k = 0; k < B_ith_row_number; ++k) {
						ValueType b = B.values_array[B_ith_row_index + k];
						IndexType bindex = B.column_indices[B_ith_row_index + k];
						ValueType c = a * b;
						temp_vec[k] = { bindex, c };
						xj++;
					}
					temp_matrixs[aindex].push_back(temp_vec);
					need_merge++;
				}
			}
			T_multiply.Stop();
			T_merge.Start();
			std::cout << "need merge = " << need_merge << std:: endl;
			
			// for (size_t i = 0; i < temp_matrixs.size(); ++i) {
			// 	for (size_t j = 0; j < temp_matrix[i].size()) {
			// 		size_t vec_size = temp_matrixs
			// 	}
			// }

			auto vecs = temp_matrixs.begin();
			for (size_t i = 0; i < temp_matrixs.size(); ++i) {
				std::vector<node> v;
				for (auto vec : *vecs) {
					v = merge_with_reduce(v, vec, cmp);
				}
				vecs++;
				temp_matrixs[i].push_front(v);
			}

			T_merge.Stop();
			for (int i = 0; i < A.num_cols; ++i) {
				C.row_offsets.push_back(C.values_array.size());
				if (temp_matrixs[i].size() != 0) {
					std::vector<node>& e = *(temp_matrixs[i].begin());
					for (auto iter : e) {
						C.values_array.push_back(iter.value);
						C.column_indices.push_back(iter.index);
					}
				}
			}
			C.row_offsets.push_back(C.values_array.size());
			C.set_nums(A.num_rows, B.num_cols, C.values_array.size(), A.num_rows + 1);
			return { T_multiply.Elapsed(), T_merge.Elapsed() };
		}


        template<typename IndexType, typename ValueType>
		std::pair<double, double> outer_spgemm_with_condensing(csr_matrix<IndexType, ValueType>& A, csr_matrix<IndexType, ValueType>& B, csr_matrix<IndexType, ValueType>& C) {
			struct node {
				IndexType index;
				ValueType value;
			};
			auto cmp = [](const node& x, const node& y) {
				if (x.index > y.index) return 1;
				else if (x.index == y.index) return 0;
				else return -1;
			};
			ftxj::system::timer T_multiply, T_merge;
            int ROUND_WIDTH = 64;
            int need_merge = 0;
			T_multiply.Start();
			int xj = 0;
			std::vector<std::list<std::vector<node>>> temp_matrixs(A.num_cols);
            for(size_t round_index = 0; round_index < A.num_cols; ++round_index) {
				for(size_t col_index = 0; col_index < ROUND_WIDTH; ++col_index) {
                    size_t now_condensing_col_index = col_index + ROUND_WIDTH * round_index;
                    std::vector<ValueType> now_col_value;
                    std::vector<IndexType> now_col_index;
					std::vector<IndexType> row_index;
                    for(size_t i = 0; i < A.num_rows; ++i) {
						if (A.row_offsets[i + 1] - A.row_offsets[i] > now_condensing_col_index) {
							now_col_value.push_back(A.values_array[A.row_offsets[i] + now_condensing_col_index]);
							now_col_index.push_back(A.column_indices[A.row_offsets[i] + now_condensing_col_index]);
							row_index.push_back(i);
						}
                    }
                    if(now_col_index.size() ==0) {
                        round_index = A.num_cols;
                        break;
                    }
					//std::vector<size_t> order = ftxj::estd::sort_indexes(&now_col_index[0], now_col_index.size());
					for (size_t i = 0; i < now_col_index.size(); ++i) {
						IndexType item_col = now_col_index[i];
						size_t number = B.row_offsets[item_col + 1] - B.row_offsets[item_col];
						IndexType aindex = row_index[i];
						if (number == 0) continue;
						std::vector<node> temp_vec(number);
						for (size_t j = 0; j < number; ++j) {
							ValueType c = now_col_value[i] * B.values_array[B.row_offsets[item_col] + j];
							temp_vec[j] = { B.column_indices[B.row_offsets[item_col] + j], c };
							xj++;
						}
						need_merge++;
						//std::cout << temp_vec.size() << std::endl;
						if (temp_matrixs[aindex].size() != 0) {
							temp_vec = merge_with_reduce(temp_vec, *temp_matrixs[aindex].begin(), cmp);
						}
						temp_matrixs[aindex].push_front(temp_vec);
					}
                }
            }
			T_multiply.Stop();
			/*T_merge.Start();
			std::cout << "need merge = " << need_merge << std:: endl;
			int cc = 0;
			auto vecs = temp_matrixs.begin();
			for (size_t i = 0; i < temp_matrixs.size(); ++i) {
				std::vector<node> v;
				for (auto vec : *vecs) {
					cc++;
					v = merge_with_reduce(v, vec, cmp);
				}
				vecs++;
				temp_matrixs[i].insert(temp_matrixs[i].begin(), v);
			}
			T_merge.Stop();*/
			for (int i = 0; i < A.num_cols; ++i) {
				C.row_offsets.push_back(C.values_array.size());
				if (temp_matrixs[i].size() != 0) {
					std::vector<node>& e = *(temp_matrixs[i].begin());
					for (auto iter : e) {
						C.values_array.push_back(iter.value);
						C.column_indices.push_back(iter.index);
					}
				}
			}
			C.row_offsets.push_back(C.values_array.size());
			C.set_nums(A.num_rows, B.num_cols, C.values_array.size(), A.num_rows + 1);
			return { T_multiply.Elapsed(), T_merge.Elapsed() };
		}

	} // namespace algorithm
}
