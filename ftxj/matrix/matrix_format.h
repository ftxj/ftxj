#pragma once

#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <iostream>
#include <fstream>
#include "ftxj/format.h"
#include "ftxj/stl_extension.h"
namespace ftxj {


template<typename IndexType, typename ValueType, typename Format>
class base_matrix 
{
public:
	typedef IndexType   index_type;
	typedef ValueType   value_type;
	typedef Format      format;
	size_t num_rows;
	size_t num_cols;
	size_t num_entries;
	size_t num_offsets;

	base_matrix()
		: num_rows(0), num_cols(0), num_entries(0) {}
	base_matrix(size_t r, size_t c, size_t e)
		: num_rows(r), num_cols(c), num_entries(e) {}


	void set_nums(size_t num_rows, size_t num_cols, size_t num_entries) {
		this->num_cols = num_cols;
		this->num_entries = num_entries;
		this->num_rows = num_rows;
	}

	void set_nums(size_t num_rows, size_t num_cols, size_t num_entries, size_t num_offsets) {
		this->num_cols = num_cols;
		this->num_entries = num_entries;
		this->num_rows = num_rows;
		this->num_offsets = num_offsets;
	}
};



template<typename IndexType, typename ValueType>
class coo_matrix : public base_matrix<IndexType, ValueType, coo_format> 
{
	typedef typename std::vector<ValueType> values_array_type;
	typedef typename std::vector<IndexType> column_indices_array_type;
	typedef typename std::vector<IndexType> row_indices_array_type;
	values_array_type values_array;
	column_indices_array_type column_indices;
	row_indices_array_type row_indices;

	coo_matrix() : base_matrix() {}
	coo_matrix(size_t num_rows, size_t num_cols, size_t num_entries)
		: base_matrix(num_rows, num_cols, num_entries) {}
	coo_matrix(IndexType* rows, IndexType* colums, ValueType* values, size_t r, size_t c, size_t nnzs) {

	}
};


template<typename IndexType, typename ValueType>
class csr_matrix : public base_matrix<IndexType, ValueType, csr_format> 
{
public:
	typedef typename std::vector<ValueType> values_array_type;
	typedef typename std::vector<IndexType> column_indices_array_type;
	typedef typename std::vector<IndexType> row_offsets_array_type;
	typedef typename ftxj::csr_matrix<IndexType, ValueType> container;
	values_array_type values_array;
	column_indices_array_type column_indices;
	row_offsets_array_type row_offsets;

	csr_matrix() {}
	csr_matrix(size_t num_rows, size_t num_cols, size_t num_entries) 
		: base_matrix(num_rows, num_cols, num_entries) {}
	csr_matrix(IndexType* offsets, IndexType* colums, ValueType* values, size_t rows, size_t cols, size_t nnzs) {
		base_matrix::set_nums(rows, cols, nnzs, rows + 1);
		for (size_t i = 0; i < nnzs; ++i) {
			column_indices.push_back(colums[i]);
			values_array.push_back(values[i]);
		}
		for (size_t i = 0; i < rows + 1; ++i) {
			row_offsets.push_back(offsets[i]);

		}
	}
	
	csr_matrix(IndexType* rows_start, IndexType* rows_end, IndexType* colums, ValueType* values, size_t rows, size_t cols) {
		IndexType nnzs = 0;
		row_offsets.push_back(0);
		for (size_t i = 0; i < rows; ++i) {
			size_t row_ith_number = rows_end[i] - rows_start[i];
			std::vector<size_t> v = ftxj::estd::sort_indexes(&colums[rows_start[i]], row_ith_number);
			for (size_t j = 0; j != row_ith_number; ++j) {
				nnzs++;
				column_indices.push_back(colums[v[j] + rows_start[i]]);
				values_array.push_back(values[v[j] + rows_start[i]]);
			}
			row_offsets.push_back(nnzs);
		}
		this->set_nums(rows, cols, nnzs, rows + 1);
	}
	IndexType entries_in_row(size_t index) {
		return row_offsets[index + 1] - row_offsets[index];
	}
	
	bool equals(csr_matrix<IndexType, ValueType> &b) {
		if (b.num_offsets != this->num_offsets || this->num_rows != b.num_rows
			|| this->num_cols != b.num_cols || this->num_entries != b.num_entries) {
			std::cout << "num error" << std::endl;
			return false;
		}
		for (size_t i = 0; i < this->num_rows; ++i) {
			if (values_array[i] != b.values_array[i] || column_indices[i] != b.column_indices[i] 
				|| row_offsets[i] != b.row_offsets[i]) {
				std::cout << "thing error "  << i << std::endl;
				return false;
			}
		}
		return row_offsets[this->num_rows] == b.row_offsets[this->num_rows];
	}
};


template<typename IndexType, typename ValueType>
class csc_matrix : public base_matrix<IndexType, ValueType, csc_format> 
{
public:
	typedef typename std::vector<ValueType> values_array_type;
	typedef typename std::vector<IndexType> row_indices_array_type;
	typedef typename std::vector<IndexType> column_offsets_array_type;
	typedef typename ftxj::csc_matrix<IndexType, ValueType> container;
	values_array_type values_array;
	row_indices_array_type row_indices;
	column_offsets_array_type column_offsets;
	
	csc_matrix() {}
	csc_matrix(size_t num_rows, size_t num_cols, size_t num_entries)
		: base_matrix(num_rows, num_cols, num_entries) {}
	csc_matrix(IndexType* offsets, IndexType* colums, ValueType* values, size_t rows, size_t cols, size_t nnzs) {
		
	}
};



template<typename IndexType, typename ValueType>
class mix_matrix_view : public base_matrix<IndexType, ValueType, mix_format> 
{
	typedef typename std::vector<ValueType> values_array_type;
	typedef typename std::vector<IndexType> column_indices_array_type;
	typedef typename std::vector<IndexType> row_indices_array_type;
	typedef typename std::vector<IndexType> column_offset_array_type;
	typedef typename std::vector<IndexType> row_offset_array_type;
	values_array_type values_array;
	column_indices_array_type column_indices;
	row_indices_array_type row_indices;
	column_offset_array_type column_offset;
	row_offset_array_type row_offset;
};


} // namespace ftxj end