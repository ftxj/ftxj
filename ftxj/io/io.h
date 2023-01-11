#pragma once
#include "ftxj/exception/exception.h"
#include "ftxj/format.h"
#include "ftxj/matrix/matrix_format.h"
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <tuple>

namespace ftxj {

namespace io {

template <typename Matrix, typename Stream>
void read_file(Matrix &mtx, Stream &input, ftxj::csc_format);
template <typename Matrix, typename Stream>
void read_file(Matrix &mtx, Stream &input, ftxj::csr_format);
template <typename Matrix>
void read_file(Matrix &mtx, const std::string &filename);
template <typename Matrix> void read_file(Matrix &mtx, std::istream &input);
template <typename IndexType>
void shrink_array(std::vector<IndexType> &oldV, std::vector<IndexType> &newV,
                  size_t num);
template <typename Matrix, typename IndexType, typename ValueType,
          typename Stream, typename T>
void basic_read(Matrix &mtx, std::vector<IndexType> &rows,
                std::vector<IndexType> &cols, std::vector<ValueType> &values,
                Stream &input, T s);
template <typename Matrix>
void output_file(Matrix &mtx, const std::string &filename);
template <typename Matrix, typename Stream>
void output_file(Matrix &mtx, Stream &output, ftxj::csr_format);
template <typename Matrix, typename Stream>
void output_file(Matrix &mtx, Stream &output, ftxj::csc_format);

template <typename Matrix, typename Stream>
void read_file(Matrix &mtx, Stream &input, ftxj::csc_format) {
  typedef std::tuple<Matrix::index_type, Matrix::index_type, Matrix::value_type>
      edge;
  auto cmp = [](const edge &x, const edge &y) {
    if (std::get<1>(x) < std::get<1>(y)) {
      return true;
    } else if (std::get<1>(x) == std::get<1>(y)) {
      return std::get<0>(x) < std::get<0>(y);
    } else {
      return false;
    }
  };
  std::set<edge, decltype(cmp)> s(cmp);
  std::vector<Matrix::index_type> cols;
  basic_read(mtx, mtx.row_indices, cols, mtx.values_array, input, s);
  shrink_array(cols, mtx.column_offsets, mtx.values_array.size() + 1);
  mtx.num_offsets = mtx.num_cols + 1;
}

template <typename Matrix, typename Stream>
void read_file(Matrix &mtx, Stream &input, ftxj::csr_format) {
  std::string line;
  do {
    std::getline(input, line);
  } while (line[0] == '%');

  size_t num_rows, num_cols, num_nnzs;
  std::vector<std::string> tokens;
  ftxj::estd::split(tokens, line);
  if (tokens.size() != 3)
    throw "csr format matrix error on headline";
  std::istringstream(tokens[0]) >> num_rows;
  std::istringstream(tokens[1]) >> num_cols;
  std::istringstream(tokens[2]) >> num_nnzs;
  mtx.set_nums(num_rows, num_cols, num_nnzs, num_rows + 1);
  size_t num_entries_read = 0;
  size_t num_row_read = 0;
  mtx.row_offsets.push_back(0);
  while (num_entries_read < num_nnzs && !input.eof()) {
    if (num_entries_read % 99999 == 0) {
      // std::cout << "read 100000 values" << std::endl;
    }
    typename Matrix::index_type a, b;
    typename Matrix::value_type c;
    std::getline(input, line);
    std::vector<std::string> tokens;
    ftxj::estd::split(tokens, line);
    std::istringstream(tokens[0]) >> a;
    std::istringstream(tokens[1]) >> b;
    std::istringstream(tokens[2]) >> c;
    while (a != num_row_read) {
      mtx.row_offsets.push_back(num_entries_read);
      num_row_read++;
    }
    mtx.column_indices.push_back(b);
    mtx.values_array.push_back(c);
    num_entries_read++;
  }
  while (mtx.row_offsets.size() < num_rows + 1) {
    mtx.row_offsets.push_back(num_entries_read);
  }
}

template <typename Matrix>
void read_file(Matrix &mtx, const std::string &filename) {
  std::ifstream file(filename.c_str());
  if (!file) {
    // TODO();
    throw FileOpenError(filename);
  }
  read_file(mtx, file, typename Matrix::format());
}

template <typename Matrix> void read_file(Matrix &mtx, std::istream &input) {
  read_file(mtx, input, typename Matrix::format());
}

template <typename IndexType>
void shrink_array(std::vector<IndexType> &oldV, std::vector<IndexType> &newV,
                  size_t num) {
  IndexType begin = 0;
  IndexType acc = 0;
  newV.push_back(acc);
  for (auto i : oldV) {
    while (i != begin) {
      newV.push_back(acc);
      begin++;
    }
    acc++;
  }
  while (newV.size() != num) {
    newV.push_back(acc);
  }
}

template <typename Matrix, typename IndexType, typename ValueType,
          typename Stream, typename T>
void basic_read(Matrix &mtx, std::vector<IndexType> &rows,
                std::vector<IndexType> &cols, std::vector<ValueType> &values,
                Stream &input, T s) {
  std::string line;
  do {
    std::getline(input, line);
  } while (line[0] == '%');
  size_t num_rows, num_cols, num_nnzs;
  std::vector<std::string> tokens;
  ftxj::estd::split(tokens, line);
  if (tokens.size() != 3)
    throw "mtx format matrix error on headline";
  std::istringstream(tokens[0]) >> num_rows;
  std::istringstream(tokens[1]) >> num_cols;
  std::istringstream(tokens[2]) >> num_nnzs;
  mtx.set_nums(num_rows, num_cols, num_nnzs);
  size_t num_entries_read = 0;
  while (num_entries_read < num_nnzs && !input.eof()) {
    std::tuple<IndexType, IndexType, ValueType> e;
    std::getline(input, line);
    std::vector<std::string> tokens;
    ftxj::estd::split(tokens, line);
    std::istringstream(tokens[0]) >> std::get<0>(e);
    std::istringstream(tokens[1]) >> std::get<1>(e);
    std::istringstream(tokens[2]) >> std::get<2>(e);
    s.insert(e);
    num_entries_read++;
  }
  for (auto e : s) {
    IndexType u = std::get<0>(e);
    IndexType v = std::get<1>(e);
    ValueType w = std::get<2>(e);
    rows.push_back(u);
    cols.push_back(v);
    values.push_back(w);
  }
}

template <typename Matrix>
void output_file(Matrix &mtx, const std::string &filename) {
  std::ofstream file(filename.c_str());
  if (!file) {
    // TODO();
  }
  output_file(mtx, file, typename Matrix::format());
}
template <typename Matrix> void output_file(Matrix &mtx, std::ostream &out) {
  output_file(mtx, out, typename Matrix::format());
}

template <typename Matrix, typename Stream>
void output_file(Matrix &mtx, Stream &output, ftxj::csr_format) {
  size_t num_entries = mtx.num_entries;
  size_t num_rows = mtx.num_rows;
  size_t num_cols = mtx.num_cols;
  for (size_t i = 0; i < num_rows; ++i) {
    size_t number = mtx.entries_in_row(i);
    output << "row " << i << "," << number << ":";
    for (size_t j = 0; j < number; ++j) {
      output << mtx.column_indices[mtx.row_offsets[i] + j] << "-"
             << mtx.values_array[mtx.row_offsets[i] + j];
      if (j + 1 != number)
        output << "\n";
    }
    output << "\n";
  }
}

template <typename Matrix, typename Stream>
void output_file(Matrix &mtx, Stream &output, ftxj::csc_format) {
  size_t num_entries = mtx.num_entries;
  size_t num_rows = mtx.num_rows;
  size_t num_cols = mtx.num_cols;
  for (size_t i = 0; i < num_cols; ++i) {
    size_t number = mtx.column_offsets[i + 1] - mtx.column_offsets[i];
    output << "col " << i << "," << number << ":\n";
    for (size_t j = 0; j < number; ++j) {
      output << mtx.row_indices[mtx.column_offsets[i] + j] << "-"
             << mtx.values_array[mtx.column_offsets[i] + j];
      if (j % 9 == 0)
        output << "\n";
      if (j + 1 != number)
        output << ",";
    }
    output << "\n";
  }
}

} // namespace io
} // namespace ftxj