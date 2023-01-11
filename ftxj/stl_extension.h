#pragma once

#include <algorithm>
#include <iostream>
#include <list>
#include <numeric>
#include <random>
#include <string>
#include <vector>
namespace ftxj {
namespace estd {
void split(std::vector<std::string> &tokens, const std::string &str,
           const std::string &delimiters = "\n\r\t ") {
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  std::string::size_type pos = str.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos) {
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    lastPos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, lastPos);
  }
}

template <typename T> std::vector<size_t> sort_indexes(T *begin, size_t size) {

  std::vector<size_t> idx(size);
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  stable_sort(idx.begin(), idx.end(),
              [&begin](size_t i1, size_t i2) { return begin[i1] < begin[i2]; });

  return idx;
}
} // namespace estd
} // namespace ftxj
