#pragma once
#include <stdexcept>
#include <string>

namespace ftxj {
class FileOpenError : public std::exception {
public:
  FileOpenError() : message("file open failed") {}
  FileOpenError(const std::string &filename)
      : message(filename + " can't open!") {}
  const char *what() const throw() { return message.c_str(); }
  std::string message;
};
} // namespace ftxj
