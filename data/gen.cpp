#include <iostream>
#include <random>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <fstream>
#include <set>

struct myComp {
	bool operator() (const std::pair<int, int> &a, const std::pair<int, int> &b)
	{
        if(a.first == b.first) {
            return a.second < b.second;
        }
		return a.first < b.first;	
	}
};

int main(int argc, const char** argv) {
    if(argc != 3) {
        std::cout << "usage: exe NNZs dimension" << std::endl;
    }
    int nnzs = atoi(argv[1]);
    int dimension = atoi(argv[2]);
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, dimension - 1);
    std::set<std::pair<int, int>, myComp> keep_unique;
    for (int i = 0; i < nnzs; ) {
        int u = distribution(generator);
        int v = distribution(generator);
        if(keep_unique.find({u, v}) == keep_unique.end()) {
            i++;
            keep_unique.insert({u, v});
        }
    }
    std::cout << dimension << " " << dimension << " " << nnzs << "\n";
    for(auto i : keep_unique) {
        std::cout << i.first << " " << i.second << " " << 1 << std::endl;
    }

    return 0;
}