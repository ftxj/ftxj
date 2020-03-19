#include <iostream>
#include <random>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <fstream>
#include <set>
#include <chrono>

using namespace std;

int log2Up(int x) {
    if(x == 2 && x < 4) {
        return 1;
    }
    else if(x < 2) {
        return 0;
    }
    else {
        return log2Up(x/2) + 1;
    }
}

int pow2(int x) {
    return pow(2, x);
}

struct edge{
    int u;
    int v;
};
struct myComp {
	bool operator() (const edge &a, const edge &b)
	{
        if(a.u == b.u) {
            return a.v < b.v;
        }
		return a.u < b.u;	
	}
};

class RMat {
public:
    int num_nodes, num_edges;
    double a, ab, abc;
    std::default_random_engine seed;
    std::uniform_real_distribution<double> dist;
    RMat(int num_nodes, int num_edges, double a, double b, double c) :
        num_nodes(num_nodes), num_edges(num_edges), a(a), ab(a+b), abc(a+b+c),
        dist(0.0, 1.0), seed(std::chrono::system_clock::now().time_since_epoch().count()){}
    
    edge generate_next_edge(int x, int y, int n) {
        double p = dist(seed);     
        //std::cout << p << std::endl;
        if(n == 1) {
            return {x, y};
        }
        if(p < a) {
            return generate_next_edge(x, y, n / 2); 
        }
        else if(p < ab) {
            return generate_next_edge(x, y + n / 2, n / 2);
        }
        else if(p < abc) {
            return generate_next_edge(x + n / 2, y, n / 2);
        }
        else {
            return generate_next_edge(x + n / 2, y + n / 2, n / 2);
        }
    }
    edge generate_next_edge(int n) {
        if(n == 1) {
            return {0, 0};
        }
        else {
            edge e = generate_next_edge(n / 2);
            double p = dist(seed);     
            if(p < a) {
                return e; 
            }
            else if(p < ab) {
                return {e.u, e.v + n / 2};
            }
            else if(p < abc) {
                return {e.u + n / 2, e.v};
            }
            else {
                return {e.u + n / 2, e.v + n / 2};
            }
        }
    }
};  

void gen_rmat_graph(std::set<edge, myComp> &keep_unique, int nodes, int edges, double a, double b, double c) {
    
    RMat r(nodes, edges, a, b, c);
    for(int i = 0; i < edges; ) {
        edge e = r.generate_next_edge(0, 0, nodes);
        if(keep_unique.find(e) == keep_unique.end()) {
            i++;
            keep_unique.insert(e);
        }
    }
}

int main(int argc, char const *argv[])
{
    if(argc != 6) {
        std::cout << "usage: exe nodes edges a b c" << std::endl;
    }
    int nodes = atoi(argv[1]);
    nodes = pow2(log2Up(nodes));
    std::cout << "real nodes = " << nodes << std::endl;
    int edges = atoi(argv[2]);
    double a = atof(argv[3]);
    double b = atof(argv[4]);
    double c = atof(argv[5]);
    std::set<edge, myComp> keep_unique;
    gen_rmat_graph(keep_unique, nodes, edges, a, b, c);
    int ta = 0, tb = 0, tc = 0, td = 0;
    std::cout << nodes << " " << nodes << " " << edges << "\n";
    for(auto i : keep_unique) {
        std::cout << i.u << " " << i.v << " " << 1 << std::endl;
    }

    // for(auto i : keep_unique) {
    //     int u = i.u;
    //     int v = i.v;
    //     if(u < nodes / 2 && v < nodes / 2) {
    //         ta ++;
    //     }
    //     else if(u < nodes / 2) {
    //         tb ++;
    //     }
    //     else if(v < nodes / 2) {
    //         tc ++;
    //     }
    //     else {
    //         td ++;
    //     }
    // }
    // double te = edges * 1.0;
    // std::cout << ta << std::endl;
    // std::cout << "real p" << double(ta/te) << "," << double(tb/te) << "," << double(tc/te) << "," << double(td/te);
    // std::cout << "\n";
    return 0;
}
