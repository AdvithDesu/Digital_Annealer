// ==== utils.hpp (rewritten for dense CSV J,h loading) ====
#ifndef UTILS_HPP
#define UTILS_HPP


#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <functional>
#include <memory>
#include <sys/time.h>


using std::string;


double rtclock();
void printtime(const char *str, double starttime, double endtime);


class ParseData
{
public:
// Reads dense NxN J matrix
ParseData(const string filename, std::vector<float>& adjMat);


// Reads h vector
void readLinearValues(const string filename, std::vector<float>& linearVect);


std::vector<unsigned int> getDataDims() const;


private:
std::unique_ptr<std::ifstream, std::function<void(std::ifstream*)>> _pifstream;
std::vector<unsigned int> _data_dims; // {N, N}


void readDenseJ(std::vector<float>& adjMat);
};


#endif
