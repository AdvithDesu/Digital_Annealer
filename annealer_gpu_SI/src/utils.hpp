// ==== utils.hpp ====
// Supports BOTH dense J (legacy) and sparse J (row_ptr, col_idx, values)

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

// -------------------- timing utils --------------------
double rtclock();
void printtime(const char *str, double starttime, double endtime);

// =====================================================
// Legacy dense J parser (UNCHANGED)
// =====================================================
class ParseData
{
public:
    // Reads dense NxN J matrix
    ParseData(const string filename, std::vector<float>& adjMat);

    // Reads h vector
    void readLinearValues(const string filename, std::vector<float>& linearVect);

    // Returns {num_spins, num_spins}
    std::vector<unsigned int> getDataDims() const;

private:
    std::unique_ptr<std::ifstream, std::function<void(std::ifstream*)>> _pifstream;
    std::vector<unsigned int> _data_dims; // {num_spins, num_spins}

    void readDenseJ(std::vector<float>& adjMat);
};

// =====================================================
// NEW sparse J parser
// =====================================================
class ParseSparseData
{
public:
    // Constructor: reads row_ptr, col_idx, values
    ParseSparseData(const string& row_ptr_file,
                    const string& col_idx_file,
                    const string& values_file);

    // Accessors
    const std::vector<int>&   getRowPtr()   const { return row_ptr; }
    const std::vector<int>&   getColIdx()   const { return col_idx; }
    const std::vector<float>& getValues()   const { return values; }

    unsigned int getNumSpins() const { return num_spins; }
    unsigned int getNNZ()      const { return nnz; }

private:
    std::vector<int>   row_ptr;
    std::vector<int>   col_idx;
    std::vector<float> values;

    unsigned int num_spins = 0;
    unsigned int nnz = 0;

    void readIntCSV(const string& filename, std::vector<int>& out);
    void readFloatCSV(const string& filename, std::vector<float>& out);
    void validate() const;
};

#endif // UTILS_HPP
