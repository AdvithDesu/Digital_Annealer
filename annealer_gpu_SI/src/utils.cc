// ==== utils.cc ====
#include "utils.hpp"

using std::vector;

// -------------------- timing utils --------------------
double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    gettimeofday(&Tp, &Tzp);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime)
{
    printf("%s%3f seconds\n", str, endtime - starttime);
}

void readLinearValues(const std::string& filename,
                      unsigned int num_spins,
                      std::vector<float>& linearVect)
{
    if (filename.empty()) {
        linearVect.assign(num_spins, 0.0f);
        return;
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "WARNING: Could not open h file. Using zeros.\n";
        linearVect.assign(num_spins, 0.0f);
        return;
    }

    std::vector<float> vals;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string val;
        while (std::getline(ss, val, ',')) {
            if (!val.empty())
                vals.push_back(std::stof(val));
        }
    }

    if (vals.size() < num_spins) vals.resize(num_spins, 0.0f);
    if (vals.size() > num_spins) vals.resize(num_spins);

    linearVect = vals;
}

// =====================================================
// Legacy dense J parser (UNCHANGED)
// =====================================================
ParseData::ParseData(const string filename, std::vector<float>& adjMat)
    : _pifstream(new std::ifstream(filename),
                 [](std::ifstream* fp){ fp->close(); })
{
    if (!_pifstream->is_open()) {
        std::cerr << "ERROR: Could not open J matrix file: "
                  << filename << std::endl;
        exit(1);
    }

    readDenseJ(adjMat);
    _pifstream->close();
}

void ParseData::readDenseJ(std::vector<float>& adjMat)
{
    vector<vector<float>> rows;
    string line;

    while (std::getline(*_pifstream, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        string val;
        vector<float> row;

        while (std::getline(ss, val, ',')) {
            if (val.empty()) continue;
            row.push_back(std::stof(val));
        }
        if (!row.empty()) rows.push_back(row);
    }

    if (rows.empty()) {
        std::cerr << "ERROR: J matrix file is empty" << std::endl;
        exit(1);
    }

    unsigned int num_spins = rows.size();

    for (unsigned int i = 0; i < num_spins; i++) {
        if (rows[i].size() != num_spins) {
            std::cerr << "ERROR: J matrix must be square" << std::endl;
            exit(1);
        }
    }

    _data_dims = { num_spins, num_spins };
    adjMat.resize(num_spins * num_spins);

    for (unsigned int i = 0; i < num_spins; i++) {
        for (unsigned int j = 0; j < num_spins; j++) {
            float v = rows[i][j];
            if (i == j) v = 0.0f;
            adjMat[i * num_spins + j] = v;
        }
    }
}

void ParseData::readLinearValues(const string filename,
                                 std::vector<float>& linearVect)
{
    if (filename.empty()) {
        unsigned int num_spins = _data_dims[0];
        linearVect.assign(num_spins, 0.0f);
        return;
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "WARNING: Could not open h file. Using zeros." << std::endl;
        unsigned int num_spins = _data_dims[0];
        linearVect.assign(num_spins, 0.0f);
        return;
    }

    vector<float> vals;
    string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        string val;
        while (std::getline(ss, val, ',')) {
            if (val.empty()) continue;
            vals.push_back(std::stof(val));
        }
    }

    unsigned int num_spins = _data_dims[0];
    if (vals.size() < num_spins) vals.resize(num_spins, 0.0f);
    if (vals.size() > num_spins) vals.resize(num_spins);

    linearVect = vals;
}

std::vector<unsigned int> ParseData::getDataDims() const
{
    return _data_dims;
}

// =====================================================
// Sparse J parser implementation
// =====================================================
ParseSparseData::ParseSparseData(const string& row_ptr_file,
                                 const string& col_idx_file,
                                 const string& values_file)
{
    readIntCSV(row_ptr_file, row_ptr);
    readIntCSV(col_idx_file, col_idx);
    readFloatCSV(values_file, values);

    validate();
}

void ParseSparseData::readIntCSV(const string& filename,
                                 std::vector<int>& out)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open file: "
                  << filename << std::endl;
        exit(1);
    }

    string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        string val;
        while (std::getline(ss, val, ',')) {
            if (val.empty()) continue;
            out.push_back(std::stoi(val));
        }
    }
}

void ParseSparseData::readFloatCSV(const string& filename,
                                   std::vector<float>& out)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open file: "
                  << filename << std::endl;
        exit(1);
    }

    string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        string val;
        while (std::getline(ss, val, ',')) {
            if (val.empty()) continue;
            out.push_back(std::stof(val));
        }
    }
}

void ParseSparseData::validate() const
{
    if (row_ptr.size() < 2) {
        std::cerr << "ERROR: row_ptr must have at least 2 entries" << std::endl;
        exit(1);
    }

    unsigned int inferred_num_spins = row_ptr.size() - 1;
    unsigned int inferred_nnz = values.size();

    if (col_idx.size() != inferred_nnz) {
        std::cerr << "ERROR: col_idx and values size mismatch" << std::endl;
        exit(1);
    }

    if ((unsigned int)row_ptr.back() != inferred_nnz) {
        std::cerr << "ERROR: row_ptr last entry must equal nnz" << std::endl;
        exit(1);
    }

    for (unsigned int i = 0; i < inferred_nnz; i++) {
        if (col_idx[i] < 0 || col_idx[i] >= (int)inferred_num_spins) {
            std::cerr << "ERROR: col_idx out of bounds at index "
                      << i << std::endl;
            exit(1);
        }
    }

    // Store validated sizes
    const_cast<ParseSparseData*>(this)->num_spins = inferred_num_spins;
    const_cast<ParseSparseData*>(this)->nnz = inferred_nnz;
}
