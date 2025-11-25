// ==== utils.cc (rewritten) ====
#include "utils.hpp"

using std::vector;

// -------------------- timing utils --------------------
double rtclock() {
    struct timezone Tzp;
    struct timeval Tp;
    gettimeofday(&Tp, &Tzp);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime) {
    printf("%s%3f seconds\n", str, endtime - starttime);
}

// -------------------- ParseData implementation --------------------
ParseData::ParseData(const string filename, std::vector<float>& adjMat)
    : _pifstream(new std::ifstream(filename), [](std::ifstream* fp){ fp->close(); })
{
    if (!_pifstream->is_open()) {
        std::cerr << "ERROR: Could not open J matrix file: " << filename << std::endl;
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
        if (line.size() == 0) continue;

        std::stringstream ss(line);
        string val;
        vector<float> row;

        while (std::getline(ss, val, ',')) {
            if (val.size() == 0) continue;
            row.push_back(std::stof(val));
        }
        if (!row.empty()) rows.push_back(row);
    }

    if (rows.empty()) {
        std::cerr << "ERROR: J matrix file is empty" << std::endl;
        exit(1);
    }

    unsigned int N = rows.size();

    // Validate square matrix
    for (unsigned int i = 0; i < N; i++) {
        if (rows[i].size() != N) {
            std::cerr << "ERROR: J matrix must be square. Row " << i
                      << " has size " << rows[i].size() << " but expected " << N << std::endl;
            exit(1);
        }
    }

    _data_dims = {N, N};

    adjMat.resize(N * N);

    // Store in row-major, and zero diagonal
    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < N; j++) {
            float v = rows[i][j];
            if (i == j) v = 0.0f; // enforce old behavior
            adjMat[i * N + j] = v;
        }
    }
}

// -------------------- Load h vector --------------------
void ParseData::readLinearValues(const string filename, std::vector<float>& linearVect)
{
    if (filename.empty()) {
        unsigned int N = _data_dims[0];
        linearVect.assign(N, 0.0f);
        return;
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "WARNING: Could not open h file. Using zeros." << std::endl;
        unsigned int N = _data_dims[0];
        linearVect.assign(N, 0.0f);
        return;
    }

    vector<float> vals;
    string line;

    while (std::getline(file, line)) {
        if (line.size() == 0) continue;
        std::stringstream ss(line);
        string val;
        while (std::getline(ss, val, ',')) {
            if (val.size() == 0) continue;
            vals.push_back(std::stof(val));
        }
    }

    unsigned int N = _data_dims[0];
    if (vals.size() < N) vals.resize(N, 0.0f);
    if (vals.size() > N) vals.resize(N);

    linearVect = vals;
}

std::vector<unsigned int> ParseData::getDataDims() const {
    return _data_dims;
}
