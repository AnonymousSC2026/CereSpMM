#include <bits/stdc++.h>
using namespace std;

/*
 * Optimized Pure Block-DIA Generator
 *
 * Usage:
 *   ./mm_to_dia <matrix.mtx> <Mt> <Kt> <MAX_NUM_DIAGS> <MAX_VALS_PER_PX>
 *
 * Outputs:
 *   A_diag_offset.csv
 *   A_row_start.csv
 *   A_len.csv
 *   A_diag_ptr.csv
 *   A_vals.csv
 *   num_diags.csv
 *   vals_len.csv
 *   has_work.csv
 *
 * NOTE:
 *   - Only valid local elements are inserted into each tile.
 *   - No row ownership / kl legality checks are needed in kernel.
 */

static const int h = 1170;
static const int w = 755;

struct COO {
    int M = 0, K = 0;
    vector<int> rows;
    vector<int> cols;
    vector<float> vals;
};

COO read_mm(const string &path) {
    ifstream fin(path);
    if (!fin) {
        cerr << "ERROR: cannot open " << path << "\n";
        exit(1);
    }

    COO A;
    string line;

    while (getline(fin, line)) {
        if (!line.empty() && line[0] != '%')
            break;
    }

    {
        stringstream ss(line);
        int nnz;
        ss >> A.M >> A.K >> nnz;
        A.rows.reserve(nnz);
        A.cols.reserve(nnz);
        A.vals.reserve(nnz);
    }

    int r, c;
    double v;
    while (fin >> r >> c >> v) {
        A.rows.push_back(r - 1);
        A.cols.push_back(c - 1);
        A.vals.push_back((float)v);
    }

    return A;
}

int main(int argc, char **argv) {

    if (argc != 6) {
        cerr << "Usage: ./mm_to_dia <matrix.mtx> <Mt> <Kt> <MAX_NUM_DIAGS> <MAX_VALS_PER_PX>\n";
        return 1;
    }

    string mm_path = argv[1];
    int Mt = stoi(argv[2]);
    int Kt = stoi(argv[3]);
    int MAX_NUM_DIAGS = stoi(argv[4]);
    int MAX_VALS_PER_PX = stoi(argv[5]);

    COO A = read_mm(mm_path);

    int T = h * w;

    struct Tile {
        unordered_map<int, vector<pair<int,float>>> diag_map;
    };

    vector<Tile> tiles(T);

    // ------------------------------------------------------------
    // Strict scatter (FULL pre-filter)
    // ------------------------------------------------------------
    for (size_t i = 0; i < A.rows.size(); i++) {

        int r = A.rows[i];
        int c = A.cols[i];
        float v = A.vals[i];

        int py = r / Mt;
        int px = c / Kt;

        if (py >= h || px >= w) continue;

        // Strict local bounds check
        if (r < py * Mt || r >= (py + 1) * Mt) continue;
        if (c < px * Kt || c >= (px + 1) * Kt) continue;

        int d = c - r;
        int tid = py * w + px;

        tiles[tid].diag_map[d].push_back({r, v});
    }

    // ------------------------------------------------------------
    // Allocate buffers
    // ------------------------------------------------------------
    vector<int32_t>  A_diag_offset(T * MAX_NUM_DIAGS, 0);
    vector<uint32_t> A_row_start  (T * MAX_NUM_DIAGS, 0);
    vector<uint32_t> A_len        (T * MAX_NUM_DIAGS, 0);
    vector<uint32_t> A_diag_ptr   (T * (MAX_NUM_DIAGS + 1), 0);
    vector<float>    A_vals       (T * MAX_VALS_PER_PX, 0.0f);
    vector<uint32_t> num_diags(T, 0);
    vector<uint32_t> vals_len (T, 0);
    vector<uint32_t> has_work (T, 0);

    // ------------------------------------------------------------
    // Build segmented Block-DIA
    // ------------------------------------------------------------
    for (int tid = 0; tid < T; tid++) {

        auto &diag_map = tiles[tid].diag_map;

        vector<int> ds;
        for (auto &kv : diag_map) ds.push_back(kv.first);
        sort(ds.begin(), ds.end());

        int diag_cnt = 0;
        int val_cnt  = 0;
        int ptr      = 0;

        for (int d : ds) {

            if (diag_cnt >= MAX_NUM_DIAGS) break;

            auto &vec = diag_map[d];
            if (vec.empty()) continue;

            sort(vec.begin(), vec.end(),
                 [](auto &a, auto &b){ return a.first < b.first; });

            int seg_start = vec[0].first;
            vector<float> seg_vals = { vec[0].second };
            int last_row = seg_start;

            for (size_t i = 1; i < vec.size(); i++) {

                int r = vec[i].first;
                float v = vec[i].second;

                if (r == last_row + 1) {
                    seg_vals.push_back(v);
                } else {

                    if (val_cnt + seg_vals.size() > MAX_VALS_PER_PX)
                        break;

                    int base = tid * MAX_NUM_DIAGS + diag_cnt;

                    A_diag_offset[base] = d;
                    A_row_start  [base] = seg_start;
                    A_len        [base] = seg_vals.size();

                    memcpy(&A_vals[tid * MAX_VALS_PER_PX + val_cnt],
                           seg_vals.data(),
                           seg_vals.size() * sizeof(float));

                    ptr += seg_vals.size();
                    A_diag_ptr[tid * (MAX_NUM_DIAGS + 1) + diag_cnt + 1] = ptr;

                    val_cnt += seg_vals.size();
                    diag_cnt++;

                    seg_start = r;
                    seg_vals = {v};
                }

                last_row = r;
            }

            if (!seg_vals.empty() && diag_cnt < MAX_NUM_DIAGS) {

                if (val_cnt + seg_vals.size() <= MAX_VALS_PER_PX) {

                    int base = tid * MAX_NUM_DIAGS + diag_cnt;

                    A_diag_offset[base] = d;
                    A_row_start  [base] = seg_start;
                    A_len        [base] = seg_vals.size();

                    memcpy(&A_vals[tid * MAX_VALS_PER_PX + val_cnt],
                           seg_vals.data(),
                           seg_vals.size() * sizeof(float));

                    ptr += seg_vals.size();
                    A_diag_ptr[tid * (MAX_NUM_DIAGS + 1) + diag_cnt + 1] = ptr;

                    val_cnt += seg_vals.size();
                    diag_cnt++;
                }
            }
        }

        num_diags[tid] = diag_cnt;
        vals_len [tid] = val_cnt;
        has_work [tid] = (diag_cnt > 0);
    }

    // ------------------------------------------------------------
    // Dump CSV
    // ------------------------------------------------------------
    auto dump = [](const string &name, auto &buf, int cols) {
        ofstream fout(name);
        int rows = buf.size() / cols;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fout << buf[i * cols + j];
                if (j + 1 < cols) fout << ",";
            }
            fout << "\n";
        }
    };

    dump("A_diag_offset.csv", A_diag_offset, MAX_NUM_DIAGS);
    dump("A_row_start.csv",   A_row_start,   MAX_NUM_DIAGS);
    dump("A_len.csv",         A_len,         MAX_NUM_DIAGS);
    dump("A_diag_ptr.csv",    A_diag_ptr,    MAX_NUM_DIAGS + 1);
    dump("A_vals.csv",        A_vals,        MAX_VALS_PER_PX);

    {
        ofstream f("num_diags.csv");
        for (auto v : num_diags) f << v << "\n";
    }
    {
        ofstream f("vals_len.csv");
        for (auto v : vals_len) f << v << "\n";
    }
    {
        ofstream f("has_work.csv");
        for (auto v : has_work) f << v << "\n";
    }

    cerr << "[DONE] Optimized Pure Block-DIA generated\n";
    return 0;
}
