#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EPS 1e-12

typedef struct {
    int    *row_ptr;   // size Mt+1
    int    *col;       // size nnz
    double *val;       // size nnz
    int     nnz;
} TileCSR;

static void die(const char *msg) {
    perror(msg);
    exit(1);
}

/* ===================== CSV writers ===================== */

static void write_row_ptr_csv(const char *name,
                              TileCSR *tiles,
                              int H, int W, int Mt) {
    FILE *f = fopen(name, "w");
    if (!f) die("open tmp_row_ptr.csv");

    for (int t = 0; t < H * W; t++) {
        for (int i = 0; i < Mt + 1; i++) {
            fprintf(f, "%d", tiles[t].row_ptr[i]);
            if (i + 1 < Mt + 1) fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

static void write_col_csv(const char *name,
                          TileCSR *tiles,
                          int H, int W) {
    FILE *f = fopen(name, "w");
    if (!f) die("open tmp_col_idx.csv");

    for (int t = 0; t < H * W; t++) {
        for (int i = 0; i < tiles[t].nnz; i++) {
            fprintf(f, "%d", tiles[t].col[i]);
            if (i + 1 < tiles[t].nnz) fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

static void write_val_csv(const char *name,
                          TileCSR *tiles,
                          int H, int W) {
    FILE *f = fopen(name, "w");
    if (!f) die("open tmp_val.csv");

    for (int t = 0; t < H * W; t++) {
        for (int i = 0; i < tiles[t].nnz; i++) {
            fprintf(f, "%.8e", tiles[t].val[i]);
            if (i + 1 < tiles[t].nnz) fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

/* ===================== Main ===================== */

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("usage: %s input.mtx H W\n", argv[0]);
        return 1;
    }

    const char *fname = argv[1];
    int H = atoi(argv[2]);   // PE grid rows
    int W = atoi(argv[3]);   // PE grid cols

    FILE *f = fopen(fname, "r");
    if (!f) die("open mtx");

    /* ---------- parse MatrixMarket header ---------- */
    char header[256];
    if (!fgets(header, sizeof(header), f))
        die("read header");

    int is_pattern   = (strstr(header, "pattern") != NULL);
    int is_symmetric = (strstr(header, "symmetric") != NULL);

    /* ---------- skip comments ---------- */
    char buf[512];
    do {
        if (!fgets(buf, sizeof(buf), f))
            die("read size line");
    } while (buf[0] == '%');

    int M, N, NNZ;
    sscanf(buf, "%d %d %d", &M, &N, &NNZ);

    long nnz_pos = ftell(f);   // ⭐ nnz start

    int Mt = (M + H - 1) / H;
    int Kt = (N + W - 1) / W;

    printf("[INFO] Matrix %d x %d padded to %d x %d\n",
           M, N, Mt * H, Kt * W);
    printf("[INFO] PE grid %d x %d, block size %d x %d\n",
           H, W, Mt, Kt);
    printf("[INFO] NNZ in file    : %d%s\n", NNZ, is_symmetric ? " (lower/upper triangle only)" : "");

    int tiles_n = H * W;

    /* ---------- pass 1: count ---------- */
    int *nnz_cnt = calloc(tiles_n, sizeof(int));
    int *row_cnt = calloc((size_t)tiles_n * Mt, sizeof(int));
    if (!nnz_cnt || !row_cnt) die("alloc pass1");

    fseek(f, nnz_pos, SEEK_SET);

    for (int k = 0; k < NNZ; k++) {
        int i, j;
        double v = 1.0;
        int ret;

        if (is_pattern) {
            ret = fscanf(f, "%d %d", &i, &j);
        } else {
            ret = fscanf(f, "%d %d %lf", &i, &j, &v);
        }
        if (ret < 2) die("read nnz pass1");

        i--; j--;

        int py = i / Mt;
        int px = j / Kt;
        if (py >= 0 && py < H && px >= 0 && px < W) {
            int tid = py * W + px;
            nnz_cnt[tid]++;
            row_cnt[tid * Mt + (i % Mt)]++;
        }

        /* expand symmetric */
        if (is_symmetric && i != j) {
            py = j / Mt;
            px = i / Kt;
            if (py >= 0 && py < H && px >= 0 && px < W) {
                int tid = py * W + px;
                nnz_cnt[tid]++;
                row_cnt[tid * Mt + (j % Mt)]++;
            }
        }
    }
    long real_nnz = 0;
    for (int t = 0; t < tiles_n; t++) {
        real_nnz += nnz_cnt[t];
    }
    printf("[INFO] Expanded NNZ (actual stored nnz): %ld\n", real_nnz);

    /* ---------- allocate tiles ---------- */
    TileCSR *tiles = calloc(tiles_n, sizeof(TileCSR));
    if (!tiles) die("alloc tiles");

    for (int t = 0; t < tiles_n; t++) {
        tiles[t].nnz = nnz_cnt[t];
        tiles[t].row_ptr = calloc(Mt + 1, sizeof(int));
        if (!tiles[t].row_ptr) die("alloc row_ptr");

        if (nnz_cnt[t] > 0) {
            tiles[t].col = malloc(sizeof(int) * nnz_cnt[t]);
            tiles[t].val = malloc(sizeof(double) * nnz_cnt[t]);
            if (!tiles[t].col || !tiles[t].val)
                die("alloc col/val");
        }

        for (int r = 0; r < Mt; r++)
            tiles[t].row_ptr[r + 1] = row_cnt[t * Mt + r];

        for (int r = 0; r < Mt; r++)
            tiles[t].row_ptr[r + 1] += tiles[t].row_ptr[r];
    }

    int *cursor = calloc((size_t)tiles_n * Mt, sizeof(int));
    if (!cursor) die("alloc cursor");

    for (int t = 0; t < tiles_n; t++)
        for (int r = 0; r < Mt; r++)
            cursor[t * Mt + r] = tiles[t].row_ptr[r];

    /* ---------- pass 2: fill ---------- */
    fseek(f, nnz_pos, SEEK_SET);

    for (int k = 0; k < NNZ; k++) {
        int i, j;
        double v = 1.0;
        int ret;

        if (is_pattern) {
            ret = fscanf(f, "%d %d", &i, &j);
        } else {
            ret = fscanf(f, "%d %d %lf", &i, &j, &v);
        }
        if (ret < 2) die("read nnz pass2");

        i--; j--;

        int py = i / Mt;
        int px = j / Kt;
        if (py >= 0 && py < H && px >= 0 && px < W) {
            int tid = py * W + px;
            int r = i % Mt;
            int pos = cursor[tid * Mt + r]++;
            tiles[tid].col[pos] = j % Kt;
            tiles[tid].val[pos] = v;
        }

        if (is_symmetric && i != j) {
            py = j / Mt;
            px = i / Kt;
            if (py >= 0 && py < H && px >= 0 && px < W) {
                int tid = py * W + px;
                int r = j % Mt;
                int pos = cursor[tid * Mt + r]++;
                tiles[tid].col[pos] = i % Kt;
                tiles[tid].val[pos] = v;
            }
        }
    }

    fclose(f);

    /* ---------- write CSV ---------- */
    write_row_ptr_csv("tmp_row_ptr.csv", tiles, H, W, Mt);
    write_col_csv    ("tmp_col_idx.csv", tiles, H, W);
    write_val_csv    ("tmp_val.csv",     tiles, H, W);

    printf("[DONE] BCSR generated (supports all common mtx kinds)\n");
    return 0;
}

