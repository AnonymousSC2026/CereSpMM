#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int    *row_ptr;   // Mt+1
    int    *col;
    double *val;
    int     nnz;
} TileCSR;

typedef struct {
    int    *x;
    int    *y;
    double *val;
    int     nnz;
} TileCOO;

static void die(const char *msg) {
    perror(msg);
    exit(1);
}

/* ================= CSV writers ================= */

static void write_row_ptr_csv(const char *name, TileCSR *tiles,
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

static void write_col_csv(const char *name, TileCSR *tiles, int H, int W) {
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

static void write_val_csv(const char *name, double **vals,
                          int *nnz, int tiles_n) {
    FILE *f = fopen(name, "w");
    if (!f) die("open tmp_val.csv");
    for (int t = 0; t < tiles_n; t++) {
        for (int i = 0; i < nnz[t]; i++) {
            fprintf(f, "%.8e", vals[t][i]);
            if (i + 1 < nnz[t]) fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

static void write_x_csv(const char *name, TileCOO *tiles, int H, int W) {
    FILE *f = fopen(name, "w");
    if (!f) die("open tmp_x.csv");
    for (int t = 0; t < H * W; t++) {
        for (int i = 0; i < tiles[t].nnz; i++) {
            fprintf(f, "%d", tiles[t].x[i]);
            if (i + 1 < tiles[t].nnz) fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

static void write_y_csv(const char *name, TileCOO *tiles, int H, int W) {
    FILE *f = fopen(name, "w");
    if (!f) die("open tmp_y.csv");
    for (int t = 0; t < H * W; t++) {
        for (int i = 0; i < tiles[t].nnz; i++) {
            fprintf(f, "%d", tiles[t].y[i]);
            if (i + 1 < tiles[t].nnz) fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

/* ================= Main ================= */

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("usage: %s input.mtx H W type(1=CSR,2=COO)\n", argv[0]);
        return 1;
    }

    const char *fname = argv[1];
    int H = atoi(argv[2]);
    int W = atoi(argv[3]);
    int type = atoi(argv[4]);

    FILE *f = fopen(fname, "r");
    if (!f) die("open mtx");

    char header[256];
    fgets(header, sizeof(header), f);

    int is_pattern   = strstr(header, "pattern") != NULL;
    int is_symmetric = strstr(header, "symmetric") != NULL;

    char buf[512];
    do {
        if (!fgets(buf, sizeof(buf), f))
            die("read size line");
    } while (buf[0] == '%');

    int M, N, NNZ;
    sscanf(buf, "%d %d %d", &M, &N, &NNZ);
    long nnz_pos = ftell(f);

    int Mt = (M + H - 1) / H;
    int Kt = (N + W - 1) / W;

    printf("[INFO] Matrix %d x %d padded to %d x %d\n",
           M, N, Mt * H, Kt * W);
    printf("[INFO] PE grid %d x %d, block %d x %d\n",
           H, W, Mt, Kt);

    int tiles_n = H * W;
    int *nnz_cnt = calloc(tiles_n, sizeof(int));
    if (!nnz_cnt) die("alloc nnz_cnt");

    /* ---------- pass1 ---------- */
    fseek(f, nnz_pos, SEEK_SET);
    for (int k = 0; k < NNZ; k++) {
        int i, j;
        double v;
        if (is_pattern)
            fscanf(f, "%d %d", &i, &j);
        else
            fscanf(f, "%d %d %lf", &i, &j, &v);
        i--; j--;

        int py = i / Mt, px = j / Kt;
        if (py < H && px < W)
            nnz_cnt[py * W + px]++;

        if (is_symmetric && i != j) {
            py = j / Mt; px = i / Kt;
            if (py < H && px < W)
                nnz_cnt[py * W + px]++;
        }
    }

    /* ---------- CSR ---------- */
    if (type == 1) {
        int *row_cnt = calloc((size_t)tiles_n * Mt, sizeof(int));
        if (!row_cnt) die("alloc row_cnt");

        fseek(f, nnz_pos, SEEK_SET);
        for (int k = 0; k < NNZ; k++) {
            int i, j;
            double v;
            if (is_pattern)
                fscanf(f, "%d %d", &i, &j);
            else
                fscanf(f, "%d %d %lf", &i, &j, &v);
            i--; j--;

            int py = i / Mt, px = j / Kt;
            if (py < H && px < W)
                row_cnt[(py * W + px) * Mt + (i % Mt)]++;

            if (is_symmetric && i != j) {
                py = j / Mt; px = i / Kt;
                if (py < H && px < W)
                    row_cnt[(py * W + px) * Mt + (j % Mt)]++;
            }
        }

        TileCSR *tiles = calloc(tiles_n, sizeof(TileCSR));
        for (int t = 0; t < tiles_n; t++) {
            tiles[t].nnz = nnz_cnt[t];
            tiles[t].row_ptr = calloc(Mt + 1, sizeof(int));
            if (nnz_cnt[t]) {
                tiles[t].col = malloc(nnz_cnt[t] * sizeof(int));
                tiles[t].val = malloc(nnz_cnt[t] * sizeof(double));
            }
            for (int r = 0; r < Mt; r++)
                tiles[t].row_ptr[r + 1] =
                    tiles[t].row_ptr[r] + row_cnt[t * Mt + r];
        }

        int *cursor = calloc((size_t)tiles_n * Mt, sizeof(int));
        for (int t = 0; t < tiles_n; t++)
            for (int r = 0; r < Mt; r++)
                cursor[t * Mt + r] = tiles[t].row_ptr[r];

        fseek(f, nnz_pos, SEEK_SET);
        for (int k = 0; k < NNZ; k++) {
            int i, j;
            double v = 1.0;
            if (is_pattern)
                fscanf(f, "%d %d", &i, &j);
            else
                fscanf(f, "%d %d %lf", &i, &j, &v);
            i--; j--;

            int py = i / Mt, px = j / Kt;
            if (py < H && px < W) {
                int t = py * W + px;
                int r = i % Mt;
                int pos = cursor[t * Mt + r]++;
                tiles[t].col[pos] = j % Kt;
                tiles[t].val[pos] = v;
            }

            if (is_symmetric && i != j) {
                py = j / Mt; px = i / Kt;
                if (py < H && px < W) {
                    int t = py * W + px;
                    int r = j % Mt;
                    int pos = cursor[t * Mt + r]++;
                    tiles[t].col[pos] = i % Kt;
                    tiles[t].val[pos] = v;
                }
            }
        }

        write_row_ptr_csv("tmp_row_ptr.csv", tiles, H, W, Mt);
        write_col_csv("tmp_col_idx.csv", tiles, H, W);

        double **vals = malloc(tiles_n * sizeof(double *));
        for (int t = 0; t < tiles_n; t++) vals[t] = tiles[t].val;
        write_val_csv("tmp_val.csv", vals, nnz_cnt, tiles_n);

        printf("[DONE] Block-CSR generated\n");
    }

    /* ---------- COO ---------- */
    else if (type == 2) {
        TileCOO *tiles = calloc(tiles_n, sizeof(TileCOO));
        for (int t = 0; t < tiles_n; t++) {
            tiles[t].nnz = nnz_cnt[t];
            if (nnz_cnt[t]) {
                tiles[t].x   = malloc(nnz_cnt[t] * sizeof(int));
                tiles[t].y   = malloc(nnz_cnt[t] * sizeof(int));
                tiles[t].val = malloc(nnz_cnt[t] * sizeof(double));
            }
            nnz_cnt[t] = 0; // reuse as cursor
        }

        fseek(f, nnz_pos, SEEK_SET);
        for (int k = 0; k < NNZ; k++) {
            int i, j;
            double v = 1.0;
            if (is_pattern)
                fscanf(f, "%d %d", &i, &j);
            else
                fscanf(f, "%d %d %lf", &i, &j, &v);
            i--; j--;

            int py = i / Mt, px = j / Kt;
            if (py < H && px < W) {
                int t = py * W + px;
                int pos = nnz_cnt[t]++;
                tiles[t].x[pos] = i % Mt;
                tiles[t].y[pos] = j % Kt;
                tiles[t].val[pos] = v;
            }

            if (is_symmetric && i != j) {
                py = j / Mt; px = i / Kt;
                if (py < H && px < W) {
                    int t = py * W + px;
                    int pos = nnz_cnt[t]++;
                    tiles[t].x[pos] = j % Mt;
                    tiles[t].y[pos] = i % Kt;
                    tiles[t].val[pos] = v;
                }
            }
        }

        write_x_csv("tmp_x.csv", tiles, H, W);
        write_y_csv("tmp_y.csv", tiles, H, W);

        double **vals = malloc(tiles_n * sizeof(double *));
        for (int t = 0; t < tiles_n; t++) vals[t] = tiles[t].val;
        write_val_csv("tmp_val.csv", vals, nnz_cnt, tiles_n);

        printf("[DONE] Block-COO generated\n");
    }

    else {
        printf("Invalid type\n");
    }

    fclose(f);
    return 0;
}
