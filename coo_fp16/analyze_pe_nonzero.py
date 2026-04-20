import csv

# 修改为你的真实路径
val_path = "tmp_val_pad.csv"
x_path = "tmp_x_pad.csv"
y_path = "tmp_y_pad.csv"

def count_active_pes(val_file, x_file, y_file):
    total_pe = 0
    active_pe = 0
    active_indices = []

    with open(val_file, newline='') as vf, open(x_file, newline='') as xf, open(y_file, newline='') as yf:
        val_reader = csv.reader(vf)
        x_reader = csv.reader(xf)
        y_reader = csv.reader(yf)

        for idx, (val_row, x_row, y_row) in enumerate(zip(val_reader, x_reader, y_reader)):
            total_pe += 1
            # 检查是否有非零项
            has_nonzero = any(float(v.strip()) != 0 for v in val_row)

            if has_nonzero:
                active_pe += 1
                active_indices.append(idx)

    print(f"Total PE: {total_pe}")
    print(f"Active PE (with non-zero values): {active_pe}")
    print(f"Percentage: {active_pe / total_pe * 100:.2f}%")
    #print(f"Active PE indices (flattened): {active_indices}")

if __name__ == "__main__":
    count_active_pes(val_path, x_path, y_path)
