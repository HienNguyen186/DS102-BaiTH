import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

from svm import SVM
from svm_sklearn import SVMSklearn

# Tự động cd vào folder chứa main.py khi chạy từ VS Code
os.chdir(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = "data/chest_xray"

def process_image(img_path):
    """Đọc ảnh, chuyển sang grayscale và resize."""
    # Đọc ảnh trực tiếp ở dạng grayscale để tránh lỗi kênh màu
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Resize về 128x128
    img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR_EXACT)
    # Trả về mảng đã flatten và chuẩn hóa
    return img.reshape(-1) / 255.0

def collect_data(split: str = "train"):
    images = []
    labels = []

    categories = {
        "NORMAL": -1,
        "PNEUMONIA": 1,
        "VIRAL": 1  # Nếu thư mục này không tồn tại, hàm sẽ tự bỏ qua
    }

    if split == "train":
        for folder, label in categories.items():
            folder_path = os.path.join(BASE_DIR, split, folder)
            
            # Kiểm tra nếu thư mục tồn tại mới duyệt
            if not os.path.exists(folder_path):
                print(f"Cảnh báo: Không tìm thấy thư mục {folder_path}, bỏ qua...")
                continue
                
            for img_file in os.listdir(folder_path):
                if img_file == ".DS_Store": continue
                
                img_path = os.path.join(folder_path, img_file)
                processed_img = process_image(img_path)
                
                if processed_img is not None:
                    images.append(processed_img)
                    labels.append(label)

    elif split == "test":
        # Duyệt qua các thư mục con trong 'test' thay vì liệt kê trực tiếp thư mục 'test'
        for folder in ["NORMAL", "PNEUMONIA"]:
            folder_path = os.path.join(BASE_DIR, split, folder)
            
            if not os.path.exists(folder_path):
                continue
            
            print(f"Đang đọc ảnh từ: {folder_path}")
            for img_file in os.listdir(folder_path):
                if img_file == ".DS_Store": continue
                
                img_path = os.path.join(folder_path, img_file)
                processed_img = process_image(img_path)
                
                if processed_img is not None:
                    images.append(processed_img)
                    # Gán nhãn: NORMAL là -1, PNEUMONIA là 1
                    labels.append(-1 if folder == "NORMAL" else 1)

    else:
        raise ValueError(f"split phải là 'train' hoặc 'test', nhận được: '{split}'")

    X = np.stack(images, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int32)

    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def plot_loss_curve(losses, n_iterations, n_train):
    losses = np.array(losses)
    epoch_losses = [losses[e * n_train:(e + 1) * n_train].mean()
                    for e in range(n_iterations)]

    plt.figure(figsize=(10, 5))
    plt.plot(range(n_iterations), epoch_losses, linewidth=2, color="steelblue")
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Average Loss", fontsize=13)
    plt.title("Training Loss Curve - Soft-margin SVM (SGD)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=100)
    plt.show()
    print(f"  Final loss: {epoch_losses[-1]:.4f}")


def plot_comparison(results):
    models = list(results.keys())
    metrics_names = ["Precision", "Recall", "F1"]
    x = np.arange(len(models))
    width = 0.25
    colors = ["steelblue", "darkorange", "seagreen"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (metric, color) in enumerate(zip(metrics_names, colors)):
        values = [results[m][metric] for m in models]
        bars = ax.bar(x + i * width, values, width, label=metric,
                      color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("So sánh kết quả các mô hình SVM", fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=100)
    plt.show()


def print_results(results, title="Kết quả trên tập Test"):
    print(f"\n{'=' * 52}")
    print(f"  {title}")
    print(f"{'=' * 52}")
    print(f"{'Model':<22} {'Precision':>9} {'Recall':>9} {'F1':>9}")
    print(f"{'=' * 52}")
    for name, m in results.items():
        print(f"{name:<22} {m['Precision']:>9.4f} {m['Recall']:>9.4f} {m['F1']:>9.4f}")
    print(f"{'=' * 52}\n")


if __name__ == "__main__":
    np.random.seed(42)

    # Load data
    print("NẠP DỮ LIỆU:")

    import os
    check_path = os.path.join("data", "chest_xray", "train")
    print(f"Kiểm tra tại: {os.path.abspath(check_path)}")
    print("Danh sách các thư mục con hiện có:", os.listdir(check_path))

    X_train, y_train = collect_data("train")
    X_test, y_test = collect_data("test")
    print(f"  Train set: {X_train.shape} | Test set: {X_test.shape}")

    # Danh sách lưu kết quả để so sánh
    train_results = {}
    test_results = {}

    # 1. Assignment 1: SVM NumPy + SGD
    print("\n" + "=" * 60)
    print(f"{'ASSIGNMENT 1: SVM NUMPY + SGD':^60}")
    print("=" * 60)
    svm_numpy = SVM(C=1, lr=1e-5, n_iterations=100)
    svm_numpy.fit(X_train, y_train)
    
    # Đánh giá trên cả 2 tập để kiểm tra Overfitting
    train_results["SVM NumPy (SGD)"] = svm_numpy.get_metrics(X_train, y_train)
    test_results["SVM NumPy (SGD)"] = svm_numpy.get_metrics(X_test, y_test)

    # 2. Assignment 2: SVM sklearn (SMO)
    print("\n" + "=" * 60)
    print(f"{'ASSIGNMENT 2: SVM SKLEARN (SMO)':^60}")
    print("=" * 60)
    
    svm_linear = SVMSklearn(kernel="linear", C=1.0)
    svm_linear.fit(X_train, y_train)
    train_results["sklearn (Linear)"] = svm_linear.get_metrics(X_train, y_train)
    test_results["sklearn (Linear)"] = svm_linear.get_metrics(X_test, y_test)

    print("\n" + " " * 15 + "BẢNG KIỂM TRA OVERFITTING (F1-SCORE)")
    print("-" * 60)
    print(f"{'Model':<22} | {'Train F1':>15} | {'Test F1':>15}")
    print("-" * 60)
    for name in train_results.keys():
        f1_train = train_results[name]['F1']
        f1_test = test_results[name]['F1']
        print(f"{name:<22} | {f1_train:>15.4f} | {f1_test:>15.4f}")
    print("-" * 60)

    print_results(test_results, title="Kết quả chi tiết trên tập Test")
    plot_comparison(test_results)
    plot_loss_curve(svm_numpy.losses, svm_numpy.n_iterations, len(X_train))
