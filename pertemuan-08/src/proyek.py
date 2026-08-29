"""Mini Project Deep Learning — kerangka kerja.

Struktur minimal yang harus kamu isi. Boleh menambah fungsi lain,
tetapi keempat fungsi di bawah wajib ada (dites secara struktural).
"""
import os
import time

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

RANDOM_STATE = 42
JUMLAH_LATIH = 12000
JUMLAH_UJI = 3000
MLP_EPOCHS = 4
CNN_EPOCHS = 8
BATCH_SIZE = 128
NAMA_KELAS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def muat_data():
    """Muat dan kembalikan data mentah proyekmu."""
    # TODO: implementasikan sesuai proposalmu
    from tensorflow.keras.datasets import fashion_mnist

    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "nama_kelas": NAMA_KELAS,
    }
    # raise NotImplementedError


def _ambil_berimbang(X, y, jumlah: int):
    """Ambil subset berimbang agar eksperimen tetap cepat dan reproducible."""
    if jumlah >= len(X):
        return X, y

    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=jumlah,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_sub, y_sub


def siapkan_data(data):
    """Praproses + split. Return (X_train, X_test, y_train, y_test)."""
    # TODO: implementasikan
    X_train, y_train = _ambil_berimbang(data["X_train"], data["y_train"], JUMLAH_LATIH)
    X_test, y_test = _ambil_berimbang(data["X_test"], data["y_test"], JUMLAH_UJI)

    X_train = (X_train.astype("float32") / 255.0)[..., np.newaxis]
    X_test = (X_test.astype("float32") / 255.0)[..., np.newaxis]
    return X_train, X_test, y_train.astype(int), y_test.astype(int)
    # raise NotImplementedError


def latih_baseline(X_train, y_train):
    """Latih model baseline sederhana. Return model."""
    # TODO: implementasikan
    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=1000,
        tol=1e-3,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    model.nama_model = "Baseline SGD Logistic"
    model.input_mode = "flat"
    return model
    # raise NotImplementedError


def _buat_mlp():
    """Bangun MLP Keras untuk input pixel flattened."""
    from tensorflow import keras

    keras.utils.set_random_seed(RANDOM_STATE)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(28 * 28,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.nama_model = "MLP"
    model.input_mode = "flat"
    return model


def latih_mlp(X_train, y_train):
    """Latih MLP sebagai model pembanding neural network."""
    model = _buat_mlp()
    mulai = time.perf_counter()
    history = model.fit(
        X_train.reshape(len(X_train), -1),
        y_train,
        epochs=MLP_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=0,
    )
    model.waktu_latih = time.perf_counter() - mulai
    model.riwayat = history.history
    return model


def _buat_cnn():
    """Bangun CNN sederhana untuk citra Fashion-MNIST 28x28."""
    from tensorflow import keras

    keras.utils.set_random_seed(RANDOM_STATE)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(64, 3, activation="relu"),
            keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.nama_model = "CNN"
    model.input_mode = "image"
    return model


def latih_model(X_train, y_train):
    """Latih model utama (MLP/CNN/RNN/dll). Return model."""
    # TODO: implementasikan
    model = _buat_cnn()
    mulai = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        epochs=CNN_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=0,
    )
    model.waktu_latih = time.perf_counter() - mulai
    model.riwayat = history.history
    return model
    # raise NotImplementedError


def _prediksi(model, X):
    """Prediksi label untuk model sklearn maupun Keras."""
    input_mode = getattr(model, "input_mode", "image")
    X_model = X.reshape(len(X), -1) if input_mode == "flat" else X
    try:
        pred = model.predict(X_model, verbose=0)
    except TypeError:
        pred = model.predict(X_model)
    pred = np.asarray(pred)
    if pred.ndim == 2:
        return pred.argmax(axis=1)
    return pred.astype(int)


def prediksi_label(model, X) -> np.ndarray:
    """Return label prediksi dalam format integer."""
    return _prediksi(model, X)


def evaluasi(model, X_test, y_test) -> dict:
    """Return dict metrik, misal {"akurasi": 0.93} atau {"mse": 0.01}."""
    # TODO: implementasikan
    mulai = time.perf_counter()
    y_pred = _prediksi(model, X_test)
    waktu_prediksi = time.perf_counter() - mulai
    return {
        "akurasi": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "waktu_latih": float(getattr(model, "waktu_latih", 0.0)),
        "waktu_prediksi": float(waktu_prediksi),
    }
    # raise NotImplementedError


def contoh_salah(model, X_test, y_test, jumlah: int = 6) -> list:
    """Return contoh prediksi salah dalam format (index, label_asli, prediksi)."""
    y_pred = _prediksi(model, X_test)
    hasil = []
    for i, (asli, pred) in enumerate(zip(y_test, y_pred)):
        if asli != pred:
            hasil.append(
                {
                    "index": int(i),
                    "label_asli": int(asli),
                    "nama_asli": NAMA_KELAS[int(asli)],
                    "prediksi": int(pred),
                    "nama_prediksi": NAMA_KELAS[int(pred)],
                }
            )
        if len(hasil) == jumlah:
            break
    return hasil


def kelas_tertukar(confusion: list, jumlah: int = 5) -> list:
    """Pasangan kelas yang paling sering tertukar berdasarkan confusion matrix."""
    cm = np.array(confusion)
    pasangan = []
    for asli in range(cm.shape[0]):
        for pred in range(cm.shape[1]):
            if asli != pred and cm[asli, pred] > 0:
                pasangan.append((int(cm[asli, pred]), NAMA_KELAS[asli], NAMA_KELAS[pred]))
    pasangan.sort(reverse=True)
    return pasangan[:jumlah]


def jalankan_eksperimen():
    """Latih baseline, MLP, dan CNN lalu kembalikan model serta metriknya."""
    data = muat_data()
    X_train, X_test, y_train, y_test = siapkan_data(data)

    mulai = time.perf_counter()
    baseline = latih_baseline(X_train, y_train)
    baseline.waktu_latih = time.perf_counter() - mulai
    mlp = latih_mlp(X_train, y_train)
    cnn = latih_model(X_train, y_train)

    models = {
        "Baseline SGD Logistic": baseline,
        "MLP": mlp,
        "CNN": cnn,
    }
    hasil = {nama: evaluasi(model, X_test, y_test) for nama, model in models.items()}
    return data, (X_train, X_test, y_train, y_test), models, hasil


def main():
    """Jalankan eksperimen dan cetak ringkasan evaluasi."""
    _, data_split, models, hasil = jalankan_eksperimen()
    _, X_test, _, y_test = data_split

    print("Evaluasi Fashion-MNIST")
    print(f"Data uji: {len(X_test)} citra | MLP epoch: {MLP_EPOCHS} | CNN epoch: {CNN_EPOCHS}")
    print(f"{'Model':24s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Train(s)':>9s}")
    print("-" * 70)
    for nama, metrik in hasil.items():
        print(
            f"{nama:24s} "
            f"{metrik['akurasi']:7.4f} "
            f"{metrik['precision_macro']:7.4f} "
            f"{metrik['recall_macro']:7.4f} "
            f"{metrik['f1_macro']:7.4f} "
            f"{metrik['waktu_latih']:9.2f}"
        )

    nama_terbaik = max(hasil, key=lambda nama: hasil[nama]["f1_macro"])
    print(f"\nModel terbaik berdasarkan F1 macro: {nama_terbaik} ({hasil[nama_terbaik]['f1_macro']:.4f})")
    print("Kelas paling sering tertukar:", kelas_tertukar(hasil[nama_terbaik]["confusion_matrix"], jumlah=5))
    print("Contoh salah model terbaik:", contoh_salah(models[nama_terbaik], X_test, y_test, jumlah=4))


if __name__ == "__main__":
    main()
