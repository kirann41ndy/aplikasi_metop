from pathlib import Path

import pandas as pd


# ============================================================
# KONFIGURASI FILE
# ============================================================

# Root proyek (satu tingkat di atas folder utils/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Logo transparan (disimpan di folder utils/)
LOGO_PATH = BASE_DIR / "utils" / "logo.png"

# Dataset bawaan (CSV tanpa header: kolom x, y)
DATASET_SMALL = BASE_DIR / "dataset" / "small.csv"
DATASET_SMALL_NAME = "dataset/small.csv"


def load_page_icon(fallback="🧭"):
    """
    Ikon tab browser dari logo. Logo berbentuk lebar, jadi diberi
    ruang transparan agar menjadi persegi (tidak gepeng).
    """

    try:

        from PIL import Image

        image = Image.open(LOGO_PATH).convert("RGBA")

        side = max(image.size)

        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))

        canvas.paste(
            image,
            ((side - image.width) // 2, (side - image.height) // 2),
            image
        )

        return canvas.resize((128, 128))

    except Exception:

        return fallback


# ============================================================
# LOAD DATA
# ============================================================

def load_data(file):

    if isinstance(file, pd.DataFrame):
        data = file.copy()

    elif isinstance(file, str):

        if file.lower().endswith(".csv"):
            data = pd.read_csv(file)

            columns = (
                data.columns
                .astype(str)
                .str.strip()
                .str.lower()
            )

            # Coba baca ulang jika CSV tidak memiliki header
            if not {"x", "y"}.issubset(columns):

                data = pd.read_csv(
                    file,
                    header=None
                )

                if data.shape[1] != 2:
                    raise ValueError(
                        "CSV harus memiliki tepat 2 kolom koordinat: x dan y."
                    )

                data.columns = ["x", "y"]

        elif file.lower().endswith((".xlsx", ".xls")):
            data = pd.read_excel(file)

        else:
            raise ValueError(
                "File harus berformat CSV atau Excel."
            )

    elif hasattr(file, "read"):

        filename = getattr(file, "name", "").lower()

        if filename.endswith(".csv"):

            data = pd.read_csv(file)

            columns = (
                data.columns
                .astype(str)
                .str.strip()
                .str.lower()
            )

            if not {"x", "y"}.issubset(columns):

                file.seek(0)

                data = pd.read_csv(
                    file,
                    header=None
                )

                if data.shape[1] != 2:
                    raise ValueError(
                        "CSV harus memiliki tepat 2 kolom koordinat: x dan y."
                    )

                data.columns = ["x", "y"]

        elif filename.endswith((".xlsx", ".xls")):
            data = pd.read_excel(file)

        else:
            raise ValueError(
                "File upload harus berformat CSV atau Excel."
            )

    else:
        raise TypeError(
            "Input harus berupa path file, DataFrame, atau uploaded file."
        )

    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    required_columns = {"x", "y"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "Data harus memiliki kolom x dan y."
        )

    # Buat node otomatis jika tidak tersedia
    if "node" not in data.columns:
        data.insert(
            0,
            "node",
            range(len(data))
        )

    data = data[["node", "x", "y"]].copy()

    if data.isnull().any().any():
        raise ValueError(
            "Data tidak boleh memiliki missing value."
        )

    data["x"] = pd.to_numeric(
        data["x"],
        errors="raise"
    )

    data["y"] = pd.to_numeric(
        data["y"],
        errors="raise"
    )

    if data["node"].duplicated().any():
        raise ValueError(
            "Node tidak boleh duplikat."
        )

    if len(data) < 3:
        raise ValueError(
            "Jumlah node minimal adalah 3."
        )

    data = data.reset_index(drop=True)

    return data
