"""下载、解析并切分 Financial PhraseBank。"""
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split



ZIP_URL = ("https://huggingface.co/datasets/takala/financial_phrasebank/"
           "resolve/main/data/FinancialPhraseBank-v1.0.zip")
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
SUBSET_FILES = {
    "allagree": "Sentences_AllAgree.txt",
    "75agree": "Sentences_75Agree.txt",
    "66agree": "Sentences_66Agree.txt",
    "50agree": "Sentences_50Agree.txt",
}

PROJECT_ROOT  = Path(__file__).resolve().parents[2]   # src/finsent/data.py → 上三层 = 项目根
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GOLD_DIR      = PROJECT_ROOT / "data" / "gold"        # Phase 6 手标金标准集
SILVER_DIR    = PROJECT_ROOT / "data" / "silver"      # Phase 7 LLM 银标签
LIVE_DIR      = PROJECT_ROOT / "data" / "live"        # Phase 8 每日抓取
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"              # 结果、预测、图
MODELS_DIR    = PROJECT_ROOT / "models"               # 保存的模型权重



def download_raw() -> Path:
    """下载并解压 zip，返回包含 txt 文件的文件夹路径。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "FinancialPhraseBank-v1.0.zip"
    if not zip_path.exists():
        print("Downloading Financial PhraseBank ...")
        urllib.request.urlretrieve(ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)
    # 解压后的文件夹名；如果找不到，打印 RAW_DIR 下的内容看看实际叫什么
    folder = RAW_DIR / "FinancialPhraseBank-v1.0"
    if not folder.exists():
        raise FileNotFoundError(f"找不到 {folder}，实际内容：{list(RAW_DIR.iterdir())}")
    return folder


def load_subset(name: str = "allagree") -> pd.DataFrame:
    """读取某个子集，返回列为 sentence, label 的 DataFrame。"""
    folder = download_raw()
    rows = []
    with open(folder / SUBSET_FILES[name], encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sentence, label = line.rsplit("@", 1)   # 从右边切一次，防止句子里本身有 @
            rows.append({"sentence": sentence.strip(), "label": LABEL2ID[label.strip()]})
    return pd.DataFrame(rows)


def make_splits(seed: int = 42) -> None:
    """在最全的 50agree 上做一次 70/15/15 分层切分，再标记每句是否属于 AllAgree。

    这样无论你在哪个子集上训练，测试集都不会和训练集重叠。
    """
    full = load_subset("50agree").drop_duplicates("sentence").reset_index(drop=True)
    train, temp = train_test_split(full, test_size=0.30, stratify=full["label"], random_state=seed)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=seed)

    allagree = set(load_subset("allagree")["sentence"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, df in [("train", train), ("val", val), ("test", test)]:
        df = df.assign(all_agree=df["sentence"].isin(allagree))
        df.to_parquet(PROCESSED_DIR / f"{split_name}.parquet", index=False)
        print(f"{split_name}: {len(df)} rows, all_agree={df['all_agree'].sum()}")


if __name__ == "__main__":
    make_splits()