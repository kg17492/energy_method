import csv
import json
import pickle
from .text import String
from typing import Any


def read_text(filename: str, encoding: str = "cp932") -> "String":
    """<filename>で指定されるファイルを文字列（ich.String）で返す。

    Args:
        filename (str): ファイルのパス
        encoding (str): 文字コード
    """
    with open(filename, "r", encoding=encoding) as fp:
        return String(fp.read())


def save_csv(filename: str, content: list[list[str]]) -> None:
    """<filename>で指定されるファイル(csv)に<content>を保存する。

    Args:
        filename (str): ファイルのパス
        content (list[list[str]]): 表形式のデータ構造を持つ文字列
    """
    with open(filename, "w", encoding="utf_8_sig", newline="") as fp:
        csv.writer(fp).writerows(content)


def load_json(filename: str, encoding: str = "utf-8") -> Any:
    """<filename>で指定されるjsonファイルを読み取る。

    Args:
        filename (str): ファイルのパス
        encoding (str): 文字コード
    """
    with open(filename, "r", encoding=encoding) as fp:
        return json.load(fp)


def save_json(filename: str, content: Any) -> None:
    """<filename>で指定されるファイル(json)に<content>を保存する。

    Args:
        filename (str): ファイルのパス
        content (Any): 保存するデータ
    """
    with open(filename, "w", encoding="utf-8") as fp:
        json.dump(content, fp, indent=2, ensure_ascii=False)


def load_bin(filename: str) -> Any:
    """<filename>で指定されるbinファイルを読み取る。

    Args:
        - filename: ファイルのパス
    """
    with open(filename, "rb") as fp:
        return pickle.load(fp)


def save_bin(filename: str, content: Any) -> None:
    """<filename>で指定されるファイル(bin)に<content>を保存する。

    Args:
        - filename: ファイルのパス
        - content: 保存するデータ
    """
    with open(filename, "wb") as fp:
        pickle.dump(content, fp)


class CSV_File_Reader:
    filename: str
    encoding: str
    delimiter: str

    def __init__(self, filename: str, encoding: str = "utf_8_sig", delimiter: str = ",") -> None:
        self.filename = filename
        self.encoding = encoding
        self.delimiter = delimiter

    def str2float(self, d: dict[str, str], keys: list[str]) -> dict:
        for key in d:
            if key in keys:
                d[key] = eval(d[key])
        return d

    def read_as_float(self, *keys: list[str]) -> list[dict]:
        def str2float(d: dict[str, str]) -> dict:
            for key in d:
                if key in keys and d[key] is not None:
                    d[key] = eval(d[key])
            return d
        with open(self.filename, "r", encoding=self.encoding) as fp:
            return [str2float(d) for d in csv.DictReader(fp, delimiter=self.delimiter)]

    def read_as_table(self) -> list[list[str]]:
        with open(self.filename, "r", encoding=self.encoding) as fp:
            return [row for row in csv.reader(fp, delimiter=self.delimiter)]
