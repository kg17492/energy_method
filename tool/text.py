import io
import csv
import numpy as np


def replace(text: str, patterns: list[tuple[str, str]]) -> str:
    for source, target in patterns:
        text = text.replace(source, target)
    return text


class String(str):
    """strの拡張クラス
    """

    def stripsplit(self, delimiter: str) -> list["String"]:
        """文字列の前後の<delimiter>を削除したのち、<delimiter>でsplitする。

        Args:
            delimiter (str): splitのdelimiterと同じ。

        ","の場合は、前後ではなく末尾のみ削除する
        """
        return [String(s) for s in (
            self.removesuffix(delimiter) if delimiter == "," else
            self.removesuffix(delimiter).removeprefix(delimiter)
        ).split(delimiter)]

    def multiple_replace(self, *args) -> "String":
        if len(args) % 2 == 1:
            return self
        string: str = self
        for i in range(len(args) // 2):
            old: str = args[2 * i + 0]
            new: str = args[2 * i + 1]
            string = string.replace(old, new)
        return String(string)

    def read_as_dict(self, *keys: list[str]) -> list[dict]:
        def str2float(d: dict[str, str]) -> dict:
            for key in d:
                if d[key] is str:
                    d[key] = (float if key in keys else String)(d[key].strip())
            return d

        with io.StringIO() as fp:
            fp.write(self)
            fp.seek(0)
            return [str2float(d) for d in csv.DictReader(fp)]

    def read_as_table(self) -> "Table":
        with io.StringIO() as fp:
            fp.write(self)
            fp.seek(0)
            return Table([row for row in csv.reader(fp)])

    def to_table(self) -> list[list["String"]]:
        return [[String(col) for col in row] for row in self.read_as_table()]


class Table(list[list[str]]):
    """文字列の配列の配列から欲しい値を得るためのオブジェクト
    """
    def __init__(self, table: list[list[str]]) -> None:
        max_row: int = max([len(row) for row in table])
        super().__init__([
            [col.strip() for col in (
                row + [""] * (max_row - len(row)) if len(row) < max_row else
                row
            )] for row in table
        ])

    def get(self, row: int, col: int) -> str:
        """インデックスで指定された位置の値を返す

        Args:
            row (int): 行
            col (int): 列
        """
        return self[row][col]

    def get_right(self, key: str, count: int = 1) -> str:
        """<key>の右隣の値を返す

        Args:
            key (str): 値の左の見出し
            count (int): いくつ右に行くかを示す。右隣は1から始まる。

        <key>が見つからなければ空文字列を返す。
        """
        for row in self:
            if key in row:
                return row[row.index(key) + count]
        return ""

    def get_below(self, key: str, count: int = 1) -> str:
        """<key>の下の値を返す

        Args:
            key (str): 値の上の見出し
            count (int): いくつ下に行くかを示す。直下は1から始まる。

        <key>が見つからなければ空文字列を返す。
        """
        for i, row in enumerate(self):
            if key in row:
                return self[i + count][row.index(key)]
        return ""

    def get_right_float(self, key: str, count: int = 1) -> float:
        """<key>の右隣の値をfloatに変換して返す

        Args:
            key (str): 値の左の見出し
            count (int): いくつ右に行くかを示す。右隣は1から始まる。

        <key>が見つからなければ空文字列を返す。
        """
        return float(self.get_right(key, count))

    def get_below_float(self, key: str, count: int = 1) -> str:
        """<key>の下の値をfloatに変換して返す

        Args:
            key (str): 値の上の見出し
            count (int): いくつ下に行くかを示す。直下は1から始まる。

        <key>が見つからなければ空文字列を返す。
        """
        return float(self.get_below(key, count))

    def filled_from_left(self) -> "Table":
        for row_idx, row in enumerate(self):
            for col_idx, _ in enumerate(row):
                if all([
                    col_idx > 0,
                    self[row_idx][col_idx] == "",
                    all([
                        self[k][col_idx] == self[k][col_idx - 1] for k in range(row_idx)
                    ]),
                ]):
                    self[row_idx - 0][col_idx] = self[row_idx - 0][col_idx - 1]

                if all([
                    col_idx > 0,
                    self[row_idx - 0][col_idx - 1] == "軸-軸",
                    self[row_idx - 0][col_idx - 0] == "",
                ]):
                    self[row_idx - 0][col_idx - 1] = "左軸"
                    self[row_idx - 0][col_idx - 0] = "右軸"
        return self

    def filled_from_upper(self) -> "Table":
        return self.transposed().filled_from_left().transposed()

    def transposed(self) -> "Table":
        return Table(np.array(self).T)

    def accumulated(self) -> list["String"]:
        return ["".join(row).strip() for row in self.transposed()]

    def to_text(self) -> str:
        with io.StringIO() as fp:
            writer = csv.writer(fp)
            writer.writerows(self)
            fp.seek(0)
            return fp.read()
