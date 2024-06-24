import ss7
import numpy as np
from .soil import Soil
from .eigen import Eigen_Analysis
from . import tool
import matplotlib.pyplot as plt
from scipy import interpolate
from typing import Callable, Iterable
import csv
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)


class Data(list[dict[str, str]]):
    def stories(self) -> list[str]:
        return sorted(set([d["階"] for d in self]), reverse=True)

    def get_array(self, key: str) -> np.ndarray:
        return np.array(
            [
                [
                    d[key] for d in filter(lambda d: d["階"] == story, self)
                ] for story in self.stories()
            ],
            dtype=np.float64
        ).T


class Energy_Method(ss7.SS7_Reader):
    load: str
    soil: Soil
    story_shear: np.ndarray
    column_shear: np.ndarray
    brace_shear: np.ndarray
    story_delta: np.ndarray
    step_delta_ultimate: np.ndarray
    t_sample: np.ndarray = None
    gs_sample: np.ndarray
    G: float = 9.80665
    E: float = 205000
    SIGMA_Y: float = 225
    mechanism: str
    damper_amplification: float = 1.00
    LAMBDA: np.ndarray = np.linspace(0, 1, 100)
    frame_bilinear: bool = True
    mode_initial_stiffness: str = "接線"
    dmax: float = 1 / 100

    def __init__(self, ss7_filename: str, load: str, include_spectrum: str = None) -> None:
        """
        Args:
            - ss7_filename SS7の出力ファイル
                - 以下を含む必要あり
                - 一部を省略しても、一部の検討は行うことが可能。
                - SS7のバージョンにより番号が一部異なる
                - 2.架構認識
                - 2.1.構造階高
                - 2.2.構造スパン
                - 2.7.10.鉛直ブレース
                - 3.剛性計算
                - 3.5.部材剛性（鉛直ブレース）
                - ダンパーの保有累積塑性変形倍率の計算に必要。
                - 4.荷重計算
                - 4.4.地震用重量
                - 4.5.地震力基本データ
                - 4.6.地震力
                - 5.応力解析（一次）
                - 5.26.Q-δ
                - 一次解析のスケルトンカーブを用いた解析に必要。
                - 10.ルート判定
                - 10.9.偏心率（雑壁なし）
                - 10.6.偏心率（雑壁なし） ver 1.1.1.15
                - 12.応力解析（二次）
                - 12.23.Q-δ
                - 12.24.Q-δ ver 1.1.1.19
                - 13.必要保有水平耐力
                - 13.2.部材群の種別
                    - 13.2.1.柱梁
                - 13.6.未降伏部材の降伏判定
                    - 13.6.1.節点の曲げ余裕度（節点振分法）
                - 主架構の保有累積塑性変形倍率の計算に必要
            - load 載荷
                - EX+
                - EX-
                - EY+
                - EY-
                - DSX+
                - DSX-
                - DSY+
                - DSY-
            - include_spectrum アンプリファイアのinclude_spectrum.csvのパス
        """
        super().__init__(ss7_filename)
        self.load = load
        data: Data = Data(self.get_q_delta())
        self.story_shear = data.get_array("Q合計kN")
        self.column_shear = data.get_array("Q柱kN")
        self.brace_shear = data.get_array("QブレースkN")
        self.story_delta = data.get_array("層間変位重心位置mm")
        self.soil = Soil(self.soil_int(), self.region_factor())
        self.step_delta_ultimate = np.mod(
            np.argmax(self.story("層間変形角") > self.dmax, axis=0) - 1,
            len(self.story_delta),
        )
        row: int = 0 if "X" in load else 1
        col: int = 0 if "+" in load else 1
        self.mechanism = (
            self.get("崩壊メカニズム")[row][col][4:] if "崩壊メカニズム" in self.keys() else
            "部分崩壊形"
        )
        if type(include_spectrum) is float:
            self.t_sample = [0, 10]
            self.gs_sample = [include_spectrum] * 2
        elif include_spectrum is not None:
            data: list[dict[str, float]] = tool.CSV_File_Reader(include_spectrum, encoding="cp932").read_as_float("固有周期(s)", "Gs")
            self.t_sample = np.array([d["固有周期(s)"] for d in data])
            self.gs_sample: np.ndarray = np.array([d["Gs"] for d in data])

    def get_q_delta(self) -> list[dict[str, str]]:
        return self.get(f'Q-δ({"一次" if self.load in ["EX+", "EX-", "EY+", "EY-"] else "二次"}) {self.load}')

    def floors(self, include_dammy: bool = False) -> list[str]:
        return [row["層"] for row in filter(lambda d: include_dammy or d["ダミー層"] == "通常層", self.get("構造階高"))]

    def load_in_japanese(self) -> str:
        return f'{self.direction()}{"正" if "+" in self.load else "負"}加力'

    def all_floor_mechanism_is(self, floor: str, mechanism: str) -> bool:
        return all([x["崩壊形"] == mechanism for x in filter(lambda x: x["層"] == floor, self.get(f'節点の曲げ余裕度(節点振分法) {self.load_in_japanese()}'))])

    def is_side_sway(self) -> bool:
        return all([self.all_floor_mechanism_is(floor, "梁") for floor in self.floors()[1:-1]])

    def stories(self, include_dammy: bool = False) -> list[str]:
        return [row["階"] for row in filter(
            lambda d: include_dammy or d["階"] in set([x["階"] for x in self.get_q_delta()]),
            self.get("構造階高")[:-1]
        )]

    def brace_is_damper(self, story: str = None) -> np.ndarray:
        if story is None:
            return np.array([
                self.brace_is_damper(s) for s in self.stories()
            ])
        else:
            return any([x["材料"] == "LYP225" for x in filter(lambda x: x["階"] == story, self.get("鉛直ブレース部材断面情報"))])

    def direction(self) -> str:
        return self.load[-2]

    def orthogonal_direction(self) -> str:
        return "Y" if self.direction() == "X" else "X"

    def number_of_stories(self) -> int:
        return len(self.get("地震力 X加力"))

    def design_shear(self) -> np.ndarray:
        return np.array([d["一次設計用Qi1kN"] if "一次設計用Qi1kN" in d else d["1次設計用Qi1kN"] for d in self.get(f"地震力 {self.direction()}加力")], dtype=np.float64)

    def ci_factor(self) -> np.ndarray:
        return np.array([d["一次設計用Ci1"] if "一次設計用Ci1" in d else d["1次設計用Ci1"] for d in self.get(f"地震力 {self.direction()}加力")], dtype=np.float64)

    def ai_factor(self) -> np.ndarray:
        return np.array([d["Ai"] for d in self.get(f"地震力 {self.direction()}加力")], dtype=np.float64)

    def eccentricity(self) -> np.ndarray:
        # return 0.1
        eccentricity: np.ndarray = np.max(np.array([
            [d["偏心率Re"] for d in self.get(f"偏心率(雑壁なし) {load}")] for load in [
                "EX+EY+",
                "EX+EY-",
                "EX-EY+",
                "EX-EY-",
            ]
        ], dtype=np.float64), axis=0)
        n: int = self.number_of_stories()
        return eccentricity[0:n] if self.direction() == "X" else eccentricity[n:]

    def soil_int(self) -> int:
        table: tool.Table = tool.Table(self.get("地震力基本データ")[0])
        tc: str = table.get_right("地盤種別による係数 Tc", 3)
        if tc == "1.00":
            return 2
        return ["", "0.40", "0.60", "0.80"].index(tc)

    def inputted_period(self) -> str:
        table: tool.Table = tool.Table(self.get("地震力基本データ")[0])
        return table.get_right(
            "一次固有周期T",
            3 if self.direction() == "X" else 4,
        )[:-1]

    def rt(self, td: np.ndarray = None) -> np.ndarray:
        if td is None:
            td = self.period()
        table: tool.Table = tool.Table(self.get("地震力基本データ")[0])
        tc: float = table.get_right_float("地盤種別による係数 Tc", 3)
        return np.sum([
            td < tc,
            (1 - 0.2 * (td / tc - 1) ** 2) * (tc <= td) * (td < 2 * tc),
            1.6 * tc / td * (2 * tc < td),
        ], axis=0)

    def gs(self, td: np.ndarray) -> np.ndarray:
        if self.t_sample is None:
            return self.soil.gs(td)
        else:
            return interpolate.interp1d(
                self.t_sample,
                self.gs_sample
            )(
                td
            )

    def region_factor(self) -> float:
        table: tool.Table = tool.Table([d[0] for d in self.get("地震力基本データ")])
        return table.get_right_float("地域係数 Z", 3)

    def period(self, step: str = None) -> np.ndarray:
        m: np.ndarray = self.story("地震用質量")
        period: np.ndarray = np.array([Eigen_Analysis(np.flip(m), np.flip(k) * 1000).period() for k in self.story("割線剛性")])
        if step is None:
            return period
        else:
            return period[self.step(step)]

    def velocity(self, key: str) -> np.ndarray:
        """エネルギー換算速度

        Args:
            - key:
                - 稀地震
                - 極稀地震
        """
        if key == "稀地震":
            return self.soil.vd(self.period())
        elif key == "極稀地震":
            return self.soil.vs(
                self.period("安全限界"),
                self.gs(self.period("安全限界")),
            )
        else:
            return self.soil.vs(
                self.period(),
                self.gs(self.period()),
            )

    def inputted_energy(self, key: str) -> np.ndarray:
        """地震動によって入力されるエネルギー

        Args:
            - key:
                - 稀地震
                - 極稀地震
        """
        m: float = sum(self.story("地震用質量"))    # t
        v: float = self.velocity(key)               # m/s
        return m * v ** 2 / 2 * 1000                # kN m

    def necessary_energy_capacity(self) -> np.ndarray:
        return self.inputted_energy("極稀地震") - self.building("安全限界時吸収エネルギー")

    def converge_delta_ultimate(self, n: int = 10) -> None:
        """想定最大応答変位と最大応答変位を収束計算する"""
        for i in range(n):
            step_delta_ultimate: np.ndarray = np.mod(
                np.argmax(self.story_delta > self.story("最大応答変位")[self.step("主架構降伏")], axis=0) - 1,
                len(self.story_delta),
            )
            if (self.step_delta_ultimate == step_delta_ultimate).all():
                print(f"最大応答変位は{i}回目の計算で収束しました。")
                return
            elif (self.story("最大応答変位")[self.step("主架構降伏")] < self.dmax).all() and (self.frame("想定最大応答変位") > self.dmax).all() and (np.abs(self.step_delta_ultimate - step_delta_ultimate) < 5).all():
                print(f"最大応答変位は{i}回目の計算で収束とみなせました。")
                return
            else:
                random: float = np.random.rand()
                self.step_delta_ultimate = np.array(
                    np.round(((1 + random) * step_delta_ultimate + self.step_delta_ultimate) / (2 + random)),
                    dtype=np.int64,
                )
        print(f"最大応答変位を{n}回計算しましたが収束しませんでした。")

    def check_ai_factor(self, filename: str = None) -> None:
        """初期剛性でのAi分布と損傷限界時のAi分布を比較する"""
        plt.figure(figsize=(3, 3))
        plt.plot(self.ai_factor(), self.story("Ai分布")[self.step("損傷限界")], "-o")
        xmax: float = max([plt.xlim()[1], plt.ylim()[1]])
        plt.plot([0, xmax], [0, xmax], "--", color="black")
        plt.xlim(0, xmax)
        plt.ylim(0, xmax)
        plt.xlabel("初期剛性Ai分布")
        plt.ylabel("損傷限界Ai分布")
        plt.grid()
        if filename is not None:
            tool.savefig(filename)

    def building(self, key: str) -> np.ndarray:
        """各ステップでの建物全体の応答を返す

        Args:
            - key:
                - 損傷限界時吸収エネルギー
                - 安全限界時吸収エネルギー
                - 平均層間変形角
                - 最大層間変形角
                - ベースシア係数
        """
        if key == "損傷限界時吸収エネルギー":
            return np.sum(self.story("損傷限界時吸収エネルギー"), axis=1)
        elif key == "安全限界時吸収エネルギー":
            return np.sum(self.story("安全限界時吸収エネルギー"), axis=1)
        return {
            "平均層間変形角": np.sum(self.story("層間変位"), axis=1) / np.sum(self.story("構造階高")),
            "最大層間変形角": np.max(self.story("層間変形角"), axis=1),
            "ベースシア係数": self.story("せん断力")[:, -1] / sum(self.story("地震用質量")) / self.G / self.soil.z / self.rt(),
        }[key]

    def story(self, key: str) -> np.ndarray:
        """各ステップでの層の応答・応力を返す

        Args:
            - key:
                - 必要吸収エネルギー
                - 保有水平耐力
                - 最大応答変位
                - 最大応答変形角
                - 損傷限界時吸収エネルギー
                - 安全限界時吸収エネルギー
                - Ai分布
                - 層間変形角
                - せん断力
                - 層間変位
                - 割線剛性
                - 構造階高
                - 意匠階高
                - 地震用質量
        """
        if key == "必要吸収エネルギー":
            mi: np.ndarray = self.story("地震用質量")
            es: float = self.necessary_energy_capacity()
            ai: np.ndarray = self.story("Ai分布")[self.step("安全限界")]
            si: np.ndarray = np.prod([
                np.cumsum(mi) / sum(mi),
                np.cumsum(mi) / sum(mi),
                ai,
                ai,
                self.frame("降伏変位") / self.frame("降伏変位")[-1],
                self.story("保有水平耐力")[-1] / self.story("保有水平耐力"),
            ], axis=0)
            alpha_i: np.ndarray = self.story("保有水平耐力") / np.cumsum(mi) / self.G
            pi: np.ndarray = alpha_i / alpha_i[-1] / ai
            re: np.ndarray = self.eccentricity()
            pti: np.ndarray = np.sum([
                1 * (re <= 0.15),
                (1.15 - re) * (0.15 < re) * (re < 0.3),
                0.85 * (0.3 <= re),
            ], axis=0)
            n: float = 4 if self.mechanism == "全体崩壊形" else 8
            si_pi_pti_n: np.ndarray = si * (pi * pti) ** (-n)
            return np.array(np.matrix(es).T * np.matrix(si_pi_pti_n)) / sum(si_pi_pti_n)
        elif key == "保有水平耐力":
            return self.frame("保有水平耐力") + self.damper("保有水平耐力")
        elif key == "最大応答変位":
            ni: float = 2
            es_bool: np.ndarray = np.array(np.matrix(self.necessary_energy_capacity() > 0).T)
            return np.sum([
                self.frame("降伏変位") * (self.frame("必要累積塑性変形倍率") / ni + 1) * es_bool,
                self.story("層間変位")[self.step("極稀地震")] * ~es_bool,
            ], axis=0)
        elif key == "最大応答変形角":
            return self.story("最大応答変位") / self.story("構造階高")
        elif key in [
            "損傷限界時吸収エネルギー",
            "安全限界時吸収エネルギー",
        ]:
            n: float = 2 if key == "損傷限界時吸収エネルギー" else 5
            return np.sum([
                self.frame(f"{key[0:4]}時弾性エネルギー"),
                self.damper("弾性エネルギー"),
                self.damper("塑性エネルギー") * 2 * n,
            ], axis=0)
        elif key == "Ai分布":
            alpha_i: np.ndarray = np.cumsum(self.story("地震用質量")) / sum(self.story("地震用質量"))
            return np.array(1 + np.matrix(2 * self.period() / (1 + 3 * self.period())).T * np.matrix(1 / np.sqrt(alpha_i) - alpha_i))
        elif key == "層間変形角":
            return self.story_delta / self.story("意匠階高")
            return self.story_delta / self.story("構造階高")
        elif key == "割線剛性":
            return self.story("せん断力") / self.story("層間変位")

        return {
            "せん断力": self.story_shear,
            "層間変位": self.story_delta,
            "構造階高": self.story_height("構造階高mm"),
            "意匠階高": self.story_height("階高mm"),
            "地震用質量": self.story_mass(),
        }[key]

    def story_height(self, key: str) -> np.ndarray:
        story_height: np.ndarray = np.zeros(self.number_of_stories())
        i: int = 0
        for row in self.get("構造階高")[:-1]:
            if row["従属層"] == "上層":
                i -= 1
            story_height[i] += float(row[key])
            i += 1
            if row["従属層"] == "下層":
                i -= 1
        return story_height

    def story_mass(self) -> np.ndarray:
        story_mass: np.ndarray = np.zeros(self.number_of_stories())
        dummy_story: list[dict[str, str]] = self.get("構造階高")
        i: int = 0
        for j, row in enumerate(self.get("地震用重量")):
            if dummy_story[j]["従属層"] == "上層":
                i -= 1
            if i > self.number_of_stories() - 1:
                break
            story_mass[i] += float(row["wi(wi/A)kN"]) / self.G
            i += 1
            if dummy_story[j]["従属層"] == "下層":
                i -= 1
        return story_mass

    def initial_stiffness(self, mode: str = None) -> str:
        """

        Args:
            - mode:
                - 接線
                - 割線
                - 最大
        """
        if mode is not None:
            self.mode_initial_stiffness = mode
        return self.mode_initial_stiffness

    def frame(self, key: str) -> np.ndarray:
        """各ステップでの主架構の応答・応力を返す

        Args:
            - key:
                - 降伏済み
                - 初期剛性
                - 降伏耐力
                - 降伏変位
                - 保有水平耐力
                - 損傷限界時弾性エネルギー
                - 安全限界時弾性エネルギー
                - 累積吸収エネルギー
                - 必要吸収エネルギー
                - 必要累積塑性変形倍率
                - 想定最大応答変位
                - 崩壊モード
                - 部材種別
                - 保有累積塑性変形倍率
                - 層間変位
                - 層間変形角
                - せん断力
        """
        if key == "降伏済み":
            return self.story("層間変位") > self.frame("降伏変位")
        elif key == "初期剛性":
            if self.initial_stiffness() == "最大":
                return np.max(self.frame("割線剛性"), axis=0)
            elif self.initial_stiffness() == "接線":
                return self.frame("割線剛性")[0, :]
            elif self.initial_stiffness() == "割線":
                return self.frame("割線剛性")[self.step("損傷限界"), :]
        elif key == "接線剛性":
            return np.diff(self.frame("せん断力"), axis=0, prepend=0) / np.diff(self.frame("層間変位"), axis=0, prepend=0)
        elif key == "割線剛性":
            return self.frame("せん断力") / self.frame("層間変位")
        elif key == "降伏耐力":
            # a: kN/mm, b: mm, c: kNmm
            step: tuple[np.ndarray, list[int]] = self.step("想定最大応答変位")
            ke: np.ndarray = self.frame("初期剛性")
            dmax: np.ndarray = self.frame("層間変位")[*step]
            ecum: np.ndarray = self.frame("累積吸収エネルギー")[*step]
            a: np.ndarray = 1 / 2 / ke
            b: np.ndarray = dmax
            c: np.ndarray = ecum
            return np.sum([
                np.nan_to_num((b - np.sqrt(b**2 - 4 * a * c)) / 2 / a) * (b**2 - 4 * a * c > 0),
                ke * dmax * (b**2 - 4 * a * c <= 0),
            ], axis=0) if self.frame_bilinear else self.frame("保有水平耐力")
        elif key == "降伏変位":
            return self.frame("降伏耐力") / self.frame("初期剛性")
        elif key == "保有水平耐力":
            return (
                self.frame("降伏耐力") if self.frame_bilinear else
                self.frame("せん断力")[self.step("保有水平耐力")]
            )
        elif key == "損傷限界時弾性エネルギー":
            return self.frame("層間変位") ** 2 * self.frame("初期剛性") / 2
        elif key == "安全限界時弾性エネルギー":
            return np.sum([
                self.frame("降伏耐力") ** 2 / self.frame("初期剛性") * self.frame("降伏済み"),
                self.frame("層間変位") ** 2 * self.frame("初期剛性") * ~self.frame("降伏済み"),
            ], axis=0) / 2
        elif key == "累積吸収エネルギー":
            d_zero_inserted: np.ndarray = np.concatenate([
                np.zeros((1, self.number_of_stories())),
                self.frame("層間変位")
            ])
            q_zero_inserted: np.ndarray = np.concatenate([
                np.zeros((1, self.number_of_stories())),
                self.frame("せん断力")
            ])
            d_delta: np.ndarray = np.diff(d_zero_inserted, axis=0)
            q_mean: np.ndarray = (q_zero_inserted[:-1, :] + q_zero_inserted[1:, :]) / 2
            return np.cumsum(q_mean * d_delta, axis=0)
        elif key == "必要吸収エネルギー":
            return self.story("必要吸収エネルギー") * self.frame("保有水平耐力") / self.story("保有水平耐力")
        elif key == "必要累積塑性変形倍率":
            return self.frame("必要吸収エネルギー") / 2 / self.frame("保有水平耐力") / self.frame("降伏変位")
        elif key == "想定最大応答変位":
            return self.frame("層間変位")[*self.step(key)]
        elif key == "崩壊モード":
            return [
                "柱" if any([
                    self.all_floor_mechanism_is(self.floors()[i], "柱"),
                    self.all_floor_mechanism_is(self.floors()[i + 1], "柱"),
                ]) else "梁" for i in range(self.number_of_stories())
            ]
        elif key == "部材種別":
            return [f'F{x["種別"].strip()}' for x in self.get(f"柱・梁群の種別 {self.load_in_japanese()}")]
        elif key == "保有累積塑性変形倍率":
            eta_cri: dict[str, dict[str, float]] = {
                "柱": {
                    "FA": 7.0,
                    "FB": 3.25,
                    "FC": 2.0,
                    "FD": 1.0,
                },
                "梁": {
                    "FA": 5.0,
                    "FB": 2.75,
                    "FC": 2.0,
                    "FD": 1.0,
                },
            }
            mechanisms: list[str] = self.frame("崩壊モード")
            member_types: list[str] = self.frame("部材種別")
            return np.array([
                eta_cri[mechanisms[i]][member_types[i]] for i in range(self.number_of_stories())
            ])

        return {
            "層間変位": self.story_delta,
            "層間変形角": self.story("層間変形角"),
            "せん断力": self.column_shear + self.brace_shear * ~self.brace_is_damper(),
        }[key]

    def damper(self, key: str) -> np.ndarray:
        """各ステップでのダンパーの応答・応力を返す

        Args:
            - key:
                - 降伏済み
                - 初期剛性
                - 降伏耐力
                - 降伏変位
                - 保有水平耐力
                - 弾性エネルギー
                - 塑性エネルギー
                - 残留変位
                - 残留変形角
                - 必要吸収エネルギー
                - 必要累積塑性変形倍率
                - 保有累積塑性変形倍率
                - 層間変位
                - 層間変形角
                - せん断力
        """
        if key == "降伏済み":
            return self.story("層間変位") > self.damper("降伏変位")
        elif key == "初期剛性":
            step: int = 1
            # return np.sum([
            #     np.diff(self.damper("せん断力"), axis=0)[step] / np.diff(self.damper("層間変位"), axis=0)[step] * self.brace_is_damper(),
            #     ~self.brace_is_damper(),
            # ], axis=0)
            return np.sum([
                self.damper("せん断力")[step, :] / self.damper("層間変位")[step, :] * self.brace_is_damper(),
                ~self.brace_is_damper(),
            ], axis=0)
        elif key == "接線剛性":
            return np.diff(self.damper("せん断力"), axis=0) / np.diff(self.damper("層間変位"), axis=0)
        elif key == "降伏耐力":
            return np.max(self.damper("せん断力"), axis=0)
        elif key == "降伏変位":
            return self.damper("降伏耐力") / self.damper("初期剛性")
        elif key == "保有水平耐力":
            return np.max(self.damper("せん断力"), axis=0)
        elif key == "弾性エネルギー":
            return np.sum([
                self.damper("降伏耐力") ** 2 / self.damper("初期剛性") * self.damper("降伏済み"),
                self.damper("層間変位") ** 2 * self.damper("初期剛性") * ~self.damper("降伏済み"),
            ], axis=0) / 2
        elif key == "塑性エネルギー":
            return (self.story("層間変位") - self.damper("降伏変位")) * self.damper("保有水平耐力") * self.damper("降伏済み")
        elif key == "残留変位":
            kappa: np.ndarray = self.damper("初期剛性") / self.frame("初期剛性")
            plasticity: np.ndarray = self.story("層間変位") / self.damper("降伏変位")
            residuality: np.ndarray = np.sum([
                (kappa / (1 + kappa) * (plasticity - 1)) * (plasticity < (kappa + 5) / 4),
                (kappa / 4) * ((kappa + 5) / 4 < plasticity) * (kappa < 5 / 3),
                (kappa / (1 + kappa) * (1 + kappa - np.sqrt((plasticity - kappa) ** 2 + 3 * kappa - 1))) * ((kappa + 5) / 4 < plasticity) * (plasticity < kappa) * (5 / 3 < kappa),
                (kappa / (1 + kappa) * (1 + kappa - np.sqrt(3 * kappa - 1))) * (5 / 3 < kappa) * (kappa < plasticity),
            ], axis=0)
            return residuality * self.damper("降伏変位")
        elif key == "残留変形角":
            return self.damper("残留変位") / self.story("構造階高")
        elif key == "必要吸収エネルギー":
            nsi: float = 20
            beta: float = 5
            ndi: float = 10
            dy: np.ndarray = self.damper("降伏変位")

            def di(key: str) -> np.ndarray:
                return self.story("層間変位")[self.step(key)]

            esi_qdui_qui: np.ndarray = self.story("必要吸収エネルギー") * self.damper("保有水平耐力") / self.story("保有水平耐力")
            esdpi: np.ndarray = 2 * nsi * (di("主架構降伏") - dy) * self.damper("保有水平耐力") * (di("主架構降伏") > dy)
            beta_eddpi: np.ndarray = 2 * beta * ndi * (di("稀地震") - dy) * self.damper("保有水平耐力") * (di("稀地震") > dy)
            return esi_qdui_qui + esdpi + beta_eddpi
        elif key == "必要累積塑性変形倍率":
            return self.damper("必要吸収エネルギー") / 2 / self.damper("保有水平耐力") / self.damper("降伏変位")
        elif key == "個数":
            return np.sum([
                np.array([len(self.dampers(story)) for story in self.stories()]) * self.brace_is_damper(),
                np.array([np.nan if not b else 0 for b in self.brace_is_damper()]),
            ], axis=0)
        elif key == "部材長":
            return np.array([
                self.zero_when_zero_length(min, tool.flatten([
                    [
                        float(d[key]) for key in filter(lambda key: key.endswith("部材長mm"), d.keys())
                    ] for d in self.dampers(story)
                ])) for story in self.stories()
            ]) * self.brace_is_damper()
        elif key == "断面積":
            return np.array([
                self.zero_when_zero_length(max, [
                    float(d["Aocm2"]) if "Aocm2" in d else
                    min([
                        float(d[key]) for key in filter(lambda key: key.endswith("Acm2"), d.keys())
                    ]) for d in self.dampers(story)
                ]) for story in self.stories()
            ]) * self.brace_is_damper()
        elif key == "保有累積塑性変形倍率":
            skd: np.ndarray = self.damper("初期剛性") / self.damper("個数")
            lbr: np.ndarray = self.damper("部材長")
            lh: np.ndarray = self.story("意匠階高")
            ld: np.ndarray = np.sqrt(lbr ** 2 - lh ** 2)
            abr: np.ndarray = self.damper("断面積")
            gamma_lambda: np.ndarray = lbr / 2 / self.E / abr * (lbr / ld) ** 2 * skd * 1000   # 注意
            # gammaは小さいほど安全側
            gamma: np.ndarray = np.array([self.LAMBDA]).T * np.array([gamma_lambda])
            mdmaxi: np.ndarray = (self.story("最大応答変位")[self.step("主架構降伏")] / self.damper("降伏変位") - 1) / gamma + 1
            ey: float = self.SIGMA_Y / self.E
            et: np.ndarray = 2 * mdmaxi * ey
            nf: np.ndarray = (100 * et / 20.48) ** (-1 / 0.49)
            return (et - 2 * ey) * nf / ey * gamma

        return {
            "層間変位": self.story_delta,
            "層間変形角": self.story_delta,
            "せん断力": self.brace_shear * self.brace_is_damper() * self.damper_amplification,
        }[key]

    def zero_when_zero_length(self, func: Callable, array: Iterable[float]) -> float:
        return 0 if len(array) == 0 else func(array)

    def dampers(self, story: str) -> list[dict[str, str]]:
        return [d for d in filter(lambda x: all([
            x["階"] == story,
            x["ﾌﾚｰﾑ"] in self.get_axes(),
        ]), self.get("鉛直ブレース剛性表"))]

    def get_axes(self) -> list[str]:
        return [d[f'<{self.orthogonal_direction()}方向>構造心とのズレ軸'] for d in self.get("構造スパン")]

    def count_dampers(self) -> None:
        for story in self.stories():
            number_of_dampers: int = len([f for f in filter(
                lambda x: x["階"] == story and x["ﾌﾚｰﾑ"] in self.get_axes(),
                self.get("鉛直ブレース部材断面情報"),
            )])
            print(f"\t\t{story}:\t{number_of_dampers}")

    def step(self, key: str) -> int | tuple[np.ndarray, list[int]]:
        """keyで指定される状態に到達するステップを返す

        Args:
            - key:
                - 損傷限界
                - 安全限界
                - 稀地震
                - 極稀地震
                - 主架構降伏
                - 最大応答変位
                - 想定最大応答変位
        """
        if key == "損傷限界":
            steps: np.ndarray = np.mod(
                np.argmax(self.story_shear > self.design_shear(), axis=0) - 1,
                len(self.story_shear),
            )
            return min(steps)
        elif key == "安全限界":
            ts_td: float = 1.2 if np.all(self.brace_shear == 0) else 1.4
            return np.clip(
                a=np.argmax(self.velocity(None)),
                a_min=self.step("損傷限界"),
                a_max=np.argmax(self.period() > ts_td * self.period("損傷限界")),
            )
        elif key == "稀地震":
            return np.argmax(self.building("損傷限界時吸収エネルギー") > self.inputted_energy("稀地震")[self.step("損傷限界")])
        elif key == "極稀地震":
            return np.argmax(self.building("安全限界時吸収エネルギー") > self.inputted_energy("極稀地震"))
        elif key == "主架構降伏":
            return (
                np.argmax(np.any(self.frame("降伏済み"), axis=1)) - 1 if self.frame_bilinear else
                self.step("損傷限界")
            )
        elif key == "ダンパー降伏":
            return (np.argmax(self.damper("降伏済み"), axis=0) - 1, [i for i in range(self.number_of_stories())])
        elif key == "想定最大応答変位":
            return (self.step_delta_ultimate, [i for i in range(self.number_of_stories())])
        # elif key == "想定最大応答変位":
        #     steps: np.ndarray = len(self.building("最大層間変形角")) - np.argmax(np.flipud(self.frame("累積吸収エネルギー") < self.frame("層間変位") * self.frame("降伏耐力") - self.frame("降伏耐力") ** 2 / 2 / self.frame("初期剛性")), axis=0) - 1
        #     return (steps, [i for i in range(self.number_of_stories())])
        elif key == "保有水平耐力":
            return np.argmax(self.building("最大層間変形角") > self.dmax) - 1
        else:
            return 0

    def plot_step(self, func: str, step_key: str) -> None:
        """keyで指定されたステップでのfuncの応答を図示する

        Args:
            - func:
                - story
                - frame
                - damper
            - step_key: stepに同じ
        """
        theta: np.ndarray = getattr(self, func)("層間変形角")
        shear: np.ndarray = getattr(self, func)("せん断力") / self.design_shear()
        step: int | tuple[int | np.ndarray, list[int]] = self.step(step_key)
        if type(step) is not tuple:
            step = (step,)
        plt.plot(theta[*step], shear[*step], "-o", color="black", lw=0.5)

    def plot_vline(self, x_axis: str, step: str) -> None:
        """stepでのx_axisを図示する

        Args:
            x_axis: buildingの引数
            step: stepの引数
        """
        x: float = self.building(x_axis)[self.step(step)]
        ymin: float = 0
        ymax: float = plt.ylim()[1]
        plt.vlines(x, ymin, ymax, color="black")
        plt.ylim(ymin, ymax)

    def plot_xlim_and_ylim(self, xmax: float = None, ymax: float = None) -> None:
        if xmax is None:
            xmax = plt.xlim()[1]
        if ymax is None:
            ymax = plt.ylim()[1]
        plt.xlim(0, xmax)
        plt.ylim(0, ymax)
        plt.grid()

    def plot(self, key: str) -> None:
        """keyで指定された図を描く

        Args:
            - key:
                - 損傷限界
                - 固有周期
                - 損傷限界時吸収エネルギー
                - 残留層間変形角
                - 安全限界
                - 想定最大応答変形角
                - 安全限界時吸収エネルギー
                - 最大応答変形角
                - 主架構必要累積塑性変形倍率
                - ダンパー必要累積塑性変形倍率
                - 最適ブレース分担率
                - 保有水平耐力分布
                - ダンパー有効性
                - Gs
                - ステップ-接線剛性
                - ステップ-分担割合
                - ステップ-せん断力
                - 層間変形角-せん断力
                - ステップ-層せん断力
                - 層間変形角-層せん断力
                - ステップ-層間変形角
                - バイリニア置換確認
        """
        if key == "損傷限界":
            xmax: float = 1 / 200
            plt.plot(self.story("層間変形角"), self.story("せん断力"))
            self.plot_step("story", "損傷限界")
            self.plot_vline("最大層間変形角", "損傷限界")
            ymax: float = plt.ylim()[1]
            self.plot_xlim_and_ylim(xmax, ymax)
            plt.xlabel("層間変形角")
            plt.ylabel("層せん断力")
        elif key == "安全限界":
            xmax: float = 1 / 50
            for i in range(self.number_of_stories()):
                plt.plot(
                    self.story("層間変形角")[:, i],
                    self.frame("せん断力")[:, i] / self.design_shear()[i],
                    color=tool.colors(i),
                    label=self.stories()[i],
                )
                plt.plot(
                    [
                        0,
                        self.frame("降伏変位")[i] / self.story("構造階高")[i],
                        self.story("層間変形角")[*self.step("想定最大応答変位")][i],
                    ],
                    [
                        0,
                        self.frame("降伏耐力")[i] / self.design_shear()[i],
                        self.frame("降伏耐力")[i] / self.design_shear()[i],
                    ],
                    "--",
                    color=tool.colors(i),
                    label=None,
                )
            self.plot_step("frame", "主架構降伏")
            self.plot_vline("最大層間変形角", "主架構降伏")
            ymax: float = plt.ylim()[1]
            self.plot_xlim_and_ylim(xmax, ymax)
            plt.xticks(np.linspace(0, 0.02, 9), [f"{1 / x:.0f}" for x in np.linspace(0, 0.02, 9)])
            plt.xlim(0, xmax)
            plt.xlabel("層間変形角")
            plt.ylabel("層せん断力")
            plt.legend()
        elif key == "固有周期":
            xmax: float = 1 / 200
            plt.plot(self.building("最大層間変形角"), self.period())
            self.plot_vline("最大層間変形角", "損傷限界")
            self.plot_xlim_and_ylim(xmax, 1.4)
            plt.xlabel("最大層間変形角")
            plt.ylabel("固有周期")
        elif key == "損傷限界時吸収エネルギー":
            plt.plot(self.building("ベースシア係数"), np.sum(self.frame("損傷限界時弾性エネルギー"), axis=1), "--", label="主架構")
            plt.plot(self.building("ベースシア係数"), np.sum(self.damper("弾性エネルギー"), axis=1), "--", label="ダンパー弾性")
            plt.plot(self.building("ベースシア係数"), np.sum(self.damper("塑性エネルギー"), axis=1) * 2 * 2, "--", label="ダンパー塑性")
            plt.plot(self.building("ベースシア係数"), self.inputted_energy("稀地震"), label="必要吸収")
            plt.plot(self.building("ベースシア係数"), self.building("損傷限界時吸収エネルギー"), "-o", label="総和")
            plt.legend()
            plt.vlines(x=[
                self.building("ベースシア係数")[self.step("損傷限界")],
            ], ymin=0, ymax=1e7, color="black")
            self.plot_xlim_and_ylim(
                0.3,
                np.ceil(
                    1.1 * max(
                        self.inputted_energy("稀地震")
                    ) / 1e6
                ) * 1e6,
            )
            plt.xticks(np.arange(0, 0.31, 0.01), [f"{i * 0.01:.2f}" if i % 5 == 0 else "" for i in range(31)])
            c0_current: float = self.building("ベースシア係数")[self.step("損傷限界")]
            c0_optical: float = self.building("ベースシア係数")[self.step("稀地震")]
            c0_inputted: float = tool.Table(self.get("地震力基本データ")[0]).get_right_float("標準せん断力係数", 2 if self.direction() == "X" else 3)
            plt.title("\n".join([
                f'入力 {self.inputted_period()} {c0_inputted:.3f}',
                f'現状 {self.period("損傷限界"):.3f} {c0_current:.3f}',
                f'最適 {self.period("稀地震"):.3f} {c0_optical:.3f}',
            ]))
            plt.xlabel("ベースシア係数C0")
            plt.ylabel("損傷限界吸収エネルギー")
        elif key == "安全限界時吸収エネルギー":
            xmax: float = 1 / 50
            plt.plot(self.building("最大層間変形角"), np.sum(self.frame("安全限界時弾性エネルギー"), axis=1), "--")
            plt.plot(self.building("最大層間変形角"), np.sum(self.damper("弾性エネルギー"), axis=1), "--")
            plt.plot(self.building("最大層間変形角"), np.sum(self.damper("塑性エネルギー"), axis=1) * 2 * 5, "--")
            plt.hlines(y=self.inputted_energy("極稀地震"), xmin=0, xmax=xmax, label="極稀地震")
            plt.plot(self.building("最大層間変形角"), self.building("安全限界時吸収エネルギー"), label="安全限界")
            self.plot_xlim_and_ylim(
                xmax,
                int(
                    2 * self.inputted_energy("極稀地震") / 1e7
                ) * 1e7,
            )
            plt.xticks(np.linspace(0, 0.02, 9), [f"{1 / x:.0f}" for x in np.linspace(0, 0.02, 9)])
            self.plot_vline("最大層間変形角", "安全限界")
            self.plot_vline("最大層間変形角", "主架構降伏")
            plt.xlabel("最大層間変形角")
            plt.ylabel("エネルギー")
            plt.legend()
        elif key == "残留層間変形角":
            xmax: float = 1 / 200
            plt.plot(self.building("最大層間変形角"), self.damper("残留変形角"))
            plt.ylim(0, 1 / 1000)
            self.plot_vline("最大層間変形角", "損傷限界")
            self.plot_xlim_and_ylim(xmax, 1 / 1000)
            plt.xticks(np.linspace(0, 0.005, 9), [f"{1 / x:.0f}" for x in np.linspace(0, 0.005, 9)])
            plt.yticks(np.linspace(0, 0.001, 9), [f"{1 / x:.0f}" for x in np.linspace(0, 0.001, 9)])
            plt.xlabel("最大層間変形角")
            plt.ylabel("残留層間変形角")
        elif key == "想定最大応答変形角":
            xmax: float = 1 / 50
            plt.plot(self.frame("想定最大応答変位") / self.story("意匠階高"), self.story("最大応答変形角")[self.step("主架構降伏")], "-o")
            plt.plot([0, xmax], [0, xmax], "--", color="black", lw=1)
            plt.hlines(y=self.dmax, xmin=0, xmax=0.02, color="black", lw=1)
            plt.vlines(x=self.dmax, ymin=0, ymax=0.02, color="black", lw=1)
            self.plot_xlim_and_ylim(xmax, xmax)
            plt.xticks(np.linspace(0, 0.02, 9), [f"{1 / x:.0f}" for x in np.linspace(0, 0.02, 9)])
            plt.yticks(np.linspace(0, 0.02, 9), [f"{1 / x:.0f}" for x in np.linspace(0, 0.02, 9)])
            plt.xlabel("想定最大応答変形角")
            plt.ylabel("最大応答変形角")
            plt.title(f'{min(self.story("意匠階高") / self.frame("想定最大応答変位")):.2f} - {min(1 / self.story("最大応答変形角")[self.step("主架構降伏")]):.2f}')
        elif key == "最大応答変形角":
            xmax: float = 1 / 50
            plt.plot(self.building("最大層間変形角"), self.story("最大応答変形角"))
            plt.hlines(y=self.dmax, xmin=0, xmax=0.02, color="black", lw=1)
            self.plot_xlim_and_ylim(xmax, xmax)
            plt.xticks(np.linspace(0, 0.02, 9), [f"{1 / x:.0f}" for x in np.linspace(0, 0.02, 9)])
            plt.yticks(np.linspace(0, 0.02, 9), [f"{1 / x:.0f}" for x in np.linspace(0, 0.02, 9)])
            self.plot_vline("最大層間変形角", "主架構降伏")
            plt.xlabel("最大層間変形角")
            plt.ylabel("最大応答変形角")
        elif key == "主架構必要累積塑性変形倍率":
            plt.title("主架構")
            plt.plot(self.building("ベースシア係数"), self.frame("必要累積塑性変形倍率"))
            xmax: float = 0.3
            plt.hlines(y=self.frame("保有累積塑性変形倍率"), xmin=0, xmax=xmax, colors=tool.colors()[0:self.number_of_stories()], lw=1, linestyle="dashed")
            self.plot_vline("ベースシア係数", "主架構降伏")
            self.plot_xlim_and_ylim(xmax=xmax)
            plt.ylabel("必要累積塑性変形倍率")
            plt.xlabel("ベースシア係数")
        elif key == "ダンパー必要累積塑性変形倍率":
            plt.title("ダンパー")
            plt.plot(self.LAMBDA, self.damper("保有累積塑性変形倍率"), "--", lw=0.5)
            plt.hlines(y=self.damper("必要累積塑性変形倍率")[self.step("主架構降伏")], xmin=0, xmax=1, color=tool.colors()[0:self.number_of_stories()])
            plt.xlabel("LAMBDA")
            plt.yscale("log")
            # self.plot_xlim_and_ylim(xmax=1, ymax=np.max(np.nan_to_num(self.damper("必要累積塑性変形倍率")[self.step("主架構降伏")])) * 1.1)
            self.plot_xlim_and_ylim(xmax=1)
        elif key == "最適ブレース分担率":
            kf: np.ndarray = np.array([self.frame("初期剛性")])
            dy: np.ndarray = np.array([self.damper("降伏変位")])
            fy: np.ndarray = np.array([self.damper("降伏耐力")])
            amp: np.ndarray = np.array([np.linspace(0, 3, 100)]).T
            q: np.ndarray = fy * amp
            we: np.ndarray = np.array([self.story("損傷限界時吸収エネルギー")[self.step("稀地震")]])
            ed: np.ndarray = max(self.inputted_energy("稀地震")) * we / np.sum(we)
            kf_delta: np.ndarray = np.sqrt(
                16 * q**2 + 7 * kf * dy * q + 2 * kf * ed
            ) - 4 * q
            delta: np.ndarray = kf_delta / kf
            m: np.ndarray = self.story("地震用質量")
            periods: np.ndarray = np.array([[Eigen_Analysis(np.flip(m), np.flip(k) * 1000).period() for k in kf + q / delta]]).T
            rt: np.ndarray = self.soil.rt(periods)
            story_ci: np.ndarray = (q + kf_delta) / np.cumsum(m) / self.G
            # story_c0: np.ndarray = story_ci / rt
            frame_c0: np.ndarray = kf_delta / np.cumsum(m) / self.G
            frame_ci: np.ndarray = rt * frame_c0 * np.array([self.ai_factor()])
            plt.yscale("log")
            # amp = periods
            # plt.plot(np.linspace(0, 3, 100), self.rt(np.linspace(0, 3, 100)), color="black")
            plt.plot(amp, frame_ci, label=self.stories())
            plt.hlines(y=np.min(frame_ci, axis=0), xmin=0, xmax=3, colors=tool.colors()[0:self.number_of_stories()], lw=1)
            plt.plot(amp, story_ci[:, -1], label="ベースシア係数", color="black")
            plt.hlines(y=np.min(story_ci[:, -1]), xmin=0, xmax=3, colors="black", lw=1)
            plt.grid()
            plt.legend()
            plt.title("予測最適ブレース耐力")
            plt.xlabel("ブレース耐力の現状に対する比")
            plt.ylabel("主架構せん断力係数")
        elif key == "最適ブレース分担率_Rt":
            kf: np.ndarray = np.array([self.frame("初期剛性")])
            dy: np.ndarray = np.array([self.damper("降伏変位")])
            fy: np.ndarray = np.array([self.damper("降伏耐力")])
            amp: np.ndarray = np.array([np.linspace(0, 3, 100)]).T
            q: np.ndarray = fy * amp
            we: np.ndarray = np.array([self.story("損傷限界時吸収エネルギー")[self.step("稀地震")]])
            ed: np.ndarray = max(self.inputted_energy("稀地震")) * we / np.sum(we)
            kf_delta: np.ndarray = np.sqrt(
                16 * q**2 + 7 * kf * dy * q + 2 * kf * ed
            ) - 4 * q
            delta: np.ndarray = kf_delta / kf
            m: np.ndarray = self.story("地震用質量")
            periods: np.ndarray = np.array([[Eigen_Analysis(np.flip(m), np.flip(k) * 1000).period() for k in kf + q / delta]]).T
            rt: np.ndarray = self.soil.rt(periods)
            story_ci: np.ndarray = (q + kf_delta) / np.cumsum(m) / self.G
            story_c0: np.ndarray = story_ci / rt
            frame_c0: np.ndarray = kf_delta / np.cumsum(m) / self.G
            frame_ci: np.ndarray = rt * frame_c0 * np.array([self.ai_factor()])
            plt.yscale("log")
            for i in range(self.number_of_stories()):
                plt.plot(amp, frame_ci[:, i], color=tool.colors(i), label=self.stories()[i])
                plt.plot(amp, frame_c0[:, i], color=tool.colors(i), linestyle="dashed")
            plt.hlines(y=np.min(frame_ci, axis=0), xmin=0, xmax=3, colors=tool.colors()[0:self.number_of_stories()], lw=1)
            plt.plot(amp, story_c0[:, -1], label="ベースシア係数", color="black")
            plt.hlines(y=np.min(story_c0[:, -1]), xmin=0, xmax=3, colors="black", lw=1)
            plt.grid()
            plt.legend()
            plt.title("予測最適ブレース耐力")
            plt.xlabel("ブレース耐力の現状に対する比")
            plt.ylabel("主架構せん断力係数")
        elif key == "保有水平耐力分布":
            mi: np.ndarray = self.story("地震用質量")
            ai: np.ndarray = np.array([self.story("Ai分布")[self.step("主架構降伏")]])
            qdi_old: np.ndarray = self.damper("保有水平耐力")
            qd0: np.ndarray = np.array([np.linspace(0, 3, 100) * qdi_old[-1]]).T
            qfi: np.ndarray = np.array([self.frame("保有水平耐力")])
            qf0: float = qfi[0, -1]
            qdi_new: np.ndarray = ai * np.array([np.cumsum(mi)]) / np.sum(mi) * (qd0 + qf0) - qfi
            plt.plot(qd0 / qdi_old[-1], qdi_new / qdi_old)
            xmin: float = 0
            xmax: float = 3
            ymin: float = 0
            ymax: float = 3
            plt.vlines(x=1, ymin=ymin, ymax=ymax, color="black")
            plt.hlines(y=1, xmin=xmin, xmax=xmax, color="black")
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.title("全層でAi分布と保有水平耐力分布が等しくなるようなブレース耐力")
            plt.xlabel("1階ブレース耐力の現状に対する比")
            plt.ylabel("各階ブレース耐力の現状に対する比")
            plt.grid()
        elif key == "ダンパー有効性":
            k: np.ndarray = self.damper("初期剛性") / self.frame("初期剛性")
            beta: np.ndarray = self.damper("保有水平耐力") / self.story("保有水平耐力")
            n: int = self.number_of_stories()
            y_story: list[int] = [n - i for i in range(n)]
            plt.plot(
                beta,
                y_story,
                "-o",
                label="耐力分担率",
            )
            plt.plot(
                k / (1 + k),
                y_story,
                "--o",
                label="履歴ダンパ成立上限",
            )
            plt.plot(
                np.ones(self.number_of_stories()) * 0.1 * (beta > 0),
                y_story,
                "--o",
                label="履歴減衰効果下限",
            )
            plt.plot(
                1 - 1 / np.sqrt(1 + k),
                y_story,
                "--p",
                label="履歴減衰効果上限",
            )
            plt.plot(
                k / (1 + k + 2 * np.sqrt(1 + k)),
                y_story,
                "-o",
                label="適正値",
            )
            plt.xlim(0, 1)
            plt.yticks(y_story, y_story)
            plt.legend()
            plt.xlabel(r"耐力分担率$\beta$")
            plt.ylabel("層")
            plt.grid()
        elif key == "Gs":
            td: np.ndarray = np.linspace(1e-4, 5, 1000) if self.t_sample is None else self.t_sample
            for i in range(3, 0, -1):
                s: Soil = Soil(i, 1.0)
                plt.plot(td, s.gs(td), label=s.soil_str())
            plt.plot(td, self.gs(td), color="black")
            td: float = self.period()[self.step("損傷限界")]
            ts_td: float = 1.2 if np.all(self.brace_shear == 0) or np.all(self.brace_is_damper()) else 1.4
            plt.axvspan(
                td,
                ts_td * td,
                color="coral",
                alpha=0.5,
            )
            plt.xlim(0, 5)
            plt.ylim(0, 3)
            plt.grid()
            plt.ylabel(r"$G_s$")
            plt.legend()
        elif key == "ステップ-接線剛性":
            for i in range(self.number_of_stories()):
                plt.plot(self.frame("接線剛性")[:, i] / self.frame("接線剛性")[10, i], color=tool.colors(i), label=self.stories()[i])
                plt.plot(self.damper("接線剛性")[:, i] / self.damper("接線剛性")[10, i], "--", color=tool.colors(i))
            plt.xlabel("ステップ")
            plt.ylabel("接線剛性/初期剛性")
            plt.grid()
            plt.ylim(0, 1.5)
        elif key == "層間変形角-接線剛性":
            for i in range(self.number_of_stories()):
                plt.plot(self.story("層間変形角")[:, i], self.frame("接線剛性")[:, i] / self.frame("接線剛性")[10, i], color=tool.colors(i), label=self.stories()[i])
                plt.plot(self.story("層間変形角")[:, i], self.damper("接線剛性")[:, i] / self.damper("接線剛性")[10, i], "--", color=tool.colors(i))
            plt.xlabel("層間変形角")
            plt.ylabel("接線剛性/初期剛性")
            plt.grid()
            plt.xticks(np.arange(0, (np.ceil(self.dmax / 0.001) + 1) * 0.001, 0.001), rotation=90)
            plt.xlim(0, self.dmax)
        elif key == "ステップ-分担割合":
            for i in range(self.number_of_stories()):
                plt.plot(self.frame("せん断力")[:, i] / self.story("せん断力")[:, i] / self.frame("せん断力")[5, i] * self.story("せん断力")[5, i], color=tool.colors(i), label=self.stories()[i])
                plt.plot(self.damper("せん断力")[:, i] / self.story("せん断力")[:, i] / self.damper("せん断力")[5, i] * self.story("せん断力")[5, i], "--", color=tool.colors(i))
            plt.xlabel("ステップ")
            plt.ylabel("せん断力分担割合/初期分担割合")
            plt.grid()
        elif key == "ステップ-せん断力":
            for i in range(self.number_of_stories()):
                plt.plot(self.frame("せん断力")[:, i], color=tool.colors(i), label=self.stories()[i])
                plt.plot(self.damper("せん断力")[:, i], "--", color=tool.colors(i))
            plt.xlabel("ステップ")
            plt.ylabel("柱・ブレースせん断力")
            plt.grid()
        elif key == "層間変形角-せん断力":
            for i in range(self.number_of_stories()):
                plt.plot(self.story("層間変形角")[:, i], self.frame("せん断力")[:, i] / self.design_shear()[i], color=tool.colors(i), label=f"{self.stories()[i]}柱")
                plt.plot(self.story("層間変形角")[:, i], self.damper("せん断力")[:, i] / self.design_shear()[i], color=tool.colors(i), label=f"{self.stories()[i]}ブレース")
                plt.plot(self.story("層間変形角")[:, i], self.story("せん断力")[:, i] / self.design_shear()[i], color=tool.colors(i), label=f"{self.stories()[i]}柱+ブレース")
                plt.plot(
                    [
                        0,
                        self.frame("降伏変位")[i] / self.story("構造階高")[i],
                        self.story("層間変形角")[*self.step("想定最大応答変位")][i],
                    ],
                    [
                        0,
                        self.frame("降伏耐力")[i] / self.design_shear()[i],
                        self.frame("降伏耐力")[i] / self.design_shear()[i],
                    ],
                    "--",
                    color=tool.colors(i),
                )
            plt.xticks(np.arange(0, self.dmax, 0.0025), [f"{1 / x:.0f}" for x in np.arange(0, self.dmax, 0.0025)])
            plt.xlim(0, self.dmax)
            plt.xlabel("層間変形角")
            plt.ylabel("柱・ブレースせん断力")
            plt.grid()
        elif key == "ステップ-層せん断力":
            for i in range(self.number_of_stories()):
                plt.plot(self.frame("せん断力")[:, i] + self.damper("せん断力")[:, i], color=tool.colors(i), label=self.stories()[i])
            plt.xlabel("ステップ")
            plt.ylabel("層せん断力")
            plt.grid()
        elif key == "層間変形角-層せん断力":
            for i in range(self.number_of_stories()):
                plt.plot(self.story("層間変形角")[:, i], self.frame("せん断力")[:, i] + self.damper("せん断力")[:, i], color=tool.colors(i), label=self.stories()[i])
            plt.xticks(np.arange(0, (np.ceil(self.dmax / 0.001) + 1) * 0.001, 0.001), rotation=90)
            plt.xlim(0, self.dmax)
            plt.xlabel("層間変形角")
            plt.ylabel("層せん断力")
            plt.grid()
        elif key == "ステップ-層間変形角":
            for i in range(self.number_of_stories()):
                plt.plot(self.story("層間変形角")[:, i], color=tool.colors(i), label=self.stories()[i])
            plt.xlabel("ステップ")
            plt.ylabel("層間変形角")
            plt.grid()
        elif key == "バイリニア置換確認":
            plt.plot(
                [0, 1.1],
                [0, 1.1],
                "--",
                color="black",
                lw=1,
            )
            x: np.ndarray = self.frame("せん断力")[self.step("保有水平耐力")]
            y: np.ndarray = self.frame("保有水平耐力")
            plt.plot(
                x / max(x),
                y / max(y),
                "-o",
            )
            plt.xlim(0, 1.1)
            plt.ylim(0, 1.1)
            plt.ylabel("バイリニア置換後")
            plt.xlabel("最大層間変形角=1/100時せん断力")
            plt.grid()

    def estimated_c0(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kf: np.ndarray = np.array([self.frame("初期剛性")])
        dy: np.ndarray = np.array([self.damper("降伏変位")])
        fy: np.ndarray = np.array([self.damper("降伏耐力")])
        amp: np.ndarray = np.array([np.linspace(0, 3, 100)]).T
        q: np.ndarray = fy * amp
        we: np.ndarray = np.array([self.story("損傷限界時吸収エネルギー")[self.step("稀地震")]])
        ed: np.ndarray = max(self.inputted_energy("稀地震")) * we / np.sum(we)
        kf_delta: np.ndarray = np.sqrt(
            16 * q**2 + 7 * kf * dy * q + 2 * kf * ed
        ) - 4 * q
        delta: np.ndarray = kf_delta / kf
        m: np.ndarray = self.story("地震用質量")
        periods: np.ndarray = np.array([[Eigen_Analysis(np.flip(m), np.flip(k) * 1000).period() for k in kf + q / delta]]).T
        rt: np.ndarray = self.soil.rt(periods)
        story_ci: np.ndarray = (q + kf_delta) / np.cumsum(m) / self.G
        story_c0: np.ndarray = story_ci / rt
        frame_c0: np.ndarray = kf_delta / np.cumsum(m) / self.G
        frame_ci: np.ndarray = rt * frame_c0 * np.array([self.ai_factor()])
        return (
            periods,
            story_c0[:, -1],
            frame_ci,
        )

    def s_column_test_ratio(self, story: str) -> list[float]:
        return [np.max(np.array([x[key] for key in [
            "M柱頭",
            "M中央",
            "M柱脚",
            "Q柱頭",
            "Q柱脚",
        ]], dtype=np.float64)) for x in filter(lambda x: x["階"] == story, self.get("S柱検定比一覧"))]

    def brace_test_ratio(self) -> list[float]:
        return [np.max(np.array([x[key] for key in [
            "N左下り",
            "N右下り",
        ]], dtype=np.float64)) for x in self.get("ブレース検定比一覧")]

    def s_beam_test_ratio(self, floor: str) -> list[float]:
        return [np.max(np.array([x[key] for key in [
            "M仕口左",
            "M左端",
            "MJoint左",
            "M中央",
            "MJoint右",
            "M右端",
            "M仕口右",
            "Q仕口左",
            "Q左端",
            "QJoint左",
            "QJoint右",
            "Q右端",
            "Q仕口右",
            "組合せJoint左",
            "組合せJoint右",
        ]], dtype=np.float64)) for x in filter(lambda x: x["層"] == floor, self.get("S梁検定比一覧"))]

    def plot_test_ratio(self, original_file: str) -> None:
        original: "Energy_Method" = Energy_Method(
            original_file,
            self.load,
        )
        plt.figure(figsize=(10, 5), tight_layout=True)
        amin: float = 0
        amax: float = 1.4
        plt.subplot(1, 2, 1)
        plt.title("S梁")
        for i, floor in enumerate(self.floors(True)):
            plt.plot(
                original.s_beam_test_ratio(floor),
                self.s_beam_test_ratio(floor),
                "o",
                color=tool.colors(i),
                label=floor,
            )
        plt.plot([amin, amax], [amin, amax], color="black", lw=1)
        plt.xlim(amin, amax)
        plt.ylim(amin, amax)
        plt.grid()
        plt.xlabel("元設計")
        plt.ylabel("エネルギー法")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title("S柱")
        for i, story in enumerate(self.stories(True)):
            plt.plot(
                original.s_column_test_ratio(story),
                self.s_column_test_ratio(story),
                "o",
                color=tool.colors(i),
                label=story,
            )
        plt.plot([amin, amax], [amin, amax], color="black", lw=1)
        plt.xlim(amin, amax)
        plt.ylim(amin, amax)
        plt.grid()
        plt.legend()
        plt.xlabel("元設計")

    def input_for_excel(self, filename: str) -> None:
        story_mass: np.ndarray = np.zeros(self.number_of_stories())
        dummy_story: list[dict[str, str]] = self.get("構造階高")
        i: int = 0
        for j, row in enumerate(self.get("地震用重量")):
            if dummy_story[j]["従属層"] == "上層":
                i -= 1
            if i > self.number_of_stories() - 1:
                break
            story_mass[i] += float(row["wi(wi/A)kN"])
            i += 1
            if dummy_story[j]["従属層"] == "下層":
                i -= 1

        re: np.ndarray = self.eccentricity()
        pti: np.ndarray = np.sum([
            1 * (re <= 0.15),
            (1.15 - re) * (0.15 < re) * (re < 0.3),
            0.85 * (0.3 <= re),
        ], axis=0)
        lbr: np.ndarray = self.damper("部材長")
        lh: np.ndarray = self.story("意匠階高")

        dictionary: dict[str, int | np.ndarray] = {
            "入力/Wi": story_mass,
            "入力/損傷限界時層せん断力": self.design_shear(),
            "入力/hi": self.story("意匠階高"),
            "入力/pti": pti,
            "損傷用準備計算/稀地震時": self.step("稀地震") + 1,
            "損傷用準備計算/ダンパーの有無": self.brace_is_damper() * 1,
            "安全用準備計算/エネルギー等価になる変形角": np.round(
                1 / self.story("層間変形角")[
                    np.argmax(self.frame("せん断力") > self.frame("降伏耐力"), axis=0) - 1,
                    [i for i in range(self.number_of_stories())]
                ]
            ),
            "◎3.Gs/Ts": self.period("安全限界"),
            "◎4.損傷限界/地域係数": self.soil.z,
            "◎5.安全限界/部材種別": self.frame("部材種別"),
            "◎5.安全限界/崩壊モード": self.frame("崩壊モード"),
            "◎5.安全限界/保有累積塑性変形倍率": self.frame("保有累積塑性変形倍率"),
            "◎5.安全限界/Ld(mm)": np.sqrt(lbr ** 2 - lh ** 2),
            "◎5.安全限界/λLbr(mm)": 0.3 * self.damper("部材長"),
            "◎5.安全限界/Abr(mm2)": self.damper("断面積"),
            "◎5.安全限界/架構数nd": self.damper("個数"),
        }
        with open(filename, "w", encoding="utf_8_sig", newline="") as fp:
            writer: csv.DictWriter = csv.DictWriter(
                fp,
                dictionary.keys(),
            )
            writer.writeheader()
            writer.writerows([
                {
                    key: (dictionary[key][i] if hasattr(dictionary[key], "__iter__") else dictionary[key]) for key in dictionary if i == 0 or hasattr(dictionary[key], "__iter__")
                } for i in range(self.number_of_stories())
            ])
