import numpy as np
import matplotlib.pyplot as plt
from . import tool


class Soil:
    """地盤種別と地域係数Zから、低減係数Gsとエネルギー換算速度Vdを返す"""
    soil_int: int
    z: float

    def __init__(self, soil_int: int, z: float) -> None:
        """
        Args:
            soil_int: 地盤種別[1, 2, 3]
            z: 地域係数Z
        """
        self.soil_int = soil_int
        self.z = z

    def soil_str(self) -> str:
        return f'第{["零", "一", "二", "三"][self.soil_int]}種地盤'

    def vd_reduction(self, td: np.ndarray) -> np.ndarray:
        ta: float = (0, 0.576, 0.846, 1.152)[self.soil_int]
        tb: float = (0, 0.640, 0.960, 1.280)[self.soil_int]
        return np.sum([
            (1.0 - 0.10 / 0.16 * td) * (td < 0.16),
            0.90 * (0.16 <= td) * (td < ta),
            (0.90 + 0.10 * (td - ta) / (tb - ta)) * (ta <= td) * (td < tb),
            1.00 * (tb <= td),
        ], axis=0)

    def gs(self, td: np.ndarray) -> np.ndarray:
        """低減係数Gs

        Args:
            td: Gsを求める固有周期
        """
        if self.soil_int == 1:
            return np.sum([
                1.5 * (td < 0.576),
                0.864 / td * (0.576 <= td) * (td < 0.64),
                1.35 * (0.64 <= td),
            ], axis=0)
        else:
            gv: float = (0, 0, 2.025, 2.7)[self.soil_int]
            tu: float = 0.64 * gv / 1.5
            return np.sum([
                1.5 * (td < 0.64),
                1.5 * td / 0.64 * (0.64 <= td) * (td < tu),
                gv * (tu <= td),
            ], axis=0)

    def vd(self, td: np.ndarray, gs: np.ndarray = None) -> np.ndarray:
        """エネルギー換算速度Vd

        Args:
            td: Vdを求める固有周期
            gs: Vdを求めるときに使うGs NoneでデフォルトのGs
        """
        if gs is None:
            gs = self.gs(td)
        return np.sum([
            td * (0.64 + 6 * td) * (td < 0.16),
            td * 1.6 * (0.16 <= td) * (td < 0.64),
            1.024 * (0.64 <= td),
        ], axis=0) / 2 / np.pi * self.z * gs * self.vd_reduction(td)

    def vs(self, td: np.ndarray, gs: np.ndarray = None) -> np.ndarray:
        return 5 * self.vd(td, gs)

    def rt(self, td: np.ndarray) -> np.ndarray:
        tc: float = (
            0.0,
            0.4,
            0.6,
            0.8
        )[self.soil_int]
        t1: float = (
            0.0,
            0.8,
            1.2,
            1.6,
        )[self.soil_int]
        v1: float = (
            0.0,
            0.64,
            0.96,
            1.28,
        )[self.soil_int]
        return np.sum([
            1 * (td < tc),
            (1 - 0.2 * (td / tc - 1)**2) * (tc <= td) * (td < t1),
            v1 / td * (t1 < td),
        ], axis=0)


def plot_soil(filename: str = None) -> None:
    """0~5秒でGsとVdを例示する
    Args:
        filename: 保存するファイル名 Noneの時保存しない
    """
    td: np.ndarray = np.linspace(1e-4, 5, 1000)
    plt.figure(figsize=(3, 6))
    plt.subplot(2, 1, 1)
    for i in range(3, 0, -1):
        s: Soil = Soil(i, 1.0)
        plt.plot(td, s.gs(td), label=s.soil_str())
    plt.xlim(0, 5)
    plt.ylim(0, 3)
    plt.grid()
    plt.ylabel(r"$G_s$")
    plt.legend()

    plt.subplot(2, 1, 2)
    for i in range(3, 0, -1):
        s: Soil = Soil(i, 1.0)
        plt.plot(td, s.vd(td), label=s.soil_str())
    plt.xlim(0, 5)
    plt.ylim(0, 0.5)
    plt.grid()
    plt.xlabel(r"$T_d$")
    plt.ylabel(r"$V_d$")
    if filename is not None:
        tool.savefig(filename)


if __name__ == "__main__":
    plot_soil("./soil.svg")
