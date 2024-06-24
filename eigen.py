from typing import Iterable
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from . import tool


class Eigen_Analysis:
    """各層の質量と剛性から固有値解析を行う
    """
    mass_list: np.ndarray
    stiffness_list: np.ndarray
    eig_val: np.ndarray
    eig_vec: np.ndarray

    def __init__(self, mass: Iterable[float], stiffness: Iterable[float]) -> None:
        """
        Args:
            mass: 下層から順に並んだ層質量
            stiffness: 下層から順に並んだ層剛性
        """
        self.mass_list = np.array(mass)
        self.stiffness_list = np.array(stiffness)
        self.eig_val, self.eig_vec = scipy.linalg.eig(self.stiffness_matrix(), self.mass_matrix())
        self.eig_val = np.abs(self.eig_val)

    def mass_matrix(self) -> np.ndarray:
        return np.diag(self.mass_list)

    def stiffness_matrix(self) -> np.ndarray:
        matrix: np.ndarray = np.diag(self.stiffness_list)
        for i, v in enumerate(self.stiffness_list):
            if i > 0:
                matrix[i - 0, i - 1] += -v
                matrix[i - 1, i - 0] += -v
                matrix[i - 1, i - 1] += v
        return matrix

    def participation_factor(self) -> np.ndarray:
        return (self.mass_list @ self.eig_vec) / (self.mass_list @ self.eig_vec**2)

    def periods(self) -> np.ndarray:
        """n次固有周期 順不同
        """
        return 2 * np.pi / np.sqrt(self.eig_val)

    def period(self) -> float:
        """一次固有周期
        """
        return max(self.periods())

    def participated_vector(self) -> np.ndarray:
        """刺激係数"""
        return np.concatenate([
            np.zeros((1, len(self.eig_val))),
            self.participation_factor() * self.eig_vec,
        ])

    def story(self) -> np.ndarray:
        return np.arange(0, len(self.mass_list) + 1, 1)

    def plot_vector(self, filename: str = None) -> None:
        """固有ベクトルを描く

        Args:
            filename: 保存するファイル名 Noneの時は書き込むだけ
        """
        plt.figure(figsize=(3, 3))
        plt.plot(
            self.participated_vector(),
            self.story(),
            "o-",
            label=[f"{t:.3f}s" for t in self.periods()],
        )
        xmax: float = plt.xlim()[1]
        plt.xlim(-xmax, xmax)
        plt.yticks(self.story())
        plt.ylim(0, max(self.story()))
        plt.grid()
        plt.title("固有モード")
        if filename is not None:
            tool.savefig(filename)


if __name__ == "__main__":
    Eigen_Analysis(
        np.array([192537, 154998, 134986, 18165]) / 9.80655,
        np.array([3073.6, 2423.5, 1750.1, 1368.6]) * 1000,
    ).plot_vector()
    plt.show()
