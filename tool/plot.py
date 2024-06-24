import matplotlib_fontja # NOQA
import matplotlib.pyplot as plt


def colors(idx: int = None) -> str:
    colors: list[str] = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    if idx is None:
        return colors
    else:
        return colors[idx % len(colors)]


def savefig(figname: str) -> None:
    """matplotlibで画像を保存する
    """
    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()
    plt.close()
