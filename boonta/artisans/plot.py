import numpy as np

from .artifact import Image


class Plot:
    """Plots a statistic of finished episodes, the episode return by default.

    Each time it runs, it takes the mean and standard deviation of the finished
    episodes in the logs it is given and adds them to its history. It returns
    the whole history as an image, so the first point is the first run.
    """

    def __init__(
        self,
        name: str = "episode_return",
        key: str = "episode_statistics/episode_return",
        xlabel: str = "Epoch",
        ylabel: str = "Episode return",
        size: tuple[float, float] = (6.0, 4.0),
        dpi: int = 100,
    ):
        self.name = name
        self.key = key
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.size = tuple(size)
        self.dpi = dpi
        self.means = []
        self.stds = []

    def craft(self, key, algorithm, environment, state, logs) -> Image:
        # matplotlib is imported here so that the library does not need it.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        returns = np.asarray(logs[self.key], dtype=np.float64)
        returns = returns[np.isfinite(returns)]
        self.means.append(returns.mean() if returns.size else np.nan)
        self.stds.append(returns.std() if returns.size else np.nan)

        mean, std = np.array(self.means), np.array(self.stds)
        figure = Figure(figsize=self.size, dpi=self.dpi)
        canvas = FigureCanvasAgg(figure)
        axes = figure.subplots()
        x = np.arange(len(mean))
        axes.plot(x, mean)
        axes.fill_between(x, mean - std, mean + std, alpha=0.25)
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.grid(alpha=0.3)
        figure.tight_layout()
        canvas.draw()
        pixels = np.asarray(canvas.buffer_rgba())[..., :3].copy()
        return Image(self.name, pixels)
