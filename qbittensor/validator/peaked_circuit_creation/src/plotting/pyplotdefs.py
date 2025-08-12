"""
Handy matplotlib.pyplot settings.
"""
from __future__ import annotations
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import matplotlib.image as img
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.colors as cm
import matplotlib.font_manager as fm
from .colordefs import (
    color_cycles,
    color_scales,
    opacity,
    dot_dash,
    Slicer,
    S,
)
from cycler import cycler
import copy

rcdefs = {
    "axes.grid"             : True,
    "axes.grid.which"       : "both",
    "axes.linewidth"        : 0.65,
    "axes.prop_cycle"       : cycler(color=color_cycles["whooie"]),
    "axes.titlesize"        : "medium",
    "errorbar.capsize"      : 1.25,
    "figure.dpi"            : 500.0,
    "figure.figsize"        : [3.375, 2.225],
    "figure.labelsize"      : "medium",
    "font.size"             : 8.0,
    "grid.color"            : "#d8d8d8",
    "grid.linewidth"        : 0.5,
    "hatch.linewidth"       : 0.65,
    "image.cmap"            : "jet",
    "image.composite_image" : False,
    "legend.borderaxespad"  : 0.25,
    "legend.borderpad"      : 0.2,
    "legend.fancybox"       : False,
    "legend.fontsize"       : "x-small",
    "legend.framealpha"     : 0.8,
    "legend.handlelength"   : 1.2,
    "legend.handletextpad"  : 0.4,
    "legend.labelspacing"   : 0.25,
    "lines.linewidth"       : 0.8,
    "lines.markeredgewidth" : 0.8,
    "lines.markerfacecolor" : "white",
    "lines.markersize"      : 2.0,
    "markers.fillstyle"     : "full",
    "savefig.bbox"          : "tight",
    "savefig.pad_inches"    : 0.05,
    "text.latex.preamble"   : r"\usepackage{physics}\usepackage{siunitx}\usepackage{amsmath}",
    "xtick.direction"       : "in",
    "xtick.major.size"      : 2.0,
    "xtick.minor.size"      : 1.5,
    "ytick.direction"       : "in",
    "ytick.major.size"      : 2.0,
    "ytick.minor.size"      : 1.5,
}
for key in rcdefs:
    pp.rcParams[key] = rcdefs[key]

hot_cold \
    = cm.LinearSegmentedColormap.from_list("hot-cold", color_scales["hot-cold"])
fire_ice \
    = cm.LinearSegmentedColormap.from_list("fire-ice", color_scales["fire-ice"])
powerade \
    = cm.LinearSegmentedColormap.from_list("powerade", color_scales["powerade"])
floral \
    = cm.LinearSegmentedColormap.from_list("floral", color_scales["floral"])
blue_hot \
    = cm.LinearSegmentedColormap.from_list("blue-hot", color_scales["blue-hot"])
cyborg \
    = cm.LinearSegmentedColormap.from_list("cyborg", color_scales["cyborg"])
sport \
    = cm.LinearSegmentedColormap.from_list("sport", color_scales["sport"])
vibrant \
    = cm.LinearSegmentedColormap.from_list("vibrant", color_scales["vibrant"])
artsy \
    = cm.LinearSegmentedColormap.from_list("artsy", color_scales["artsy"])
pix \
    = cm.LinearSegmentedColormap.from_list("pix", color_scales["pix"])
sunset \
    = cm.LinearSegmentedColormap.from_list("sunset", color_scales["sunset"])

colormaps = {
    "hot-cold"  : hot_cold,
    "fire-ice"  : fire_ice,
    "powerade"  : powerade,
    "floral"    : floral,
    "blue-hot"  : blue_hot,
    "cyborg"    : cyborg,
    "sport"     : sport,
    "vibrant"   : vibrant,
    "artsy"     : artsy,
    "pix"       : pix,
    "sunset"    : sunset,
}
for cmap in colormaps.values():
    mpl.colormaps.register(cmap)

def figure3D(*fig_args, **fig_kwargs):
    fig = pp.figure(*fig_args, **fig_kwargs)
    ax = p3.Axes3D(fig)
    return fig, ax

def set_font(path, name):
    fe = fm.FontEntry(fname=path, name=name)
    fm.fontManager.ttflist.insert(0, fe)
    pp.rcParams["font.family"] = fe.name

def set_fontsize(s: int | str):
    pp.rcParams["font.size"] = s

def use_tex(u=True):
    pp.rcParams["text.usetex"] = u
    if u:
        pp.rcParams["font.serif"] = ["Computer Modern Roman"]
        pp.rcParams["font.sans-serif"] = ["Computer Modern Sans Serif"]
        pp.rcParams["font.family"] = ["serif"]
    else:
        pp.rcParams["font.serif"] = ["DejaVu Serif"]
        pp.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        pp.rcParams["font.family"] = ["sans-serif"]

def grid(onoff=True, axes=None):
    if axes:
        axes.minorticks_on()
        if onoff:
            axes.grid(onoff, "major", color="#d8d8d8", zorder=-10)
            axes.grid(onoff, "minor", color="#e0e0e0", linestyle=":", zorder=-10)
            axes.tick_params(which="both", direction="in")
        else:
            axes.grid(onoff)
            axes.grid(onoff)
    else:
        pp.minorticks_on()
        if onoff:
            pp.grid(onoff, "major", color="#d8d8d8", zorder=-10)
            pp.grid(onoff, "minor", color="#e0e0e0", linestyle=":", zorder=-10)
            pp.tick_params(which="both", direction="in")
        else:
            pp.grid(onoff)
            pp.grid(onoff)

def set_color_cycle(c):
    if c in color_cycles.keys():
        pp.rcParams["axes.prop_cycle"] = cycler(color=color_cycles[c])
    else:
        print(f"plotdefs.set_color_cycle: cycle name '{c}' undefined. Colors were not modified.")

def show():
    pp.show()

class Plots:
    def __init__(self, plots: list[Plotter], nrows: int=1, ncols: int=1):
        assert all(p.fig is plots[0].fig for p in plots)
        self.fig = plots[0].fig
        self.plots = plots
        self.nrows = nrows
        self.ncols = ncols

    def __getitem__(self, pos):
        return self.plots[pos]

    def tight_layout(self, *args, **kwargs):
        X = self.fig.tight_layout(*args, **kwargs)
        for p in self.plots:
            p.outputs.append(X)
        return self

    def savefig(self, *args, **kwargs):
        X = self.fig.savefig(*args, **kwargs)
        for p in self.plots:
            p.outputs.append(X)
        return self

    def show(self):
        pp.show()
        return self

    def close(self):
        pp.close(self.fig)

    def f(self, f, *args, **kwargs):
        X = f(*args, **kwargs)
        for p in self.plots:
            p.outputs.append(X)
        return self

    def to_plotarray(self) -> PlotArray:
        return PlotArray(self.plots, self.nrows, self.ncols)

class PlotArray:
    def __init__(self, plots: list[Plotter], nrows: int=1, ncols: int=1):
        assert all(p.fig is plots[0].fig for p in plots)
        self.fig = plots[0].fig
        self.plots = plots
        self.nrows = nrows
        self.ncols = ncols
        self._selector = (0, 0)
        self._method = None

    def _verify_loc(self, loc: int | (int, int)) -> (int, int):
        if loc is None:
            return self._selector
        if not (
            (
                isinstance(loc, tuple)
                and all(isinstance(x, int) for x in loc)
                and len(loc) == 2
                and loc[0] in range(self.nrows)
                and loc[1] in range(self.ncols)
            )
            or (
                isinstance(loc, int)
                and loc < self.ncols * self.nrows
            )
        ):
            raise ValueError(
                "Index or indices must be within the ranges of the numbers of"
                " rows and columns"
            )
        if isinstance(loc, int):
            _selector = (loc // self.ncols, loc % self.ncols)
        else:
            _selector = loc
        return _selector

    def _loc_idx(self, loc: int | (int, int)) -> int:
        i, j = self._verify_loc(loc)
        return self.ncols * i + j

    def __getitem__(self, loc: int | (int, int)):
        self._selector = self._verify_loc(loc)
        return self

    def do(self, f: str, *args, **kwargs):
        return getattr(self, f)(*args, **kwargs)

    def __getattr__(self, attr: str):
        if attr in dir(self):
            return getattr(self, attr)
        else:
            _attr = getattr(self.plots[self._loc_idx(self._selector)], attr)
            if isinstance(_attr, type(self.do)):
                self._method = _attr
                return self._process_method
            else:
                return _attr

    def _process_method(self, *args, **kwargs):
        if self._method is None:
            raise Exception("method is None")
        ret = self._method(*args, **kwargs)
        self._method = None
        return self if isinstance(ret, Plotter) else ret

    def tight_layout(self, *args, **kwargs):
        X = self.fig.tight_layout(*args, **kwargs)
        for p in self.plots:
            p.outputs.append(X)
        return self

    def savefig(self, *args, **kwargs):
        X = self.fig.savefig(*args, **kwargs)
        for p in self.plots:
            p.outputs.append(X)
        return self

    def show(self):
        pp.show()
        return self

    def close(self):
        pp.close(self.fig)

    def f(self, f, *args, **kwargs):
        X = f(*args, **kwargs)
        for p in self.plots:
            p.outputs.append(X)
        return self

    def to_plots(self):
        return Plots(self.plots, self.nrows, self.ncols)

class Plotter:
    def __init__(self, fig=None, ax=None, log=True, chain=True):
        if fig is None or ax is None:
            self.fig, self.ax = pp.subplots()
        else:
            self.fig = fig
            self.ax = ax
        self.outputs = list()
        self.im = None
        self.cbar = None
        self.log = log
        self.chain = chain
        self._method = None

    @staticmethod
    def new(*args, **kwargs):
        if kwargs.get("fig", None) is None \
                or kwargs.get("ax", None) is None:
            kw = kwargs.copy()
            if "log" in kw.keys():
                del kw["log"]
            if "chain" in kw.keys():
                del kw["chain"]
            if "as_plotarray" in kw.keys():
                del kw["as_plotarray"]
            fig, ax = pp.subplots(*args, **kw)
        else:
            fig, ax = kwargs["fig"], kwargs["ax"]
        log = kwargs.get("log", True)
        chain = kwargs.get("chain", True)
        as_plotarray = kwargs.get("as_plotarray", False)
        if isinstance(ax, (np.ndarray, list, tuple)):
            shape = np.shape(ax)
            nrows = kwargs.get("nrows", shape[0])
            ncols = kwargs.get("ncols", shape[1] if len(shape) > 1 else 1)
            if as_plotarray:
                return PlotArray(
                    [Plotter(fig, a, log, chain)
                        for a in np.array(ax).flatten()],
                    nrows=nrows,
                    ncols=ncols
                )
            else:
                return Plots(
                    [Plotter(fig, a, log, chain)
                        for a in np.array(ax).flatten()],
                    nrows=nrows,
                    ncols=ncols
                )
        elif as_plotarray:
            return PlotArray([Plotter(fig, ax, log, chain)], nrows=1, ncols=1)
        else:
            return Plotter(fig, ax, log, chain)

    @staticmethod
    def new_3d(*args, **kwargs):
        fig = pp.figure(*args, **kwargs)
        ax = p3.Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        log = kwargs.get("log", True)
        chain = kwargs.get("chain", True)
        return Plotter(fig, ax, log, chain)

    @staticmethod
    def new_gridspec(gridspec_kw, pos, shareax=None, *args, **kwargs):
        fig = kwargs.get("fig", pp.figure(*args, **kwargs))
        gs = fig.add_gridspec(**gridspec_kw)
        shareax = dict() if shareax is None else shareax
        ax = list()
        for k, p in enumerate(pos):
            share = shareax.get(k, dict())
            sharex = share.get("x", None)
            sharey = share.get("y", None)
            subplot = fig.add_subplot(
                gs[p],
                sharex=ax[sharex] if sharex is not None else None,
                sharey=ax[sharey] if sharey is not None else None,
            )
            ax.append(subplot)
        log = kwargs.get("log", True)
        chain = kwargs.get("chain", True)
        return Plots(
            [Plotter(fig, a, log, chain) for a in ax],
            nrows=len(ax)
        )

    def twinx(self, *args, **kwargs):
        return (self, Plotter(self.fig, self.ax.twinx(), self.log, self.chain))

    def twiny(self, *args, **kwargs):
        return (self, Plotter(self.fig, self.ax.twiny(), self.log, self.chain))

    def sharex(self, *args, **kwargs):
        return self._process_call(self.ax.sharex, args, kwargs)

    def sharey(self, *args, **kwargs):
        return self._process_call(self.ax.sharey, args, kwargs)

    def plot(self, *args, **kwargs):
        return self._process_call(self.ax.plot, args, kwargs)

    def plot_surface(self, *args, **kwargs):
        return self._process_call(self.ax.plot_surface, args, kwargs)

    def plot_trisurf(self, *args, **kwargs):
        return self._process_call(self.ax.plot_trisurf, args, kwargs)

    def errorbar(self, *args, **kwargs):
        return self._process_call(self.ax.errorbar, args, kwargs)

    def semilogx(self, *args, **kwargs):
        return self._process_call(self.ax.semilogx, args, kwargs)

    def semilogy(self, *args, **kwargs):
        return self._process_call(self.ax.semilogy, args, kwargs)

    def loglog(self, *args, **kwargs):
        return self._process_call(self.ax.loglog, args, kwargs)

    def scatter(self, *args, **kwargs):
        return self._process_call(self.ax.scatter, args, kwargs)

    def contour(self, *args, **kwargs):
        log = kwargs.pop("log") if "log" in kwargs.keys() else self.log
        chain = kwargs.pop("chain") if "chain" in kwargs.keys() else self.chain
        mut = kwargs.pop("mut") if "mut" in kwargs.keys() else True
        X = self.ax.contour(*args, **kwargs)
        if log:
            self.outputs.append(X)
        if mut:
            self.im = X
        return self if chain else X

    def contourf(self, *args, **kwargs):
        log = kwargs.pop("log") if "log" in kwargs.keys() else self.log
        chain = kwargs.pop("chain") if "chain" in kwargs.keys() else self.chain
        mut = kwargs.pop("mut") if "mut" in kwargs.keys() else True
        X = self.ax.contourf(*args, **kwargs)
        if log:
            self.outputs.append(X)
        if mut:
            self.im = X
        return self if chain else X

    def axhline(self, *args, **kwargs):
        return self._process_call(self.ax.axhline, args, kwargs)

    def axvline(self, *args, **kwargs):
        return self._process_call(self.ax.axvline, args, kwargs)

    def axline(self, *args, **kwargs):
        return self._process_call(self.ax.axline, args, kwargs)

    def fill(self, *args, **kwargs):
        return self._process_call(self.ax.fill, args, kwargs)

    def fill_between(self, *args, **kwargs):
        return self._process_call(self.ax.fill_between, args, kwargs)

    def imshow(self, *args, **kwargs):
        log = kwargs.pop("log") if "log" in kwargs.keys() else self.log
        chain = kwargs.pop("chain") if "chain" in kwargs.keys() else self.chain
        mut = kwargs.pop("mut") if "mut" in kwargs.keys() else True
        X = self.ax.imshow(*args, **kwargs)
        if log:
            self.outputs.append(X)
        if mut:
            self.im = X
        return self if chain else X

    def colorplot(
        self,
        x,
        y,
        Z,
        *args,
        interpolation: str="nearest",
        **kwargs
    ):
        log = kwargs.pop("log") if "log" in kwargs.keys() else self.log
        chain = kwargs.pop("chain") if "chain" in kwargs.keys() else self.chain
        mut = kwargs.pop("mut") if "mut" in kwargs.keys() else True
        dx = [x[1] - x[0], x[-1] - x[-2]]
        dy = [y[1] - y[0], y[-1] - y[-2]]
        extent = [
            x.min() - dx[0] / 2.0, x.max() + dx[1] / 2.0,
            y.min() - dy[0] / 2.0, y.max() + dy[1] / 2.0,
        ]
        im = img.NonUniformImage(
            self.ax,
            interpolation=interpolation,
            extent=extent,
            *args,
            **kwargs
        )
        im.set_data(x, y, Z)
        X = self.ax.add_image(im)
        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        if log:
            self.outputs.append(X)
        if mut:
            self.im = X
        return self if chain else X

    def hist(self, *args, **kwargs):
        return self._process_call(self.ax.hist, args, kwargs)

    def hist2d(self, *args, **kwargs):
        return self._process_call(self.ax.hist2d, args, kwargs)

    def bar(self, *args, **kwargs):
        return self._process_call(self.ax.bar, args, kwargs)

    def barh(self, *args, **kwargs):
        return self._process_call(self.ax.barh, args, kwargs)

    def quiver(self, *args, **kwargs):
        return self._process_call(self.ax.quiver, args, kwargs)

    def streamplot(self, *args, **kwargs):
        return self._process_call(self.ax.streamplot, args, kwargs)

    def arrow(self, *args, **kwargs):
        return self._process_call(self.ax.arrow, args, kwargs)

    def indicate_inset(self, *args, **kwargs):
        return self._process_call(self.ax.indicate_inset, args, kwargs)

    def indicate_inset_zoom(self, *args, **kwargs):
        return self._process_call(self.ax.indicate_inset_zoom, args, kwargs)

    def inset_axes(self, *args, **kwargs):
        log = kwargs.pop("log") if "log" in kwargs.keys() else self.log
        chain = kwargs.pop("chain") if "chain" in kwargs.keys() else self.chain
        mut = kwargs.pop("mut") if "mut" in kwargs.keys() else True
        X = self.ax.inset_axes(*args, **kwargs)
        if log:
            self.outputs.append(X)
        if mut:
            self.ax = X
        return self if chain else X

    def secondary_xaxis(self, *args, **kwargs):
        log = kwargs.pop("log") if "log" in kwargs.keys() else self.log
        chain = kwargs.pop("chain") if "chain" in kwargs.keys() else self.chain
        mut = kwargs.pop("mut") if "mut" in kwargs.keys() else True
        X = self.ax.secondary_xaxis(*args, **kwargs)
        if log:
            self.outputs.append(X)
        if mut:
            self.ax = X
        return self if chain else X

    def secondary_yaxis(self, *args, **kwargs):
        log = kwargs.pop("log") if "log" in kwargs.keys() else self.log
        chain = kwargs.pop("chain") if "chain" in kwargs.keys() else self.chain
        mut = kwargs.pop("mut") if "mut" in kwargs.keys() else True
        X = self.ax.secondary_yaxis(*args, **kwargs)
        if log:
            self.outputs.append(X)
        if mut:
            self.ax = X
        return self if chain else X

    def text(self, *args, **kwargs):
        return self._process_call(self.ax.text, args, kwargs)

    def text_ax(self, *args, **kwargs):
        return self._process_call(
            self.ax.text, args, {**kwargs, "transform": self.ax.transAxes})

    def get_xlim(self, *args, **kwargs):
        return self.ax.get_xlim(*args, **kwargs)

    def get_ylim(self, *args, **kwargs):
        return self.ax.get_ylim(*args, **kwargs)

    def get_clim(self, *args, **kwargs):
        return self.im.get_clim(*args, **kwargs)

    def get_xticks(self, *args, **kwargs):
        return self.ax.get_xticks(*args, **kwargs)

    def get_yticks(self, *args, **kwargs):
        return self.ax.get_yticks(*args, **kwargs)

    def get_cticks(self, *args, **kwargs):
        return self.cbar.get_ticks(*args, **kwargs)

    def get_xticklabels(self, *args, **kwargs):
        return self.ax.get_xticklabels(*args, **kwargs)

    def get_yticklabels(self, *args, **kwargs):
        return self.ax.get_yticklabels(*args, **kwargs)

    def get_cticklabels(self, *args, **kwargs):
        return self.cbar.get_ticklabels(*args, **kwargs)

    def get_xlabel(self, *args, **kwargs):
        return self.ax.get_xlabel(*args, **kwargs)

    def get_ylabel(self, *args, **kwargs):
        return self.ax.get_ylabel(*args, **kwargs)

    def get_zlabel(self, *args, **kwargs):
        return self.ax.get_zlabel(*args, **kwargs)

    def get_clabel(self, *args, **kwargs):
        return self.cbar.get_label(*args, **kwargs)

    def get_title(self, *args, **kwargs):
        return self.ax.get_title(*args, **kwargs)

    def set_xscale(self, *args, **kwargs):
        return self._process_call(self.ax.set_xscale, args, kwargs)

    def set_yscale(self, *args, **kwargs):
        return self._process_call(self.ax.set_yscale, args, kwargs)

    def set_zscale(self, *args, **kwargs):
        return self._process_call(self.ax.set_zscale, args, kwargs)

    def set_xlim(self, *args, **kwargs):
        return self._process_call(self.ax.set_xlim, args, kwargs)

    def set_ylim(self, *args, **kwargs):
        return self._process_call(self.ax.set_ylim, args, kwargs)

    def set_clim(self, *args, **kwargs):
        return self._process_call(self.im.set_clim, args, kwargs)

    def set_xticks(self, *args, **kwargs):
        return self._process_call(self.ax.set_xticks, args, kwargs)

    def set_yticks(self, *args, **kwargs):
        return self._process_call(self.ax.set_yticks, args, kwargs)

    def set_cticks(self, *args, **kwargs):
        return self._process_call(self.cbar.set_ticks, args, kwargs)

    def set_xticklabels(self, *args, **kwargs):
        return self._process_call(self.ax.set_xticklabels, args, kwargs)

    def set_yticklabels(self, *args, **kwargs):
        return self._process_call(self.ax.set_yticklabels, args, kwargs)

    def set_cticklabels(self, *args, **kwargs):
        return self._process_call(self.cbar.set_ticklabels, args, kwargs)

    def tick_params(self, *args, **kwargs):
        return self._process_call(self.ax.tick_params, args, kwargs)

    def set_xlabel(self, *args, **kwargs):
        return self._process_call(self.ax.set_xlabel, args, kwargs)

    def set_ylabel(self, *args, **kwargs):
        return self._process_call(self.ax.set_ylabel, args, kwargs)

    def set_zlabel(self, *args, **kwargs):
        return self._process_call(self.ax.set_zlabel, args, kwargs)

    def set_clabel(self, *args, **kwargs):
        return self._process_call(self.cbar.set_label, args, kwargs)

    def set_title(self, *args, **kwargs):
        return self._process_call(self.ax.set_title, args, kwargs)

    def set(self, **kwargs):
        which = kwargs.pop("which") if "which" in kwargs.keys() else "ax"
        if which == "ax":
            O = self.ax
        elif which == "fig":
            O = self.fig
        elif which == "cbar":
            O = self.cbar
        elif which == "im":
            O = self.im
        else:
            raise Exception("invalid 'which'")
        return self._process_call(O.set, [], kwargs)

    def invert_xaxis(self, *args, **kwargs):
        return self._process_call(self.ax.invert_xaxis, args, kwargs)

    def invert_yaxis(self, *args, **kwargs):
        return self._process_call(self.ax.invert_yaxis, args, kwargs)

    def colorbar(self, *args, **kwargs):
        log = kwargs.pop("log") if "log" in kwargs.keys() else self.log
        chain = kwargs.pop("chain") if "chain" in kwargs.keys() else self.chain
        mut = kwargs.pop("mut") if "mut" in kwargs.keys() else True
        X = self.fig.colorbar(self.im, ax=self.ax, *args, **kwargs)
        if log:
            self.outputs.append(X)
        if mut:
            self.cbar = X
        return self if chain else X

    def grid(self, *args, **kwargs):
        return self._process_call(self.ax.grid, args, kwargs)

    def ggrid(self, onoff=True, *args, **kwargs):
        log = kwargs.pop("log") if "log" in kwargs.keys() else self.log
        chain = kwargs.pop("chain") if "chain" in kwargs.keys() else self.chain
        X = grid(onoff, self.ax)
        if log:
            self.outputs.append(X)
        return self if chain else X

    def legend(self, *args, **kwargs):
        return self._process_call(self.ax.legend, args, kwargs)

    def tight_layout(self, *args, **kwargs):
        return self._process_call(self.fig.tight_layout, args, kwargs)

    def set_box_aspect(self, *args, **kwargs):
        return self._process_call(self.ax.set_box_aspect, args, kwargs)

    def axis(self, *args, **kwargs):
        return self._process_call(self.ax.axis, args, kwargs)

    def savefig(self, *args, **kwargs):
        return self._process_call(self.fig.savefig, args, kwargs)

    def show(self, *args, **kwargs):
        return self._process_call(pp.show, args, kwargs)

    def close(self):
        pp.close(self.fig)

    def f(self, f, *args, **kwargs):
        return self._process_call(f, args, kwargs)

    def do(self, f: str, *args, **kwargs):
        return self._process_call(getattr(self, f), args, kwargs)

    def _process_call(self, f, args, kwargs):
        log = kwargs.pop("log") if "log" in kwargs.keys() else self.log
        chain = kwargs.pop("chain") if "chain" in kwargs.keys() else self.chain
        X = f(*args, **kwargs)
        if log:
            self.outputs.append(X)
        return self if chain else X

    def _process_method(self, *args, **kwargs):
        if self._method is None:
            raise Exception("method is None")
        ret = self._process_call(self._method, args, kwargs)
        self._method = None
        return ret

    def __getattr__(self, f):
        if f in dir(self):
            return getattr(self, f)
        for X in [self.ax, self.fig, self.im, self.cbar]:
            if f in dir(X):
                _f = getattr(X, f)
                if isinstance(_f, type(self.do)):
                    self._method = _f
                    return self._process_method
                else:
                    return _f
        raise AttributeError

class FigSize:
    def __init__(self, wh):
        assert isinstance(wh, (list, tuple))
        assert len(wh) == 2
        self.wh = list(wh)

    def _opcheck(self, other):
        assert isinstance(other, (int, float, list, tuple, FigSize))
        if isinstance(other, (list, tuple)):
            assert len(other) == 2

    def __abs__(self, /):
        return FigSize([abs(self.__w), abs(self.__h)])

    def __pos__(self, /):
        return self
    
    def __neg__(self, /):
        return FigSize([-self.__w, -self.__h])

    def __invert__(self, /):
        return FigSize([self.__h, self.__w])

    def __contains__(self, val, /):
        return val in self.__wh

    def __eq__(self, val, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            return (self.__w == val) and (self.__h == val)
        elif isinstance(val, (list, tuple)):
            return (self.__w == val[0]) and (self.__h == val[0])
        elif isinstance(val, FigSize):
            return (self.__w == val.w) and (self.__h == val.h)

    def __ne__(self, val, /):
        return not (self == val)

    def __getitem__(self, key, /):
        assert key in [0, "w", 1, "h"]
        if key in [0, "w"]:
            return self.__w
        elif key in [1, "h"]:
            return self.__h

    def __setitem__(self, key, val, /):
        assert key in [0, "w", 1, "h"]
        if key in [0, "w"]:
            self.w = val
        elif key in [1, "h"]:
            self.h = val

    def __iter__(self, /):
        return iter(self.__wh)

    def __reversed__(self, /):
        return reversed(self.__wh)

    def __len__(self, /):
        return len(self.__wh)

    @property # wh
    def wh(self):
        return self.__wh
    @wh.setter
    def wh(self, wh, /):
        assert isinstance(wh, (list, tuple))
        assert len(wh) == 2
        assert isinstance(wh[0], (int, float)) and isinstance(wh[1], (int, float))
        self.__wh = list(wh)
        self.__w = wh[0]
        self.__h = wh[1]

    @property # w
    def w(self):
        return self.__w
    @w.setter
    def w(self, w, /):
        assert isinstance(w, (int, float))
        self.__wh[0] = w
        self.__w = w

    @property # h
    def h(self):
        return self.__h
    @h.setter
    def h(self, h, /):
        assert isinstance(h, (int, float))
        self.__wh[1] = h
        self.__h = h

    def __add__(self, val, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            return FigSize([self.__w+val, self.__h+val])
        elif isinstance(val, (list, tuple)):
            return FigSize([self.__w+val[0], self.__h+val[1]])
        elif isinstance(val, FigSize):
            return FigSize([self.__w+val.w, self.__h+val.h])
    def __radd__(self, val, /):
        return self.__add__(val)
    def __iadd__(self, val, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            self.w = self.w + val
            self.h = self.h + val
        elif isinstance(val, (list, tuple)):
            self.w = self.w + val[0]
            self.h = self.h + val[1]
        elif isinstance(val, FigSize):
            self.w = self.w + val.w
            self.h = self.h + val.h

    def __sub__(self, val, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            return FigSize([self.__w-val, self.__h-val])
        elif isinstance(val, (list, tuple)):
            return FigSize([self.__w-val[0], self.__h-val[1]])
        elif isinstance(val, FigSize):
            return FigSize([self.__w-val.w, self.__h-val.h])
    def __rsub__(self, val, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            return FigSize([val-self.__w, val-self.__h])
        elif isinstance(val, (list, tuple)):
            return FigSize([val[0]-self.__w, val[1]-self.__h])
        elif isinstance(val, FigSize):
            return FigSize([val.w-self.__w, val.h-self.__h])
    def __isub__(self, val, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            self.w = self.__w - val
            self.h = self.__h - val
        elif isinstance(val, (list, tuple)):
            self.w = self.__w - val[0]
            self.h = self.__h - val[1]
        elif isinstance(val, FigSize):
            self.w = self.__w - val.w
            self.h = self.__h - val.h

    def __mul__(self, val, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            return FigSize([self.__w*val, self.__h*val])
        elif isinstance(val, (list, tuple)):
            return FigSize([self.__w*val[0], self.__h*val[1]])
        elif isinstance(val, FigSize):
            return FigSize([self.__w*val.w, self.__h*val.h])
    def __rmul__(self, val, /):
        return self.__mul__(val)
    def __isub__(self, val, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            self.w = self.__w * val
            self.h = self.__h * val
        elif isinstance(val, (list, tuple)):
            self.w = self.__w * val[0]
            self.h = self.__h * val[1]
        elif isinstance(val, FigSize):
            self.w = self.__w * val.w
            self.h = self.__h * val.h

    def __truediv__(self, val, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            return FigSize([self.__w/val, self.__h/val])
        elif isinstance(val, (list, tuple)):
            return FigSize([self.__w/val[0], self.__h/val[1]])
        elif isinstance(val, FigSize):
            return FigSize([self.__w/val.w, self.__h/val.h])
    def __rtruediv__(self, val, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            return FigSize([val/self.__w, val/self.__h])
        elif isinstance(val, (list, tuple)):
            return FigSize([val[0]/self.__w, val[1]/self.__h])
        elif isinstance(val, FigSize):
            return FigSize([val.w/self.__w, val.h/self.__h])
    def __itruediv__(self, val, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            self.w = self.__w / val
            self.h = self.__h / val
        elif isinstance(val, (list, tuple)):
            self.w = self.__w / val[0]
            self.h = self.__h / val[1]
        elif isinstance(val, FigSize):
            self.w = self.__w / val.w
            self.h = self.__h / val.h

    def __floordiv__(self, val, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            return FigSize([self.__w//val, self.__h//val])
        elif isinstance(val, (list, tuple)):
            return FigSize([self.__w//val[0], self.__h//val[1]])
        elif isinstance(val, FigSize):
            return FigSize([self.__w//val.w, self.__h//val.h])
    def __rfloordiv__(self, val, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            return FigSize([val//self.__w, val//self.__h])
        elif isinstance(val, (list, tuple)):
            return FigSize([val[0]//self.__w, val[1]//self.__h])
        elif isinstance(val, FigSize):
            return FigSize([val.w//self.__w, val.h//self.__h])
    def __ifloordiv__(self, val, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            self.w = self.__w // val
            self.h = self.__h // val
        elif isinstance(val, (list, tuple)):
            self.w = self.__w // val[0]
            self.h = self.__h // val[1]
        elif isinstance(val, FigSize):
            self.w = self.__w // val.w
            self.h = self.__h // val.h

    def __mod__(self, val, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            return FigSize([self.__w%val, self.__h%val])
        elif isinstance(val, (list, tuple)):
            return FigSize([self.__w%val[0], self.__h%val[1]])
        elif isinstance(val, FigSize):
            return FigSize([self.__w%val.w, self.__h%val.h])
    def __rmod__(self, val, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            return FigSize([val%self.__w, val%self.__h])
        elif isinstance(val, (list, tuple)):
            return FigSize([val[0]%self.__w, val[1]%self.__h])
        elif isinstance(val, FigSize):
            return FigSize([val.w%self.__w, val.h%self.__h])
    def __imod__(self, val, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            self.w = self.__w % val
            self.h = self.__h % val
        elif isinstance(val, (list, tuple)):
            self.w = self.__w % val[0]
            self.h = self.__h % val[1]
        elif isinstance(val, FigSize):
            self.w = self.__w % val.w
            self.h = self.__h % val.h

    def __pow__(self, val, mod=None, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            return FigSize([pow(self.__w, val, mod), pow(self.__h, val, mod)])
        elif isinstance(val, (list, tuple)):
            return FigSize([pow(self.__w, val[0], mod), pow(self.__h, val[1], mod)])
        elif isinstance(val, FigSize):
            return FigSize([pow(self.__w, val.w, mod), pow(self.__h, val.h, mod)])
    def __ipow__(self, val, /):
        self._opcheck(val)
        if isinstance(val, (int, float)):
            self.w = self.__w ** val
            self.h = self.__h ** val
        elif isinstance(val, (list, tuple)):
            self.w = self.__w ** val[0]
            self.h = self.__h ** val[1]
        elif isinstance(val, FigSize):
            self.w = self.__w ** val.w
            self.h = self.__h ** val.h

    def __repr__(self, /):
        return "FigSize("+str(self.__wh)+")"

    def __str__(self, /):
        return "FigSize("+str(self.__wh)+")"

