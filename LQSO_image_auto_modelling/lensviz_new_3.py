
"""
lensviz — A lightweight toolkit for visualising Lenstronomy modelling outputs.

Version: 0.3.1
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Sequence
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
from astropy.wcs import WCS

__all__ = ["LensViz", "show_rgb_gallery", "build_rgb_from_band_data"]

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _find_band_file(results_dir: Path, band: str) -> Path:
    for name in (f"{band}_modelling.fits", f"{band}_modeling.fits"):
        p = results_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"[{band}] FITS not found in {results_dir}")

def _list_2d_arrays(hdul) -> Dict[str, np.ndarray]:
    out = {}
    for i, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)
        if data is not None and hasattr(data, "ndim") and data.ndim == 2:
            name = hdu.name.strip() if hdu.name else f"EXT{i}"
            out[name.upper()] = np.array(data, dtype=float)
    return out

def _guess_component(arr_dict: Dict[str,np.ndarray]) -> Dict[str,str]:
    keys = list(arr_dict.keys())
    def pick(cands, exclude=None):
        for k in keys:
            if any(c in k for c in cands) and not any(ex in k for ex in (exclude or [])):
                return k
        return None
    return dict(
        data=pick(["DATA","IMAGE","OBS"], exclude=["MODEL"]),
        model=pick(["MODEL_IMAGE","MODELIMG","MODEL"]),
        lens=pick(["LENS_LIGHT","LENS","GAL"], exclude=["SOURCE"]),
        source=pick(["SOURCE_LIGHT","SOURCE"]),
        ps=pick(["POINT","PS","AGN"]),
    )

def _build_lens(arr_dict: Dict[str,np.ndarray], comp: Dict[str,str]):
    if comp.get("lens") and comp["lens"] in arr_dict:
        return arr_dict[comp["lens"]]
    if all(comp.get(k) and comp[k] in arr_dict for k in ("model","source")):
        model = arr_dict[comp["model"]]
        source = arr_dict[comp["source"]]
        ps = arr_dict[comp["ps"]] if comp.get("ps") and comp["ps"] in arr_dict else 0.0
        return model - source - ps
    return np.full_like(arr_dict[comp["data"]], np.nan)

def _try_wcs(hdul):
    for h in hdul:
        hdr = getattr(h, "header", None)
        if hdr and "CTYPE1" in hdr and "CTYPE2" in hdr:
            try:
                w = WCS(hdr)
                if w.naxis >= 2:
                    return w
            except Exception:
                pass
    return None

def _compute_global_limits(images: Sequence[np.ndarray]):
    zs = ZScaleInterval()
    vals = []
    for img in images:
        finite = np.isfinite(img)
        if finite.any():
            vals.append(zs.get_limits(img[finite]))
    if not vals:
        return 0.0, 1.0
    vmin = min(lo for lo, _ in vals)
    vmax = max(hi for _, hi in vals)
    return vmin, vmax

def _norm_from_limits(vmin, vmax, kind="asinh", sym=False):
    if sym:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m
    if kind == "asinh":
        return ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())
    else:
        from astropy.visualization import MinMaxInterval
        return ImageNormalize(vmin=vmin, vmax=vmax)
# ---------------------------------------------------------------------
try:
    from astropy.visualization import make_lupton_rgb as _lupton
    HAS_LUPTON = True
except Exception:
    HAS_LUPTON = False

def _simple_asinh_rgb(r,g,b,perc_lo=0.25,perc_hi=99.5,alpha=3.0):
    def prep(x):
        finite = np.isfinite(x)
        lo, hi = np.nanpercentile(x[finite], [perc_lo, perc_hi]) if finite.any() else (0,1)
        x = (x - lo) / max(hi - lo, 1e-12)
        return np.clip(x, 0, 1)
    stack = np.stack([prep(r), prep(g), prep(b)], axis=-1)
    stack = np.arcsinh(alpha*stack)/np.arcsinh(alpha)
    return stack

# ---------------------------------------------------------
# Optional HumVI support (using local humvi.zip if present)
HAS_HUMVI = False
def _ensure_humvi():
    """Try to import humvi. If not installed, try loading from a nearby humvi.zip."""
    global HAS_HUMVI
    if HAS_HUMVI:
        return
    try:
        import humvi  # noqa: F401
        HAS_HUMVI = True
        return
    except Exception:
        pass
    try:
        import sys as _sys
        from pathlib import Path as _Path
        here = _Path(__file__).resolve()
        candidates = [
            here.with_name('humvi.zip'),
            _Path.cwd() / 'humvi.zip',
            here.parent / 'deps' / 'humvi.zip',
        ]
        for c in candidates:
            if c.exists():
                _sys.path.insert(0, str(c))
                try:
                    import humvi  # noqa: F401
                    HAS_HUMVI = True
                    return
                except Exception:
                    continue
    except Exception:
        pass
    HAS_HUMVI = False

def _humvi_rgb(r_band, g_band, b_band, *, Q=1.5, alpha=10.0, scales=(0.6, 0.8, 1.7)):
    """Build an RGB image using the HumVI algorithm.
    Parameters mirror HumVI's compose: Q, alpha, and per-channel scales.
    Input arrays are assumed float images (not uint8), any negatives are expected clipped earlier.
    Returns an array in [0,1] with shape (H,W,3).
    """
    _ensure_humvi()
    if not HAS_HUMVI:
        raise ImportError("humvi is not available. Place humvi.zip next to lensviz.py or install the package.")
    import numpy as _np
    from tempfile import TemporaryDirectory
    from astropy.io import fits as _fits
    from PIL import Image as _PILImage
    import humvi
    # HumVI works on files; write temporary FITS, call compose, then read PNG
    with TemporaryDirectory() as td:
        from pathlib import Path as _Path
        td_path = _Path(td)
        rf = td_path / 'R.fits'
        gf = td_path / 'G.fits'
        bf = td_path / 'B.fits'
        _fits.writeto(rf, _np.asarray(r_band, dtype='float32'), overwrite=True)
        _fits.writeto(gf, _np.asarray(g_band, dtype='float32'), overwrite=True)
        _fits.writeto(bf, _np.asarray(b_band, dtype='float32'), overwrite=True)
        outpng = td_path / 'rgb.png'
        # HumVI expects files ordered as (r,g,b) in its color sense. Our RGB mapping is (z->R, r->G, g->B) at call site.
        humvi.compose(str(rf), str(gf), str(bf), scales=scales, Q=float(Q), alpha=float(alpha), vb=False, outfile=str(outpng))
        im = _PILImage.open(outpng).convert('RGB')
        arr = _np.asarray(im, dtype='float32') / 255.0
        # flip vertically to match origin='lower'
        arr = arr[::-1, :, :]
    return arr

def build_rgb_from_band_data(band_data, component="data", method="lupton",
                            Q=8, stretch=0.1, minimum=0.0,
                            perc_lo=0.25, perc_hi=99.5, alpha=3.0,
                            per_band_scale=None, mask_negative=True,
                            scales=(0.6, 0.8, 1.7)):
    g = band_data["g"][component].astype(float)
    r = band_data["r"][component].astype(float)
    z = band_data["z"][component].astype(float)
    if per_band_scale:
        g *= per_band_scale.get("g",1); r *= per_band_scale.get("r",1); z *= per_band_scale.get("z",1)
    if mask_negative:
        g = np.where(g>0,g,0); r=np.where(r>0,r,0); z=np.where(z>0,z,0)
    if method=="lupton" and HAS_LUPTON:
        try:
            rgb_uint8 = _lupton(z,r,g,Q=Q,stretch=stretch,minimum=minimum)
        except TypeError:
            rgb_uint8 = _lupton(z,r,g,Q=Q,stretch=stretch)
        rgb = rgb_uint8/255.0
    elif method=="humvi":
        # HumVI preferred defaults (if user did not override): Q=1.5, alpha=10
        if Q == 8:
            Q = 1.5
        if alpha == 3.0:
            alpha = 10.0
        rgb = _humvi_rgb(z, r, g, Q=Q, alpha=alpha, scales=scales)
    else:
        rgb = _simple_asinh_rgb(z,r,g,perc_lo,perc_hi,alpha)
    return rgb

def show_rgb_gallery(
    band_data,
    components=('data','model','datalens'),
    method='lupton',
    save_prefix: Optional[str]=None,
    results_dir: Optional[Path]=None,
    mosaic_name: Optional[str]=None,
    show_titles: bool=True, title_fontsize: int=12, **kwargs
):
    """
    生成并显示/保存多组分 RGB 图。
    method 可选：'lupton'（Astropy），'asinh'（内置），'humvi'（用 humvi.zip）。
    若 method='humvi'，可传 Q, alpha, scales（默认 Q=1.5, alpha=10, scales=(0.6,0.8,1.7)）。
    - 单独保存的小图：{save_prefix}_{component}_{method}.png
    - 额外保存的拼接大图：{save_prefix}_{method}_mosaic.png  (或自定义 mosaic_name)
    """
    import matplotlib.pyplot as plt

    n = len(components)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    if n == 1:
        axes = [axes]

    # 若需要保存但未给路径，默认当前目录
    if save_prefix and results_dir is None:
        results_dir = Path(".")
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)

    for ax, comp in zip(axes, components):
        try:
            rgb = build_rgb_from_band_data(
                band_data, component=comp, method=method, **kwargs
            )
            ax.imshow(rgb, origin="lower")
                        
            # 标题映射（严格一致的 5 个标签）
            _title_map = {
                'data':'Data', 'psf':'PSF',
                'data_minus_psf':'Data-PSF',
                'lens':'Lens', 'residual':'Residual'
            }
            if show_titles:
                ax.set_title(_title_map.get(comp, comp), fontsize=title_fontsize)
        
            ax.axis("off")
            if save_prefix and results_dir:
                out = results_dir / f"{save_prefix}_{comp}_{method}.png"
                plt.imsave(out, rgb)
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center")
            ax.axis("off")

    plt.tight_layout()

    # 保存拼接大图
    if save_prefix and results_dir:
        mosaic_filename = mosaic_name or f"{save_prefix}_{method}_mosaic.png"
        fig.savefig(results_dir / mosaic_filename, dpi=180, bbox_inches="tight")
        print(f"[lensviz] 已保存小图与拼接大图到: {results_dir.resolve()}")

    plt.show()

# ---------------------------------------------------------------------
class LensViz:
    def __init__(self, *, ra:float, dec:float, results_root:Path=Path("results"),
                 bands=("g","r","z"), stretch_kind="linear", fig_dpi=180):
        self.ra=ra; self.dec=dec
        self.results_dir = results_root / f"ra_{ra:.5f}_dec_{dec:.5f}"
        self.bands=bands
        self.stretch_kind=stretch_kind
        self.fig_dpi=fig_dpi
        self.band_data={}
        self._global_vmin=self._global_vmax=0
        self._res_vmin_sym=self._res_vmax_sym=0

    # ------------------
    def load(self, print_hdus=False):
        band_data = {}
        for b in self.bands:
            fpath = _find_band_file(self.results_dir, b)
            with fits.open(fpath) as hdul:
                arr_dict = _list_2d_arrays(hdul)
                if print_hdus:
                    print(f"== {b}-band ==")
                    for k, a in arr_dict.items():
                        print(f" {k:>15s} {a.shape} min={np.nanmin(a):.3g} max={np.nanmax(a):.3g}")

                comp = _guess_component(arr_dict)

                # 原始各分量
                data  = arr_dict[comp["data"]]
                model = arr_dict[comp["model"]]
                lens  = _build_lens(arr_dict, comp)          # 透镜光(可能通过model-source-ps重建)

                # 各种派生分量
                residual = data - model                      # data - model
                datalens = data - lens                       # data - lens
                model_minus_lens = model - lens              # PSF+源 (即“PSF”)
                data_minus_model_minus_lens = data - model_minus_lens  # == lens + residual

                # 新增：PSF 与 data-PSF（给 show_rgb_gallery 用）
                psf = model_minus_lens
                data_minus_psf = data - psf                  # 与 data_minus_model_minus_lens 相同

                wcs = _try_wcs(hdul)

            band_data[b] = dict(
                data=data,
                model=model,
                lens=lens,
                residual=residual,
                datalens=datalens,
                model_minus_lens=model_minus_lens,
                data_minus_model_minus_lens=data_minus_model_minus_lens,
                psf=psf,
                data_minus_psf=data_minus_psf,
                comp=comp,          # 保存 _guess_component 返回的映射，便于调试
                wcs=wcs,
            )

        self.band_data = band_data

        # 计算全局显示范围
        collect = []
        for b in self.bands:
            bd = band_data[b]
            collect.extend([bd["data"], bd["model"], bd["lens"], bd["data_minus_model_minus_lens"]])
        self._global_vmin, self._global_vmax = _compute_global_limits(collect)

        all_res = [band_data[b]["residual"] for b in self.bands]
        res_vmin, res_vmax = _compute_global_limits(all_res)
        m = max(abs(res_vmin), abs(res_vmax))
        self._res_vmin_sym, self._res_vmax_sym = -m, m   # 修正原来的语法错误

        return self

    # ------------------
    def _plot_single(self,ax,img,title,norm,wcs=None):
        im=ax.imshow(img,origin='lower',cmap="bwr",norm=norm)
        ax.set_title(title,fontsize=9)
        ax.grid(False)
        if wcs is None:
            ax.set_xlabel("X (pix)")
            ax.set_ylabel("Y (pix)")
        else:
            ax.set_xlabel("RA")
            ax.set_ylabel("Dec")
        plt.colorbar(im,ax=ax,fraction=0.045,pad=0.02)

    def plot_band(self,band:str,*,save=True,unique=True):
        if band not in self.band_data:
            raise ValueError(f"band {band} not loaded")
        d=self.band_data[band]
        panels=[
            ("data",d["data"],_norm_from_limits(self._global_vmin,self._global_vmax,self.stretch_kind)),
            ("model",d["model"],_norm_from_limits(self._global_vmin,self._global_vmax,self.stretch_kind)),
            ("lens",d["lens"],_norm_from_limits(self._global_vmin,self._global_vmax,self.stretch_kind)),
            ("data-model",d["residual"],_norm_from_limits(self._res_vmin_sym,self._res_vmax_sym,self.stretch_kind,True)),
            ("data-lens",d["datalens"],_norm_from_limits(self._res_vmin_sym,self._res_vmax_sym,self.stretch_kind,True)),
            ("data-(model-lens)",d["data_minus_model_minus_lens"],
             _norm_from_limits(self._global_vmin,self._global_vmax,self.stretch_kind))
        ]
        n=len(panels)
        fig,axes=plt.subplots(1,n,figsize=(3.2*n,4),
                              subplot_kw={'projection':d["wcs"] if d["wcs"] is not None else None})
        if n==1: axes=[axes]
        for ax,(ttl,img,norm) in zip(axes,panels):
            self._plot_single(ax,img,f"{band}-band {ttl}",norm,d["wcs"])
        plt.tight_layout()
        if save:
            out=self.results_dir/f"{band}_6panels_bwr.png"
            self.results_dir.mkdir(parents=True,exist_ok=True)
            plt.savefig(out,dpi=self.fig_dpi)
            try:
                rel=out.resolve().relative_to(Path.cwd())
            except Exception:
                rel=out
            print(f"[lensviz] saved: {rel}")
        plt.show()

    def plot_all(self,*,save=True,unique=True):
        done=set()
        for b in self.bands:
            if unique and b in done: continue
            self.plot_band(b,save=save)
            done.add(b)

    def show_rgb_gallery(self,*,components=('data','model','datalens'),method='lupton',save_prefix="grz_rgb",show_titles: bool=True, title_fontsize: int=12, **kwargs):
        show_rgb_gallery(self.band_data,components=components,method=method,
                        save_prefix=save_prefix,results_dir=self.results_dir, show_titles=show_titles, title_fontsize=title_fontsize,**kwargs)

# Convenience constructor
    @classmethod
    def from_results_subdir(cls, subdir:Path, **kwargs):
        m=cls._parse_ra_dec_from_dir(subdir.name)
        if m is None:
            raise ValueError("Unable to parse ra/dec from "+subdir.name)
        return cls(ra=m[0],dec=m[1],results_root=subdir.parent,**kwargs)

    @staticmethod
    def _parse_ra_dec_from_dir(name:str):
        import re
        m=re.match(r"ra_([\\d.]+)_dec_([\\d.]+)",name)
        if not m: return None
        return float(m.group(1)),float(m.group(2))
