import numpy as np
from typing import Tuple, Dict, Optional
from scipy.ndimage import zoom as ndi_zoom, shift as ndi_shift, center_of_mass

# ----------------------------
# 基础工具
# ----------------------------
def _ensure_odd(n: int) -> int:
    """保证为奇数，有利于以像素中心为原点的对齐。"""
    return n if n % 2 == 1 else n + 1

def _pad_or_crop_center(arr: np.ndarray, target: int) -> np.ndarray:
    """以中心对齐，把数组 pad/crop 到 target×target。"""
    h, w = arr.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ty = tx = target
    out = np.zeros((ty, tx), dtype=arr.dtype)

    # 目标中心
    Cy, Cx = (ty - 1) / 2.0, (tx - 1) / 2.0

    # 计算在源数组中要取的整数边界，以及在目标中的放置位置
    y0_src = int(np.floor(cy - Cy))
    x0_src = int(np.floor(cx - Cx))
    y1_src = y0_src + ty
    x1_src = x0_src + tx

    # 与源/目标相交的范围
    y0_s = max(0, y0_src)
    x0_s = max(0, x0_src)
    y1_s = min(h, y1_src)
    x1_s = min(w, x1_src)

    y0_t = max(0, -y0_src)
    x0_t = max(0, -x0_src)
    y1_t = y0_t + (y1_s - y0_s)
    x1_t = x0_t + (x1_s - x0_s)

    if (y1_s > y0_s) and (x1_s > x0_s):
        out[y0_t:y1_t, x0_t:x1_t] = arr[y0_s:y1_s, x0_s:x1_s]
    return out

def _recentre(arr: np.ndarray, how: Optional[str] = "peak") -> np.ndarray:
    """把 PSF 重心或峰值移动到图像中心。"""
    h, w = arr.shape
    center = np.array([(h - 1) / 2.0, (w - 1) / 2.0])

    if how is None:
        return arr

    if how == "peak":
        peak = np.unravel_index(np.argmax(arr), arr.shape)
        pos = np.array(peak, dtype=float)
    elif how == "com":
        pos = np.array(center_of_mass(np.clip(arr, a_min=0, a_max=None)))
    else:
        raise ValueError("how must be one of {'peak','com',None}")

    shift_vec = center - pos  # dy, dx
    return ndi_shift(arr, shift=shift_vec, order=3, mode="constant", cval=0.0, prefilter=True)

# ----------------------------
# 主函数：重采样到统一像元与尺寸
# ----------------------------
def resample_psf_to_common(
    psf: np.ndarray,
    pixscale_in: float,                    # 输入 PSF 像元尺度 (arcsec/pix)，例如 BASS ~0.454
    pixscale_out: float = 0.262,           # 目标像元尺度 (arcsec/pix)
    target_size: int = 63,                 # 目标阵列大小 (像素)
    recenter: Optional[str] = "peak",      # {'peak','com',None}
    clip_negative: bool = True,
    renormalize: bool = True,
) -> np.ndarray:
    """
    把任意 PSF stamp 统一到 pixscale_out 和 target_size×target_size。
    过程：插值缩放 -> 居中 -> pad/crop -> 归一化。

    返回：统一后的 PSF（sum=1，如果 renormalize=True）
    """
    if psf.ndim != 2:
        raise ValueError("psf must be a 2D array")

    # 1) 缩放：zoom = s_in / s_out
    zoom_factor = float(pixscale_in) / float(pixscale_out)
    out_h = _ensure_odd(int(round(psf.shape[0] * zoom_factor)))
    out_w = _ensure_odd(int(round(psf.shape[1] * zoom_factor)))
    zoom_y = out_h / psf.shape[0]
    zoom_x = out_w / psf.shape[1]

    psf_rs = ndi_zoom(psf, zoom=(zoom_y, zoom_x), order=3, prefilter=True)

    # 2) 居中
    psf_rs = _recentre(psf_rs, how=recenter)

    # 3) pad/crop 到目标大小
    psf_rs = _pad_or_crop_center(psf_rs, target=_ensure_odd(target_size))

    # 4) 处理负值（插值可能引入极小负值）
    if clip_negative:
        psf_rs = np.clip(psf_rs, a_min=0.0, a_max=None)

    # 5) 归一化
    if renormalize:
        s = psf_rs.sum()
        if s > 0:
            psf_rs = psf_rs / s

    return psf_rs

# ----------------------------
# FWHM 估计（基于二阶矩，近似高斯）
# ----------------------------
def psf_fwhm_from_moments(
    psf: np.ndarray,
    pixscale: float,  # arcsec/pix
) -> Dict[str, float]:
    """
    用二阶矩估计椭圆高斯的 σ_major/σ_minor，并给出 FWHM。
    返回单位：arcsec。
    """
    arr = np.asarray(psf, dtype=float)
    total = arr.sum()
    if total <= 0:
        return dict(fwhm_major=np.nan, fwhm_minor=np.nan, fwhm_circ=np.nan)

    h, w = arr.shape
    y, x = np.mgrid[0:h, 0:w]
    y0 = (h - 1) / 2.0
    x0 = (w - 1) / 2.0
    dy = y - y0
    dx = x - x0

    # 一阶矩（这里假定已居中；若未居中，先重心居中更稳）
    mu_x = (arr * dx).sum() / total
    mu_y = (arr * dy).sum() / total

    # 二阶中心矩
    xx = (arr * (dx - mu_x) ** 2).sum() / total
    yy = (arr * (dy - mu_y) ** 2).sum() / total
    xy = (arr * (dx - mu_x) * (dy - mu_y)).sum() / total

    # 协方差矩阵的特征值即主轴方向上的方差
    trace = xx + yy
    det = xx * yy - xy * xy
    # 防止数值问题
    term = max(trace * trace / 4.0 - det, 0.0)
    lam1 = trace / 2.0 + np.sqrt(term)  # 最大
    lam2 = trace / 2.0 - np.sqrt(term)  # 最小

    sigma_major_pix = np.sqrt(max(lam1, 0.0))
    sigma_minor_pix = np.sqrt(max(lam2, 0.0))

    # 高斯 FWHM = 2*sqrt(2*ln2) * sigma
    K = 2.0 * np.sqrt(2.0 * np.log(2.0))
    fwhm_major = K * sigma_major_pix * pixscale
    fwhm_minor = K * sigma_minor_pix * pixscale
    fwhm_circ = K * np.sqrt(sigma_major_pix * sigma_minor_pix) * pixscale  # 圆等效

    return dict(
        fwhm_major=float(fwhm_major),
        fwhm_minor=float(fwhm_minor),
        fwhm_circ=float(fwhm_circ),
    )

# ----------------------------
