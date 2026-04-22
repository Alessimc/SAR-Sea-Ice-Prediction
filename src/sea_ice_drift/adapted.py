"""
Sea ice drift from coregistered Sentinel-1-like TIFFs using the
Korosov & Muckenhuber (2016) FT + PM algorithm, with minimal
adaptation of image input (no Nansat dependency).

Requirements:
    numpy, scipy, cv2, rioxarray, matplotlib

Provide two TIFFs as input with bands:
    0: HH
    1: HV
    2: LIA (not used here)
    3: mask (1 = valid, 0 = invalid)

Pixel size isset to be 100 m as default.
"""

from __future__ import absolute_import, print_function

import time
from datetime import datetime

import numpy as np
import rioxarray as rxr
from scipy import ndimage as nd
from scipy.interpolate import griddata
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.ndimage import map_coordinates



# CONSTANTS & FAKE NANSAT

AVG_EARTH_RADIUS = 6371.0  # km


class FakeNansat(object):
    """
    Minimal replacement for Nansat, with just what's needed by
    the original SeaIceDrift FT + PM code.

    Key idea:
        - We embed the pixel grid into a synthetic lat/lon space
          so that get_displacement_km() gives the correct metric
          distance: distance ≈ sqrt(dx^2 + dy^2) * pixel_size_km.
    """

    def __init__(self, img_uint8, pixel_size_m=100.0, timestamp=None):
        self._img = img_uint8  # 2D uint8 image
        self.pixel_size_km = pixel_size_m / 1000.0
        # emulate time_coverage_start used in max_drift_filter
        self.time_coverage_start = timestamp or datetime(2000, 1, 1)

    # emulate Nansat indexing: n[1] returns the image
    def __getitem__(self, key):
        if key == 1:
            return self._img
        raise KeyError("FakeNansat only supports n[1] for the image band")

    def shape(self):
        return self._img.shape

    def transform_points(self, x, y, direction=None, *args):
        """
        Minimal emulation of Nansat.transform_points:
            direction = 0 or None: (col,row) -> (lon,lat)
            direction = 1:         (lon,lat) -> (col,row)

        We define a synthetic mapping such that Haversine gives
        distance = pixel_distance * pixel_size_km.

        lat_deg = row * factor
        lon_deg = col * factor
        where factor = pixel_size_km / R * 180/pi.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        # factor in degrees per pixel
        factor = self.pixel_size_km / AVG_EARTH_RADIUS * (180.0 / np.pi)

        if direction is None or direction == 0:
            # pixels -> "lon, lat"
            lon = x * factor
            lat = y * factor
            return lon, lat
        elif direction == 1:
            # "lon, lat" -> pixels
            col = x / factor
            row = y / factor
            return col, row
        else:
            # we don't use other modes
            raise ValueError("Unsupported direction in transform_points")


# IMAGE PREPROCESSING (replace get_n)

def get_uint8_image(image, vmin, vmax, pmin, pmax):
    """Exact copy of original get_uint8_image from lib.py."""
    if vmin is None:
        vmin = np.nanpercentile(image, pmin)
        print('VMIN: ', vmin)
    if vmax is None:
        vmax = np.nanpercentile(image, pmax)
        print('VMAX: ', vmax)
    uint8Image = 1 + 254 * (image - vmin) / (vmax - vmin)
    uint8Image[uint8Image < 1] = 1
    uint8Image[uint8Image > 255] = 255
    uint8Image[~np.isfinite(image)] = 0
    return uint8Image.astype('uint8')


def preprocess_tiff_to_fakenansat(
    filename,
    hv_band_index=1,
    mask_band_index=3,
    pixel_size_m=100.0,
    vmin_db=-28.0,  # from Table 3: sigma0MHIVN
    vmax_db=-14.0,  # from Table 3: sigma0MHAVX
    pmin=10,
    pmax=99,
    timestamp=None,
):
    """
    Replacement for get_n(): open TIFF, extract HV, mask, convert to dB,
    scale to uint8, wrap in FakeNansat.

    This is the ONLY major deviation from original SeaIceDrift I/O.
    """
    da = rxr.open_rasterio(filename)

    hv = da.isel(band=hv_band_index).data.astype('float32')
    mask = da.isel(band=mask_band_index).data

    hv[mask == 0] = np.nan
    hv[hv <= 0] = np.nan
    hv_db = 10.0 * np.log10(hv)

    img_uint8 = get_uint8_image(hv_db, vmin_db, vmax_db, pmin, pmax)

    n = FakeNansat(img_uint8, pixel_size_m=pixel_size_m, timestamp=timestamp)
    return n


# LIB FUNCTIONS NEEDED BY FT/PM (UNCHANGED)

def get_displacement_km(n1, x1, y1, n2, x2, y2):
    """Exact copy of lib.get_displacement_km, but using FakeNansat."""
    lon1, lat1 = n1.transform_points(x1, y1)
    lon2, lat2 = n2.transform_points(x2, y2)

    lt1, ln1, lt2, ln2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lt2 - lt1
    dlon = ln2 - ln1
    d = (np.sin(dlat * 0.5) ** 2 +
         np.cos(lt1) * np.cos(lt2) * np.sin(dlon * 0.5) ** 2)
    return 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))


def get_speed_ms(n1, x1, y1, n2, x2, y2):
    """Exact copy of lib.get_speed_ms."""
    dt = (n2.time_coverage_start - n1.time_coverage_start).total_seconds()
    return 1000.0 * get_displacement_km(n1, x1, y1, n2, x2, y2) / abs(dt)


def interpolation_poly(x1, y1, x2, y2, x1grd, y1grd, order=1, **kwargs):
    """Exact copy of lib.interpolation_poly."""
    A = [np.ones(len(x1)), x1, y1]
    if order > 1:
        A += [x1**2, y1**2, x1*y1]
    if order > 2:
        A += [x1**3, y1**3, x1**2*y1, y1**2*x1]

    A = np.vstack(A).T
    Bx = np.linalg.lstsq(A, x2, rcond=-1)[0]
    By = np.linalg.lstsq(A, y2, rcond=-1)[0]
    x1grdF = x1grd.flatten()
    y1grdF = y1grd.flatten()

    A = [np.ones(len(x1grdF)), x1grdF, y1grdF]
    if order > 1:
        A += [x1grdF**2, y1grdF**2, x1grdF*y1grdF]
    if order > 2:
        A += [x1grdF**3, y1grdF**3, x1grdF**2*y1grdF, y1grdF**2*x1grdF]
    A = np.vstack(A).T
    x2grd = np.dot(A, Bx).reshape(x1grd.shape)
    y2grd = np.dot(A, By).reshape(x1grd.shape)

    return x2grd, y2grd


def interpolation_near(x1, y1, x2, y2, x1grd, y1grd, method='linear', **kwargs):
    """Exact copy of lib.interpolation_near."""
    src = np.array([y1, x1]).T
    dst = np.array([y1grd, x1grd]).T
    x2grd = griddata(src, x2, dst, method=method)
    y2grd = griddata(src, y2, dst, method=method)
    return x2grd, y2grd


def _fill_gpi(shape, gpi, data):
    """Exact copy of lib._fill_gpi."""
    y = np.zeros(shape).flatten() + np.nan
    y[gpi] = data
    return y.reshape(shape)


# FEATURE TRACKING (ftlib.py, UNCHANGED EXCEPT DOMAIN)

def find_key_points(image,
                    edgeThreshold=34,
                    nFeatures=100000,
                    nLevels=7,
                    patchSize=34,
                    verbose=False,
                    **kwargs):
    """Exact copy of ftlib.find_key_points."""
    if cv2.__version__.startswith('3.') or cv2.__version__.startswith('4.'):
        detector = cv2.ORB_create()
        detector.setEdgeThreshold(edgeThreshold)
        detector.setMaxFeatures(nFeatures)
        detector.setNLevels(nLevels)
        detector.setPatchSize(patchSize)
    else:
        detector = cv2.ORB()
        detector.setInt('edgeThreshold', edgeThreshold)
        detector.setInt('nFeatures', nFeatures)
        detector.setInt('nLevels', nLevels)
        detector.setInt('patchSize', patchSize)
    keyPoints, descriptors = detector.detectAndCompute(image, None)
    if verbose:
        print('Key points found: %d' % len(keyPoints))
    return keyPoints, descriptors


def _get_matches(descriptors1, descriptors2, matcher=cv2.BFMatcher,
                 norm=cv2.NORM_HAMMING, verbose=False):
    """Exact copy of ftlib._get_matches."""
    t0 = time.time()
    bf = matcher(norm)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    t1 = time.time()
    if verbose:
        print('Keypoints matched', t1 - t0)
    return matches


def _filter_matches(matches, ratio_test, keyPoints1, keyPoints2, verbose=False):
    """Exact copy of ftlib._filter_matches."""
    good = []
    for m, n in matches:
        if m.distance < ratio_test * n.distance:
            good.append(m)
    if verbose:
        print('Ratio test %f found %d keypoints' % (ratio_test, len(good)))

    x1 = np.array([keyPoints1[m.queryIdx].pt[0] for m in good])
    y1 = np.array([keyPoints1[m.queryIdx].pt[1] for m in good])
    x2 = np.array([keyPoints2[m.trainIdx].pt[0] for m in good])
    y2 = np.array([keyPoints2[m.trainIdx].pt[1] for m in good])
    return x1, y1, x2, y2


def get_match_coords(keyPoints1, descriptors1,
                     keyPoints2, descriptors2,
                     matcher=cv2.BFMatcher,
                     norm=cv2.NORM_HAMMING,
                     ratio_test=0.7,
                     verbose=False,
                     **kwargs):
    """Exact copy of ftlib.get_match_coords."""
    matches = _get_matches(descriptors1,
                           descriptors2, matcher, norm, verbose)
    x1, y1, x2, y2 = _filter_matches(matches, ratio_test,
                                     keyPoints1, keyPoints2, verbose)
    return x1, y1, x2, y2


def domain_filter(n, keyPoints, descr, domain, domainMargin=0, verbose=False, **kwargs):
    """
    Exact copy of ftlib.domain_filter, but using FakeNansat for both n and domain.
    In this case, both images are same grid, so this effectively just checks bounds.
    """
    cols = [kp.pt[0] for kp in keyPoints]
    rows = [kp.pt[1] for kp in keyPoints]
    lon, lat = n.transform_points(cols, rows, 0)
    colsD, rowsD = domain.transform_points(lon, lat, 1)
    gpi = ((colsD >= 0 + domainMargin) *
           (rowsD >= 0 + domainMargin) *
           (colsD <= domain.shape()[1] - domainMargin) *
           (rowsD <= domain.shape()[0] - domainMargin))
    if verbose:
        print('Domain filter: %d -> %d' % (len(keyPoints), len(gpi[gpi])))
    return list(np.array(keyPoints)[gpi]), descr[gpi]


def max_drift_filter(n1, x1, y1, n2, x2, y2,
                     max_speed=0.5, max_drift=None, verbose=False, **kwargs):
    """
    Exact copy of ftlib.max_drift_filter.

    IMPORTANT:
        Because FakeNansat.transform_points maps pixels to a synthetic lat/lon
        where 1 pixel = pixel_size_km, the displacement in km computed by
        get_displacement_km() is exactly pixel_distance * pixel_size_km.

        So F2 = 8 km works exactly as in the paper, just on chosen grid.
    """
    try:
        n1_time_coverage_start = n1.time_coverage_start
        n2_time_coverage_start = n2.time_coverage_start
    except ValueError:
        data_has_timestamp = False
    else:
        data_has_timestamp = True

    if data_has_timestamp and max_speed is not None:
        gpi = get_speed_ms(n1, x1, y1, n2, x2, y2) <= max_speed
    elif max_drift is not None:
        gpi = 1000.0 * get_displacement_km(n1, x1, y1, n2, x2, y2) <= max_drift
    else:
        raise ValueError("""
        Error while filtering matching vectors!
        Input data does not have time stamp, and <max_drift> is not set.
        """)

    if verbose:
        print('MaxDrift filter: %d -> %d' % (len(x1), len(gpi[gpi])))
    return x1[gpi], y1[gpi], x2[gpi], y2[gpi]


def lstsq_filter(x1, y1, x2, y2, psi=200, order=2, verbose=False, **kwargs):
    """Exact copy of ftlib.lstsq_filter."""
    if len(x1) == 0:
        return map(np.array, [[], [], [], []])
    x2sim, y2sim = interpolation_poly(x1, y1, x2, y2, x1, y1, order=order)
    err = np.hypot(x2 - x2sim, y2 - y2sim)
    gpi = err < psi
    if verbose:
        print('LSTSQ filter: %d -> %d' % (len(x1), len(gpi[gpi])))
    return x1[gpi], y1[gpi], x2[gpi], y2[gpi]


def feature_tracking(n1, n2, **kwargs):
    """Exact copy of ftlib.feature_tracking; n1,n2 are FakeNansat."""
    kp1, descr1 = find_key_points(n1[1], **kwargs)
    kp2, descr2 = find_key_points(n2[1], **kwargs)
    if len(kp1) < 2 or len(kp2) < 2:
        return (np.array([]),) * 4

    kp1, descr1 = domain_filter(n1, kp1, descr1, n2, **kwargs)
    if len(kp1) < 2:
        return (np.array([]),) * 4
    kp2, descr2 = domain_filter(n2, kp2, descr2, n1, **kwargs)
    if len(kp2) < 2:
        return (np.array([]),) * 4

    x1, y1, x2, y2 = get_match_coords(kp1, descr1, kp2, descr2, **kwargs)
    x1, y1, x2, y2 = max_drift_filter(n1, x1, y1, n2, x2, y2, **kwargs)
    x1, y1, x2, y2 = lstsq_filter(x1, y1, x2, y2, **kwargs)
    return x1, y1, x2, y2


# PATTERN MATCHING (pmlib.py, LOGIC UNCHANGED)

def get_hessian(ccm, hes_norm=True, hes_smth=False, **kwargs):
    """Exact copy of pmlib.get_hessian."""
    if hes_smth:
        ccm2 = nd.filters.gaussian_filter(ccm, 1)
    else:
        ccm2 = ccm
    dcc_dy, dcc_dx = np.gradient(ccm2)
    d2cc_dx2 = np.gradient(dcc_dx)[1]
    d2cc_dy2 = np.gradient(dcc_dy)[0]
    hes = np.hypot(d2cc_dx2, d2cc_dy2)
    if hes_norm:
        hes = (hes - np.median(hes)) / np.std(hes)
    return hes


def get_distance_to_nearest_keypoint(x1, y1, shape):
    """Exact copy of pmlib.get_distance_to_nearest_keypoint."""
    seed = np.zeros(shape, dtype=bool)
    seed[np.uint16(y1), np.uint16(x1)] = True
    dist = nd.distance_transform_edt(~seed,
                                     return_distances=True,
                                     return_indices=False)
    return dist


def get_initial_rotation(n1, n2):
    """
    Original pmlib.get_initial_rotation uses real geolocation.
    For coregistered images in the same grid, the rotation is ~0.
    We return 0.0 to keep the API. (MOD: simplified)
    """
    return 0.0


def get_template(img, c, r, a, s, rot_order=0, **kwargs):
    """Exact copy of pmlib.get_template."""
    tc = int(s / 2.) + 1
    tc = np.array([tc, tc])
    a = np.radians(a)
    transform = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    offset = np.array([r, c]) - tc.dot(transform)
    t = nd.interpolation.affine_transform(
        img, transform.T, order=rot_order, offset=offset,
        output_shape=(s, s), cval=0.0, output=np.uint8)
    return t


def rotate_and_match(img1, c1, r1, img_size, image2, alpha0,
                     angles=(-3, 0, 3),
                     mtype=cv2.TM_CCOEFF_NORMED,
                     template_matcher=cv2.matchTemplate,
                     mcc_norm=False,
                     **kwargs):
    """Exact copy of pmlib.rotate_and_match."""
    res_shape = [image2.shape[0] - img_size + 1] * 2
    best_r = -np.inf
    for angle in angles:
        template = get_template(img1, c1, r1, angle - alpha0, img_size, **kwargs)
        if ((template.min() == 0) or
                (template.shape[0] < img_size or template.shape[1] < img_size)):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        result = template_matcher(image2, template, mtype)
        ij = np.unravel_index(np.argmax(result), result.shape)

        if result.max() > best_r:
            best_r = result.max()
            best_a = angle
            best_result = result
            best_template = template
            best_ij = ij

    best_h = get_hessian(best_result, **kwargs)[best_ij]
    dr = best_ij[0] - (image2.shape[0] - template.shape[0]) / 2.
    dc = best_ij[1] - (image2.shape[1] - template.shape[1]) / 2.

    if mcc_norm:
        best_r = (best_r - np.median(best_result)) / np.std(best_result)

    return dc, dr, best_a, best_r, best_h, best_result, best_template


def use_mcc(c1, r1, c2fg, r2fg, border, img1, img2, img_size, alpha0, **kwargs):
    """Exact copy of pmlib.use_mcc."""
    hws = int(img_size / 2.)
    image = img2[int(r2fg - hws - border):int(r2fg + hws + border + 1),
                 int(c2fg - hws - border):int(c2fg + hws + border + 1)]
    dc, dr, best_a, best_r, best_h, best_result, best_template = rotate_and_match(
        img1, c1, r1, img_size, image, alpha0, **kwargs)
    c2 = c2fg + dc
    r2 = r2fg + dr
    return c2, r2, best_a, best_r, best_h


# Shared global for multiprocessing, as in original
shared_args = None
shared_kwargs = None


def use_mcc_mp(i):
    """Exact copy of pmlib.use_mcc_mp, using shared_args."""
    global shared_args, shared_kwargs
    c2, r2, a, r, h = use_mcc(shared_args[0][i],
                              shared_args[1][i],
                              shared_args[2][i],
                              shared_args[3][i],
                              shared_args[4][i],
                              shared_args[5],
                              shared_args[6],
                              shared_args[7],
                              shared_args[8],
                              **shared_kwargs)
    if i % 100 == 0:
        print('%02.0f%% %07.1f %07.1f %07.1f %07.1f %+05.1f %04.2f %04.2f' % (
            100 * float(i) / len(shared_args[0]),
            shared_args[0][i], shared_args[1][i], c2, r2, a, r, h), end='\r')
    return c2, r2, a, r, h


def prepare_first_guess(c2pm1, r2pm1, n1, c1, r1, n2, c2, r2, img_size,
                        min_fg_pts=5,
                        min_border=20,
                        max_border=50,
                        old_border=True, **kwargs):
    """Exact copy of pmlib.prepare_first_guess (with FakeNansat geometry)."""
    n2_shape = n2.shape()
    lon1, lat1 = n1.transform_points(c1, r1)
    c1n2, r1n2 = n2.transform_points(lon1, lat1, 1)

    c2p2, r2p2 = np.round(interpolation_poly(c1n2, r1n2, c2, r2, c2pm1, r2pm1, **kwargs))
    c2fg, r2fg = np.round(interpolation_near(c1n2, r1n2, c2, r2, c2pm1, r2pm1, **kwargs))

    if old_border:
        border_img = get_distance_to_nearest_keypoint(c2, r2, n2_shape)
        border = np.zeros(c2pm1.size) + max_border
        gpi = ((c2pm1 >= 0) * (c2pm1 < n2_shape[1]) *
               (r2pm1 >= 0) * (r2pm1 < n2_shape[0]))
        border[gpi] = border_img[np.round(r2pm1[gpi]).astype(np.int16),
                                 np.round(c2pm1[gpi]).astype(np.int16)]
    else:
        c2tst, r2tst = interpolation_poly(c1n2, r1n2, c2, r2, c1n2, r1n2, **kwargs)
        c2dif, r2dif = interpolation_near(c1n2, r1n2,
                                          c2 - c2tst, r2 - r2tst,
                                          c2pm1, r2pm1,
                                          **kwargs)
        border = np.hypot(c2dif, r2dif)

    border[border < min_border] = min_border
    border[border > max_border] = max_border
    border[np.isnan(c2fg)] = max_border
    border = np.floor(border)

    c2fg[np.isnan(c2fg)] = c2p2[np.isnan(c2fg)]
    r2fg[np.isnan(r2fg)] = r2p2[np.isnan(r2fg)]

    return c2fg, r2fg, border


def pattern_matching(lon_pm1, lat_pm1,
                     n1, c1, r1, n2, c2, r2,
                     margin=0,
                     img_size=35,
                     threads=5,
                     srs='+proj=latlong +datum=WGS84 +ellps=WGS84 +no_defs',
                     **kwargs):
    """
    Original pmlib.pattern_matching, slightly adapted:

    - We ignore true SRS; lon_pm1, lat_pm1 are synthetic lon/lat
      from FakeNansat.transform_points of PM grid.
    - drift output is in synthetic coordinates; you will usually
      convert to pixel units afterwards.
    """
    t0 = time.time()
    img1, img2 = n1[1], n2[1]
    dst_shape = lon_pm1.shape

    c2pm1, r2pm1 = n2.transform_points(lon_pm1.flatten(), lat_pm1.flatten(), 1)
    c2pm1i, r2pm1i = np.round([c2pm1, r2pm1])

    lon1i, lat1i = n2.transform_points(c2pm1i, r2pm1i)
    c1pm1i, r1pm1i = n1.transform_points(lon1i, lat1i, 1)

    c2fg, r2fg, brd2 = prepare_first_guess(c2pm1i, r2pm1i, n1, c1, r1, n2, c2, r2, img_size, **kwargs)

    hws = round(img_size / 2) + 1
    hws_hypot = np.hypot(hws, hws)
    gpi = ((c2fg - brd2 - hws - margin > 0) *
           (r2fg - brd2 - hws - margin > 0) *
           (c2fg + brd2 + hws + margin < n2.shape()[1]) *
           (r2fg + brd2 + hws + margin < n2.shape()[0]) *
           (c1pm1i - hws_hypot - margin > 0) *
           (r1pm1i - hws_hypot - margin > 0) *
           (c1pm1i + hws_hypot + margin < n1.shape()[1]) *
           (r1pm1i + hws_hypot + margin < n1.shape()[0]))

    alpha0 = get_initial_rotation(n1, n2)

    def _init_pool(*args):
        global shared_args, shared_kwargs
        shared_args = args[:9]
        shared_kwargs = args[9]

    if threads <= 1:
        _init_pool(c1pm1i[gpi], r1pm1i[gpi], c2fg[gpi], r2fg[gpi],
                   brd2[gpi], img1, img2, img_size, alpha0, kwargs)
        results = [use_mcc_mp(i) for i in range(len(gpi[gpi]))]
    else:
        p = Pool(threads, initializer=_init_pool,
                 initargs=(c1pm1i[gpi], r1pm1i[gpi], c2fg[gpi], r2fg[gpi],
                           brd2[gpi], img1, img2, img_size, alpha0, kwargs))
        results = p.map(use_mcc_mp, range(len(gpi[gpi])))
        p.close()
        p.terminate()
        p.join()
        del p

    print('\n', 'Pattern matching - OK! (%3.0f sec)' % (time.time() - t0))

    if len(results) == 0:
        u = np.zeros(dst_shape) + np.nan
        v = np.zeros(dst_shape) + np.nan
        a = np.zeros(dst_shape) + np.nan
        r = np.zeros(dst_shape) + np.nan
        h = np.zeros(dst_shape) + np.nan
        lon_pm2_grd = np.zeros(dst_shape) + np.nan
        lat_pm2_grd = np.zeros(dst_shape) + np.nan
    else:
        results = np.array(results)
        c2pm2i = results[:, 0]
        r2pm2i = results[:, 1]

        dci, dri = c2pm1 - c2pm1i, r2pm1 - r2pm1i
        c2pm2, r2pm2 = c2pm2i + dci[gpi], r2pm2i + dri[gpi]

        # synthetic coords: xpm1,ypm1,xpm2,ypm2
        xpm1, ypm1 = n2.transform_points(c2pm1, r2pm1, 0)
        xpm1_grd = xpm1.reshape(dst_shape)
        ypm1_grd = ypm1.reshape(dst_shape)

        xpm2, ypm2 = n2.transform_points(c2pm2, r2pm2, 0)
        xpm2_grd = _fill_gpi(dst_shape, gpi, xpm2)
        ypm2_grd = _fill_gpi(dst_shape, gpi, ypm2)

        lon_pm2, lat_pm2 = xpm2, ypm2  # same synthetic coords
        lon_pm2_grd = _fill_gpi(dst_shape, gpi, lon_pm2)
        lat_pm2_grd = _fill_gpi(dst_shape, gpi, lat_pm2)

        u = xpm2_grd - xpm1_grd
        v = ypm2_grd - ypm1_grd

        a = results[:, 2]
        r = results[:, 3]
        h = results[:, 4]
        a = _fill_gpi(dst_shape, gpi, a)
        r = _fill_gpi(dst_shape, gpi, r)
        h = _fill_gpi(dst_shape, gpi, h)

    return u, v, a, r, h, lon_pm2_grd, lat_pm2_grd


# HIGH-LEVEL WRAPPER FOR TIFFS

class SeaIceDriftFromTiff(object):
    """
    Minimal high-level wrapper:
        - uses FakeNansat + original FT/PM
        - takes paths to TIFFs
        - pixel_size_m = 100 by default
    """

    def __init__(self, file1, file2,
                 time1=None, time2=None,
                 pixel_size_m=100.0):
        self.file1 = file1
        self.file2 = file2

        self.n1 = preprocess_tiff_to_fakenansat(
            file1, pixel_size_m=pixel_size_m, timestamp=time1
        )
        self.n2 = preprocess_tiff_to_fakenansat(
            file2, pixel_size_m=pixel_size_m, timestamp=time2
        )

    def get_drift_FT(self, max_speed=0.5, max_drift=8000.0,
                     ratio_test=0.7, psi=200, verbose=False, **kwargs):
        """
        Run original Feature Tracking (FT). Displacement is returned in
        synthetic lon/lat space. Usually you convert to pixels or meters.

        max_drift = 8000 m (F2 = 8 km) from Table 3.

        Using parameters from https://www.mdpi.com/2072-4292/9/3/258 (Korosov & Rampal, 2017)
        """
        x1, y1, x2, y2 = feature_tracking(
            self.n1, self.n2,
            max_speed=max_speed, max_drift=max_drift,
            ratio_test=ratio_test, psi=psi, verbose=verbose, **kwargs
        )

        # convert to pixel displacements
        # use transform_points mapping backwards
        # synthetic lon/lat to pixels:
        lon1, lat1 = self.n1.transform_points(x1, y1, 0)
        lon2, lat2 = self.n2.transform_points(x2, y2, 0)
        c1, r1 = self.n1.transform_points(lon1, lat1, 1)
        c2, r2 = self.n2.transform_points(lon2, lat2, 1)
        du_pix = c2 - c1
        dv_pix = r2 - r1
        return c1, r1, du_pix, dv_pix

    def get_drift_PM(self, grid_step_pix=40,
                     img_size=40,
                     min_border=40,
                     max_border=80,
                     threads=4,
                     **kwargs):
        """
        Run original Pattern Matching (PM) on a regular grid of points
        in image coordinates with spacing grid_step_pix.

        Returns:
            u_pix, v_pix, a_deg, r_mcc, h_hess, pm_cols, pm_rows

        Using parameters from https://www.mdpi.com/2072-4292/9/3/258 (Korosov & Rampal, 2017)
        """
        img = self.n1[1]
        ny, nx = img.shape
        rows = np.arange(0, ny, grid_step_pix)
        cols = np.arange(0, nx, grid_step_pix)
        pm_cols, pm_rows = np.meshgrid(cols, rows)

        # synthetic lon/lat of PM grid
        lon_pm1, lat_pm1 = self.n1.transform_points(pm_cols, pm_rows, 0)

        # get FT in pixel coords to feed PM
        c1_ft, r1_ft, du_ft, dv_ft = self.get_drift_FT()
        # convert FT to synthetic coords
        lon1_ft, lat1_ft = self.n1.transform_points(c1_ft, r1_ft, 0)
        lon2_ft, lat2_ft = self.n2.transform_points(
            c1_ft + du_ft, r1_ft + dv_ft, 0
        )
        # convert back to "pixel" coords for PM interface
        x1_ft, y1_ft = self.n1.transform_points(lon1_ft, lat1_ft, 1)
        x2_ft, y2_ft = self.n2.transform_points(lon2_ft, lat2_ft, 1)

        u_syn, v_syn, a, r, h, _, _ = pattern_matching(
            lon_pm1, lat_pm1,
            self.n1, x1_ft, y1_ft,
            self.n2, x2_ft, y2_ft,
            img_size=img_size,
            threads=threads,
            min_border=min_border,
            max_border=max_border,
            **kwargs
        )

        # convert synthetic displacements back to pixels
        xpm1, ypm1 = self.n1.transform_points(pm_cols, pm_rows, 0)
        xpm2 = xpm1 + u_syn
        ypm2 = ypm1 + v_syn
        cpm2, rpm2 = self.n1.transform_points(xpm2, ypm2, 1)
        u_pix = cpm2 - pm_cols
        v_pix = rpm2 - pm_rows

        return u_pix, v_pix, a, r, h, pm_cols, pm_rows


    def interpolate_to_dense_image_grid(self, pm_cols, pm_rows, u, v, good_mask,
                                    img_shape,
                                    method='linear', fill_method='nearest'):
        """
        Interpolate PM vectors (coarse grid) onto a full-resolution image grid.

        Parameters
        ----------
        pm_cols, pm_rows : 2D arrays
            Coordinates where PM vectors are computed (coarse grid).
        u, v : 2D arrays
            PM horizontal and vertical drift (pixel units).
        good_mask : 2D bool array
            Quality mask (rpm*hpm > threshold).
        img_shape : tuple
            (rows, cols) of the target full-resolution image (e.g., 1024x1024).
        method : str
            'linear' or 'cubic' interpolation
        fill_method : str
            fallback interpolation ('nearest')

        Returns
        -------
        u_dense, v_dense : 2D arrays (full-resolution)
        """
        H, W = img_shape

        # Flatten PM grid
        xx = pm_cols.flatten()
        yy = pm_rows.flatten()
        uu = u.flatten()
        vv = v.flatten()
        mask_flat = good_mask.flatten()

        pts = np.column_stack((xx[mask_flat], yy[mask_flat]))

        # Define full-resolution target grid
        dense_rows, dense_cols = np.meshgrid(
            np.arange(H), np.arange(W), indexing='ij'
        )
        pts_full = np.column_stack((dense_cols.flatten(), dense_rows.flatten()))

        # Interpolate U
        u_lin = griddata(pts, uu[mask_flat], pts_full, method=method)
        if np.any(np.isnan(u_lin)):
            u_fill = griddata(pts, uu[mask_flat], pts_full, method=fill_method)
            u_lin[np.isnan(u_lin)] = u_fill[np.isnan(u_lin)]

        # Interpolate V
        v_lin = griddata(pts, vv[mask_flat], pts_full, method=method)
        if np.any(np.isnan(v_lin)):
            v_fill = griddata(pts, vv[mask_flat], pts_full, method=fill_method)
            v_lin[np.isnan(v_lin)] = v_fill[np.isnan(v_lin)]

        # reshape to full image
        u_dense = u_lin.reshape((H, W))
        v_dense = v_lin.reshape((H, W))

        return u_dense, v_dense
    

def warp_image_with_flow(img, u, v):
    """
    Warp image using flow fields u(x,y), v(x,y).
    
    img : 2D numpy array
    u, v : 2D arrays (same shape)
        Pixel displacements (positive right/down)
    """
    rows, cols = img.shape

    # coordinate grid
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

    # Backward mapping:
    # sample img1 at (r - v, c - u)
    r_src = rr - v
    c_src = cc - u

    # Handle out-of-bounds gracefully
    r_src = np.clip(r_src, 0, rows - 1)
    c_src = np.clip(c_src, 0, cols - 1)

    # warped = map_coordinates(img, [r_src, c_src], order=1, mode='nearest')
    warped = map_coordinates(img.astype(float),
                         [r_src, c_src],
                         order=1,
                         mode='constant',
                         cval=np.nan)
    return warped