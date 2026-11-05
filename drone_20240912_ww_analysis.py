import numpy as np, geopandas as gpd, rasterio, pandas as pd
import matplotlib.pyplot as plt
from rasterio import Affine
from rasterio.features import geometry_mask
from rasterio.mask import mask
from rasterio.plot import show
from scipy import stats
from sklearn.cluster import KMeans

path = r'/Users/kennychen/Desktop/R sydney/drone_run_20240912_rerun.tif'
gdir = r'/Users/kennychen/Desktop/gee_upload/'

scale = 5
with rasterio.open(path) as src:
    out_shape = (src.height // scale, src.width // scale)
    rs = rasterio.enums.Resampling.average
    rgb = src.read([3, 2, 1], out_shape=(3, *out_shape), resampling=rs).astype("float32")
    alpha = src.read(7, out_shape=out_shape, resampling=rs)
    raster_crs = src.crs
    scaled_tf = src.transform * Affine.scale(src.width / out_shape[1],
                                             src.height / out_shape[0])

def stretch(b, low=2, high=98):
    lo, hi = np.nanpercentile(b, [low, high])
    return np.clip((b - lo) / (hi - lo), 0, 1)

# Weir West = control + treatment thirds together
ctrl = gpd.read_file(gdir + 'control_thirds.shp')
trt  = gpd.read_file(gdir + 'treatment_thirds.shp')
paddock = gpd.GeoDataFrame(
    geometry=[pd.concat([ctrl.geometry, trt.geometry]).union_all()],   # old pandas: .unary_union
    crs=ctrl.crs).to_crs(raster_crs)
# if the outline looks wrong, try the single polygon instead:
# paddock = gpd.read_file(gdir + 'weir_west_tc.shp').to_crs(raster_crs)

# 1) check: outline on the full mosaic
foot = alpha > 32767
rgb_full = rgb.copy(); rgb_full[:, ~foot] = np.nan
img = np.nan_to_num(np.dstack([stretch(b) for b in rgb_full]), nan=1.0)

inv = ~scaled_tf
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(img)
for geom in paddock.geometry:
    xs, ys = geom.exterior.xy
    c_, r_ = zip(*[inv * (x, y) for x, y in zip(xs, ys)])
    ax.plot(c_, r_, "r-", lw=2)
ax.axis("off"); plt.show()

# 2) clip
in_poly = geometry_mask(paddock.geometry, out_shape=out_shape,
                        transform=scaled_tf, invert=True)
valid = foot & in_poly
rgb[:, ~valid] = np.nan
transform = scaled_tf            # used by the 10 m uniformity block

rgb_img = np.nan_to_num(np.dstack([stretch(b) for b in rgb]), nan=1.0)   # no crop: keeps the full grid
plt.figure(figsize=(12, 8)); plt.imshow(rgb_img)
plt.title("Weir West, clipped (flight 2024-09-12)"); plt.axis("off"); plt.show()

with rasterio.open(path) as src:
    red = src.read(3, out_shape=out_shape, resampling=rs).astype("float32")
    nir = src.read(5, out_shape=out_shape, resampling=rs).astype("float32")
    rededge = src.read(4, out_shape=out_shape, resampling=rs).astype("float32")

ndvi = (nir - red) / (nir + red)
ndre = (nir - rededge) / (nir + rededge)

ndvi[~valid] = np.nan

#plot ndvi
plt.figure(figsize=(12, 8))
plt.imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.9)
plt.colorbar(label="NDVI")
plt.axis("off")
plt.show()

# histogram ndvi
plt.hist(ndvi[valid].ravel(), bins=100)
plt.xlabel("NDVI")
plt.show()



## clustering on NDVI
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.stats import kruskal
from sklearn.metrics import silhouette_score

K = 3   # number of clusters

# --- field cuts (clean CSV with coordinates) ---------------------------------
CUTS_CSV = r'/Users/kennychen/Desktop/R sydney/biomass_cuts_clean.csv'

def load_cuts(site="WW", date="2024-09-12"):
    """Cuts for one site/date from biomass_cuts_clean.csv.
    Returns: name, biomass (g/m2, dry), water_g_m2 (wet - dry, g/m2), water_frac (wet - dry)/wet, northing, easting (EPSG:7855)."""
    c = pd.read_csv(CUTS_CSV)
    c["cut_date"] = pd.to_datetime(c["cut_date"], dayfirst=True)   # dd/mm/yyyy
    c = c[(c.site == site) & (c.cut_date == date)].copy()
    c = c.rename(columns={"cut_id": "name", "biomass_g_m2": "biomass"})
    wet_g = c.wet_plus_bag_g - c.wet_bag_g                       # fresh sample weight (g)
    dry_g = c.dry_plus_bag_g - c.dry_bag_g                       # dry sample weight (g)
    c["water_g_m2"] = (wet_g - dry_g) / c.quadrat_m2             # NEW: water per m2 (wet - dry)
    c["water_frac"] = (wet_g - dry_g) / wet_g                    # NEW: fraction of fresh weight that is water
    c = c.dropna(subset=["name", "biomass", "water_g_m2", "water_frac", "easting", "northing"])
    return c[["name", "biomass", "water_g_m2", "water_frac", "northing", "easting"]].reset_index(drop=True)
# -----------------------------------------------------------------------------

ndre[~valid] = np.nan
cvalid = valid & np.isfinite(ndvi) & np.isfinite(ndre)   # drop any divide-by-zero pixels

stack = np.stack([ndvi, ndre], axis=-1)
flat = stack[cvalid]

km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(flat)

# renumber clusters from lowest to highest NDVI, so 0 = least green
order = np.argsort(km.cluster_centers_[:, 0])
remap = np.empty(K, dtype=int)
remap[order] = np.arange(K)
labels = remap[km.labels_]

clustered = np.full(valid.shape, np.nan)
clustered[cvalid] = labels

summary = pd.DataFrame(flat, columns=["NDVI", "NDRE"])
summary["cluster"] = labels

print(summary.groupby("cluster").mean().round(3))
print(summary.groupby("cluster").size())

palette = {3: ["#a6611a", "#dfc27d", "#018571"],
           5: ["#a6611a", "#dfc27d", "#f5f5f5", "#80cdc1", "#018571"]}
colors = palette.get(K, plt.cm.BrBG(np.linspace(0.1, 0.9, K)).tolist())
cmapK = ListedColormap(colors)
normK = BoundaryNorm(np.arange(-0.5, K + 0.5, 1), cmapK.N)
ndvi_by_cluster = summary.groupby("cluster").NDVI.mean()

plt.figure(figsize=(12, 8))
plt.imshow(clustered, cmap=cmapK, norm=normK)
handles = [Patch(facecolor=colors[k], edgecolor="grey",
                 label=f"cluster {k}  (NDVI {ndvi_by_cluster[k]:.2f})") for k in range(K)]
plt.legend(handles=handles, loc="lower left", fontsize=8)
plt.axis("off")
plt.show()






# how mixed is each 10 m square? (aligned to the Sentinel-2 grid)
from rasterio.warp import reproject, transform_bounds, Resampling
from rasterio.transform import from_origin



BLOCK_M = 10
DST_CRS = "EPSG:32755"          # WGS84 / UTM zone 55S — same grid as Sentinel-2

with rasterio.open(path) as src:
    src_crs = src.crs
    left, bottom, right, top = transform_bounds(src_crs, DST_CRS, *src.bounds)

# pick a pixel size that fits exactly in 10 m, as close as possible to the drone pixel size
with rasterio.open(path) as src:
    if src_crs.is_geographic:
        lat = (src.bounds.top + src.bounds.bottom) / 2
        approx = abs(src.res[0]) * scale * 111320 * np.cos(np.radians(lat))
    else:
        approx = abs(src.res[0]) * scale

n = max(1, round(BLOCK_M / approx))      # pixels per 10 m
res = BLOCK_M / n
print(f"drone pixel ≈ {approx:.2f} m  ->  {res:.2f} m used, {n} x {n} pixels per 10 m square")

# start the grid at a multiple of 10 m so it coincides with Sentinel-2
x0 = np.floor(left / BLOCK_M) * BLOCK_M
y1 = np.ceil(top / BLOCK_M) * BLOCK_M
W2 = int(np.ceil((right - x0) / BLOCK_M))       # number of 10 m squares
H2 = int(np.ceil((y1 - bottom) / BLOCK_M))
dst_transform = from_origin(x0, y1, res, res)

dst = np.full((H2 * n, W2 * n), np.nan, dtype="float32")
reproject(
    source=clustered.astype("float32"), destination=dst,
    src_transform=transform, src_crs=src_crs,
    dst_transform=dst_transform, dst_crs=DST_CRS,
    src_nodata=np.nan, dst_nodata=np.nan,
    resampling=Resampling.nearest,              # classes must not be averaged
)

blocks = dst.reshape(H2, n, W2, n).transpose(0, 2, 1, 3).reshape(H2, W2, n * n)

counts = np.stack([(blocks == k).sum(-1) for k in range(K)], axis=-1)
nvalid = counts.sum(-1)
full = nvalid >= 0.8 * n * n                    # only squares that mostly fall inside the flight

dominance = np.full((H2, W2), np.nan)
dominance[full] = counts[full].max(-1) / nvalid[full]

d = dominance[full]
print(f"number of 10 m squares: {full.sum()}")
print(f"mean share of the largest class: {d.mean() * 100:.0f}%")
print(f"uniform (>90% one class): {np.mean(d > 0.9) * 100:.0f}% of squares")
print(f"mixed (<60% one class): {np.mean(d < 0.6) * 100:.0f}% of squares")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
im = axes[0].imshow(dominance, cmap="RdYlGn", vmin=1 / K, vmax=1)
fig.colorbar(im, ax=axes[0], label="share of largest class")
axes[0].set_title("How uniform is each 10 m Sentinel-2 pixel? (green = uniform, red = mixed)")
axes[0].axis("off")
axes[1].hist(d, bins=30)
axes[1].set_xlabel("share of largest class per 10 m square")
axes[1].set_ylabel("number of squares")
plt.tight_layout()
plt.show()

dominant = np.full((H2, W2), np.nan)
dominant[full] = counts[full].argmax(-1)

res_tab = pd.DataFrame({"class": dominant[full], "share": dominance[full]})
print(res_tab.groupby("class").share.agg(["mean", "count"]).round(2))

# are the cut locations in uniform or mixed 10 m squares?
from rasterio.transform import rowcol

cuts = load_cuts("WW", "2024-09-12")        # Weir West cuts, drone flight 2024-09-12

cg = gpd.GeoDataFrame(
    cuts, geometry=gpd.points_from_xy(cuts.easting, cuts.northing), crs="EPSG:7855"
).to_crs(DST_CRS)

blk_transform = from_origin(x0, y1, BLOCK_M, BLOCK_M)
rr, cc = rowcol(blk_transform, cg.geometry.x.values, cg.geometry.y.values)
rr, cc = np.asarray(rr), np.asarray(cc)

ok = (rr >= 0) & (rr < H2) & (cc >= 0) & (cc < W2)
cg = cg[ok].copy()
cg["share"] = dominance[rr[ok], cc[ok]]
cg["class"] = dominant[rr[ok], cc[ok]]
cg = cg.dropna(subset=["share"])

print(cg[["name", "biomass", "water_frac", "class", "share"]].round(2).to_string())
print(f"\nmean share at the cut locations: {cg.share.mean() * 100:.0f}%")
print(f"mean share over the whole paddock: {d.mean() * 100:.0f}%")

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(dominance, cmap="RdYlGn", vmin=1 / K, vmax=1,
               extent=[x0, x0 + W2 * BLOCK_M, y1 - H2 * BLOCK_M, y1])
fig.colorbar(im, ax=ax, label="share of largest class")
cg.plot(ax=ax, color="black", edgecolor="white", markersize=60)
for _, r in cg.iterrows():
    ax.annotate(r["name"], (r.geometry.x, r.geometry.y), fontsize=7,
                xytext=(4, 4), textcoords="offset points")
ax.set_title("Cut locations on the uniformity map (green = uniform, red = mixed)")
plt.show()









# regression  (Weir West cuts 2024-09-12 vs drone flight 2024-09-12)
bm = load_cuts("WW", "2024-09-12")
print(bm)

gdf = gpd.GeoDataFrame(
    bm, geometry=gpd.points_from_xy(bm.easting, bm.northing), crs="EPSG:7855"
)

with rasterio.open(path) as src:
    raster_crs = src.crs
    b = src.bounds
    pts = gdf.to_crs(raster_crs)
    inside = pts.geometry.apply(
        lambda p: b.left <= p.x <= b.right and b.bottom <= p.y <= b.top
    )
print(f"{inside.sum()} of {len(gdf)} samples fall inside the raster")
gdf = gdf[inside.values].copy()

buf = gdf.geometry.buffer(0.564).to_crs(raster_crs)

means, npix = [], []
with rasterio.open(path) as src:
    for geom in buf:
        arr, _ = mask(src, [geom], crop=True, filled=False, indexes=[3, 4, 5])
        means.append([arr[i].mean() for i in range(3)])
        npix.append(arr[0].count())

gdf[["Red", "RedEdge", "NIR"]] = np.array(means)
gdf["npix"] = npix
gdf["NDVI"] = (gdf.NIR - gdf.Red) / (gdf.NIR + gdf.Red)
gdf["NDRE"] = (gdf.NIR - gdf.RedEdge) / (gdf.NIR + gdf.RedEdge)

print(gdf[["name", "biomass", "water_g_m2", "water_frac", "npix", "NDVI", "NDRE"]].round(3))

# NDVI vs dry biomass AND vs water content
targets = {"biomass": "dry biomass (g/m²)",
           "water_frac": "water fraction (wet−dry)/wet",
           "water_g_m2": "water (g/m²)"}

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
for ax_, (col, lab) in zip(axes, targets.items()):
    d = gdf.dropna(subset=["NDVI", col])
    r = stats.linregress(d.NDVI, d[col])
    print(f"NDVI vs {col:10s}: R2={r.rvalue**2:.2f}  slope={r.slope:+.2f}  p={r.pvalue:.3f}  n={len(d)}")
    ax_.scatter(d.NDVI, d[col])
    x = np.linspace(d.NDVI.min(), d.NDVI.max(), 10)
    ax_.plot(x, r.intercept + r.slope * x)
    ax_.set_xlabel("NDVI"); ax_.set_ylabel(lab)
    ax_.set_title(f"NDVI vs {col} — R²={r.rvalue**2:.2f}")
plt.tight_layout()
plt.show()

# testing and verifying
for radius in [0.564, 1, 2, 3, 5, 10]:
    buf = gdf.geometry.buffer(radius).to_crs(raster_crs)
    vals = []
    with rasterio.open(path) as src:
        for geom in buf:
            arr, _ = mask(src, [geom], crop=True, filled=False, indexes=[3, 5])
            r_, n_ = arr[0].mean(), arr[1].mean()
            vals.append((n_ - r_) / (n_ + r_))
    out = []
    for col in ["biomass", "water_frac", "water_g_m2"]:
        dd = pd.DataFrame({"ndvi": vals, "y": gdf[col].values}).dropna()
        res = stats.linregress(dd.ndvi, dd.y)
        out.append(f"{col}: R2={res.rvalue**2:.2f} p={res.pvalue:.3f}")
    print(f"r={radius:>5}m  " + "   ".join(out))


print(gdf.NDVI.describe())
print(gdf.biomass.describe())
print(gdf.water_frac.describe())
print(gdf.water_g_m2.describe())
print(gdf.loc[gdf.NDVI.idxmax(), ["name", "biomass", "water_frac", "NDVI", "easting", "northing"]])