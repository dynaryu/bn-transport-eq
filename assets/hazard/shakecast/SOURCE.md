# Source

**STATUS: source decided and implemented in the forward pilot; not yet a registry asset.**
There is no `asset.ttl`, `model.py` or `tests/` here yet. The working code is
`src/forward_pilot/shakemap.py`, with tests in `src/forward_pilot/tests/test_shakemap.py`.

## Which ShakeCast

**A USGS ShakeMap product consumed as a grid** — specifically `grid.xml` (median
`PSA10`, in %g) and `uncertainty.xml` (`STDPSA10`, ln units) for one event. It can
come from either of two places, and both are read the same way:

- **USGS ComCat**: the FDSN event service at
  `https://earthquake.usgs.gov/fdsnws/event/1/query`, using
  `producttype=shakemap` to search and `eventid=` to fetch. The preferred
  ShakeMap product is downloaded and cached.
- **A local USGS ShakeCast v4 server**: its data directory,
  `$SHAKECAST_USER_DIRECTORY/data/<event_id>/<event_id>-<version>/`, which
  defaults to `~/.shakecast`. ShakeCast downloads `grid.xml` and, with its default
  `eq_pref_products`, `uncertainty.xml`. The pilot's ComCat cache uses the same
  layout. Source: `usgs/shakecast`, `shakecast/app/orm/objects.py`
  (`ShakeMap.directory_name`) and `shakecast/app/util.py` (`get_data_dir`). The
  repository was archived by its owners on 2024-06-10, so ShakeCast v4 is no
  longer developed.

**Consequence for the BN:** this is a forward hazard factor, not only evidence.
For a given event it sets each bridge site's `SA1 ~ LogNormal(ln median, sigma)`
from the grid, which is the same node the GMPE path parameterises. So it can
`SampleForward` and `EvaluateDensity` per site. It does **not** supply the
spatial correlation of residuals between sites, so sites are independent given
the map. That matches the GMPE path, and the `EventResidual/SpatialResidual -> SA1`
extension point is still where correlation would go. The ShakeMap's own
conditioning on stations is already in its median and sigma. Treating station
recordings as BN evidence instead (an L2 role) is not done.

## Record

- **Citation:** Worden, C.B., Thompson, E.M., Hearne, M., and Wald, D.J. (2020).
  *ShakeMap Manual Online: technical manual, user's guide, and software guide.*
  U.S. Geological Survey. https://doi.org/10.5066/F7D21VPQ. Historical events come
  from the ShakeMap Atlas: Garcia, D., et al. (2012), *ShakeMap Atlas 2.0: an
  improved suite of recent historical earthquake ShakeMaps for global hazard
  analyses and loss model calibration*, 15th World Conference on Earthquake
  Engineering.
- **Retrieved:** 2026-10-12 (format checked on `iscgem849480`, Xingtai 1966
  M6.8, ShakeMap 4.0.2, version 1).
- **Development background:** USGS National Earthquake Information Center. Ground
  motion is interpolated from seismic stations and macroseismic reports where
  they exist, and from GMPEs otherwise. Atlas events are reconstructions made
  after the fact.
- **Licence:** USGS data products are U.S. Government works in the public domain.
  Grids are not stored in this repository anyway: they are about 30 MB each and
  re-fetched on demand. The cache is `experiments/forward-bn-pilot/shakemaps/`,
  which is gitignored.
- **Limitations (stated by USGS, and checked here):**
  - For most historical events in China, the Atlas maps carry no station records.
    `seismic_stations="0"` for Xingtai 1966; Tangshan 1976 has 6 stations and 84
    felt reports. Their medians are therefore GMPE estimates, with correspondingly
    large `STDPSA10` (about 0.7 ln units at Xingtai).
  - Older ShakeMap versions may have no `uncertainty.xml`. The pilot then uses
    sigma = 0 and warns that failure probabilities are understated.
  - The map extent is finite. A bridge outside it is refused rather than given
    zero shaking.
