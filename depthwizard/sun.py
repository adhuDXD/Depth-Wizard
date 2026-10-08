"""Solar position (NOAA algorithm), used when image metadata lacks sun angles."""
from __future__ import annotations

import math
from datetime import datetime, timezone


def sun_position(when: datetime, lat: float, lon: float) -> tuple[float, float]:
    """Return (elevation, azimuth) in degrees; azimuth clockwise from north."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    when = when.astimezone(timezone.utc)
    jd = when.timestamp() / 86400.0 + 2440587.5
    t = (jd - 2451545.0) / 36525.0

    l0 = (280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    mr = math.radians(m)
    c = (math.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
         + math.sin(2 * mr) * (0.019993 - 0.000101 * t) + math.sin(3 * mr) * 0.000289)
    true_long = l0 + c
    omega = 125.04 - 1934.136 * t
    app_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    eps0 = 23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60) / 60
    eps = math.radians(eps0 + 0.00256 * math.cos(math.radians(omega)))
    decl = math.asin(math.sin(eps) * math.sin(math.radians(app_long)))

    y = math.tan(eps / 2) ** 2
    l0r = math.radians(l0)
    eq_time = 4 * math.degrees(
        y * math.sin(2 * l0r) - 2 * e * math.sin(mr) + 4 * e * y * math.sin(mr) * math.cos(2 * l0r)
        - 0.5 * y * y * math.sin(4 * l0r) - 1.25 * e * e * math.sin(2 * mr)
    )
    minutes = when.hour * 60 + when.minute + when.second / 60
    true_solar = (minutes + eq_time + 4 * lon) % 1440
    hour_angle = true_solar / 4 - 180
    ha = math.radians(hour_angle)
    latr = math.radians(lat)

    cos_zen = math.sin(latr) * math.sin(decl) + math.cos(latr) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    elevation = 90 - math.degrees(zen)
    az = math.degrees(math.atan2(
        math.sin(ha), math.cos(ha) * math.sin(latr) - math.tan(decl) * math.cos(latr)
    )) + 180
    return elevation, az % 360
