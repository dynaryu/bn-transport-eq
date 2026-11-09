"""Monitoring real earthquakes: watch the USGS feed for ShakeMaps near the example networks.

* ``feedwatch`` -- the always-on, feed-only watcher: notes new ShakeMap versions and emails
  new earthquakes. Standard library only (plus boto3 on AWS Lambda).
* ``watch`` -- the local watcher: the same poll, then each affected ShakeMap assembly's
  session, run headless on the new map.
"""
