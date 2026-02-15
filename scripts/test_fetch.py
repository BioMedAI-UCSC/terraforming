#!/usr/bin/env python3
import ssl
import urllib.request

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

url = "https://pds-geosciences.wustl.edu/mex/urn-nasa-pds-mex_marsis_optim/radargram_data/"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

with urllib.request.urlopen(req, context=ssl_ctx, timeout=60) as resp:
    html = resp.read().decode("utf-8", errors="ignore")
    print(f"Length: {len(html)}")
    print("---")
    print(html[:3000])


