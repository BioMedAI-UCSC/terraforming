#!/usr/bin/env python3
"""
Parse MARSIS radargram XML labels to find orbits with polar coverage.
For orbits we don't have locally, we'll need to download their XMLs first.

This script reads the collection inventory and outputs recommended orbits
for: south polar cap, north polar cap, and equatorial regions.
"""

import re
from pathlib import Path

# Extract unique orbit numbers from the collection inventory
COLLECTION_CSV = Path(__file__).parent.parent / "data" / "marsis" / "collection_radargram_data.csv"

def extract_orbit_numbers():
    """Extract unique orbit numbers from the collection CSV."""
    orbits = set()
    with open(COLLECTION_CSV) as f:
        for line in f:
            # Format: P,urn:nasa:pds:mex_marsis_optim:radargram_data:o_01867_chapman_no-wind_f::1.0
            match = re.search(r':o_(\d+)_', line)
            if match:
                orbits.add(int(match.group(1)))
    return sorted(orbits)

def main():
    orbits = extract_orbit_numbers()
    print(f"Total unique orbits in collection: {len(orbits)}")
    print(f"Orbit range: {min(orbits)} - {max(orbits)}")
    print()
    
    # Mars Express orbit geometry is roughly:
    # - Periapsis near south pole in early mission (orbits ~1800-5000)
    # - Orbit precesses over time
    # 
    # Based on published MARSIS papers, good polar coverage orbits:
    # South Polar Layered Deposits (SPLD): orbits 1855-2500, 4000-5000
    # North Polar Layered Deposits (NPLD): orbits 3000-4000, 5500-7000
    #
    # For baseline, recommend:
    # - 5 orbits over south polar cap
    # - 5 orbits over north polar cap  
    # - 2 orbits near equator
    
    # Filter orbits in our collection that are likely polar
    south_polar_candidates = [o for o in orbits if 1850 <= o <= 2500 or 4000 <= o <= 5000]
    north_polar_candidates = [o for o in orbits if 3000 <= o <= 4000 or 5500 <= o <= 7000]
    equatorial_candidates = [o for o in orbits if 8000 <= o <= 10000]  # Later mission, varied geometry
    
    print("=== RECOMMENDED ORBITS FOR BASELINE ===")
    print()
    print("South Polar Cap (SPLD) - sample 5 orbits:")
    south_sample = south_polar_candidates[:5] if len(south_polar_candidates) >= 5 else south_polar_candidates
    for o in south_sample:
        folder = f"{o // 100:03d}xx"
        print(f"  o_{o:05d}  (folder: {folder})")
    
    print()
    print("North Polar Cap (NPLD) - sample 5 orbits:")
    north_sample = north_polar_candidates[:5] if len(north_polar_candidates) >= 5 else north_polar_candidates
    for o in north_sample:
        folder = f"{o // 100:03d}xx"
        print(f"  o_{o:05d}  (folder: {folder})")
    
    print()
    print("Equatorial region - sample 2 orbits:")
    eq_sample = equatorial_candidates[:2] if len(equatorial_candidates) >= 2 else equatorial_candidates
    for o in eq_sample:
        folder = f"{o // 100:03d}xx"
        print(f"  o_{o:05d}  (folder: {folder})")
    
    print()
    print("=== DOWNLOAD COMMANDS ===")
    print("(Download only the 'optim_wind' variant to save space)")
    print()
    
    all_recommended = south_sample + north_sample + eq_sample
    base_url = "https://pds-geosciences.wustl.edu/mex/urn-nasa-pds-mex_marsis_optim/radargram_data"
    
    for o in all_recommended:
        folder = f"{o // 100:03d}xx"
        orbit_dir = f"o_{o:05d}"
        print(f"# Orbit {o}")
        print(f"wget --no-check-certificate -P data/marsis/radargram_data/{folder}/{orbit_dir}/ \\")
        print(f"  {base_url}/{folder}/{orbit_dir}/{orbit_dir}_optim_wind_f.img \\")
        print(f"  {base_url}/{folder}/{orbit_dir}/{orbit_dir}_optim_wind_f.xml")
        print()

if __name__ == "__main__":
    main()


