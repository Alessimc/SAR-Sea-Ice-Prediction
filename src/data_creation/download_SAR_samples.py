import subprocess
from sentinelhub import CRS, BBox, bbox_to_dimensions
from datetime import datetime, timedelta, timezone
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from shapely.geometry import shape, box
from shapely.ops import unary_union
import yaml
import argparse
import os
from src.utils import init_logging
import xarray as xr
import pandas as pd
import numpy as np
import requests
from http.client import RemoteDisconnected
from requests.exceptions import ConnectionError, Timeout
import time

MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds


logger = init_logging()

# Parse command-line args
parser = argparse.ArgumentParser(description="Download SAR data in time intervals.")
parser.add_argument("--start", required=True, help="Start datetime in format YYYY-MM-DDTHH:MM")
parser.add_argument("--end", required=True, help="End datetime in format YYYY-MM-DDTHH:MM")
parser.add_argument("--bbox", required=True, help="Bounding box as lon_min,lat_min,lon_max,lat_max")
parser.add_argument("--client_nr", required=False, help="OAuth Client nr")
parser.add_argument(
    "--oauth_config",
    default="configs/copernicus_OAuth.yaml",
    help="Path to Copernicus OAuth credentials YAML (default: configs/copernicus_OAuth.yaml)",
)
parser.add_argument(
    "--data_paths_config",
    default="configs/data_paths.yaml",
    help="Path to data paths YAML (default: configs/data_paths.yaml)",
)
args = parser.parse_args()

with open(args.oauth_config, "r") as f:
    config = yaml.safe_load(f)

with open(args.data_paths_config, "r") as f:
    path_config = yaml.safe_load(f)

target_data_path = path_config.get("SAR_sea_ice_dataset")


# Extract values
CLIENT_ID = config.get(f"client_id{args.client_nr}")
CLIENT_SECRET = config.get(f"client_secret{args.client_nr}")

logger.info(f"Using OAuth Client nr: {args.client_nr}")

# create the oauth session to retrieve the metadata
client = BackendApplicationClient(client_id=CLIENT_ID)
oauth = OAuth2Session(client=client)

def sentinelhub_compliance_hook(response):
    response.raise_for_status()
    return response

oauth.register_compliance_hook("access_token_response", sentinelhub_compliance_hook)


# the time interval on which to query converted to timezone-aware datetimes
datetime_start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
datetime_end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

logger.info(f"Downloading from {datetime_start} to {datetime_end}")


# Region of interest
bounding_box = [float(x) for x in args.bbox.split(",")]
lon_min, lat_min, lon_max, lat_max = bounding_box
logger.info(f"Bounding box: {bounding_box}")
region = f"region-{lon_min}-{lat_min}-{lon_max}-{lat_max}".replace('.', '_')


# Sea ice chart parameters
SIC_DIR = path_config.get("Ice_Chart_data")

logger.info(f"Loading sea ice chart data from {SIC_DIR}")
ds_sic = xr.open_dataset(SIC_DIR)
lon_sic_grid = ds_sic["lon"].values
lat_sic_grid = ds_sic["lat"].values

box_mask = (lon_sic_grid >= lon_min) & (lon_sic_grid <= lon_max) & (lat_sic_grid >= lat_min) & (lat_sic_grid <= lat_max)


# loop over time intervals to avoid too large queries
increment = timedelta(days=10)

current_start = datetime_start
while current_start < datetime_end:
    current_end = min(current_start + increment, datetime_end)
    logger.info(f"Querying data from {current_start} to {current_end}")

    # Get token for the session; note that this expires - if you are slow, may need to ask for a new one
    _ = oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                            client_secret=CLIENT_SECRET, include_client_id=True)


    # prepare the request to the catalog for specific data
    # for information about the collections that can be queried: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data.html#listing-Sentinelhub_data-page=1
    data_query_params = {
        "bbox": bounding_box,
        "datetime": f"{current_start.isoformat(timespec='seconds')}/{current_end.isoformat(timespec='seconds')}",
        "collections": ["sentinel-1-grd"],
        "limit": 100 # max nr of results to return
    }

    evalscript_raw = """
    //VERSION=3
    function setup() {
    return {
        input: ["HH", "HV", "localIncidenceAngle", "dataMask"],
        output: { bands: 4,
                sampleType: "FLOAT32"}
    };
    }

    function evaluatePixel(sample) {
    return [sample.HH, sample.HV, sample.localIncidenceAngle, sample.dataMask];
    }
    """

    desired_resolution_m = 100
    pixels = bbox_to_dimensions(BBox(bbox=bounding_box, crs=CRS.WGS84), desired_resolution_m)

    # actually perform the request
    url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
    response = oauth.post(url, json=data_query_params)

    if response.status_code == 200:
        response_data = response.json()
    else:
        raise RuntimeError(f"got error {response.status_code} on json query:\n {data_query_params}")


    features = response_data.get('features', [])
    sorted_features = sorted(features, key=lambda f: f['properties']['datetime'])
    aoi = box(*bounding_box)



    # group features by time
    times = [datetime.fromisoformat(f["properties"]["datetime"].replace("Z", "+00:00")) 
            for f in sorted_features]

    groups, current_group = [], [0]
    for i, t in enumerate(times[1:], start=1):
        if (t - times[current_group[0]]) <= timedelta(hours=2):
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    groups.append(current_group)

    logger.info(f"Found {len(groups)} time groups")

    # filter groups by AOI coverage
    kept_features = []
    for g in groups:
        polys = [shape(sorted_features[i]["geometry"]) for i in g]
        union_poly = unary_union(polys)

        inter_area = union_poly.intersection(aoi).area
        coverage = inter_area / aoi.area

        if coverage >= 0.9: # threshold for keeping the group now 90% coverage
            last_feature = sorted_features[g[-1]]  # keep last one in group
            # Get SIC for the date of the last feature
            date = last_feature['properties']['datetime'].split("T")[0]
            ds_sel = ds_sic.sel(time=pd.Timestamp(date), method="nearest")
            sic_t = ds_sel["sic"].values.astype(float)
            # Compute percentage of pixels >25% SIC using ice charts: https://documentation.marine.copernicus.eu/PUM/CMEMS-SI-PUM-011-002.pdf
            # only including 'Open drift ice' and closer categories.
            box_values = sic_t[box_mask]
            valid = ~np.isnan(box_values)
            percent_above_25 = np.sum(box_values[valid] > 25) / valid.sum() * 100

            if percent_above_25 >= 100:  # threshold for keeping based on SIC, used to be 95
                kept_features.append(last_feature)
                logger.info(f"Group {g}: coverage {coverage:.1%}, SIC>25%: {percent_above_25:.1f}% → kept {last_feature['properties']['datetime']}")
            else:
                logger.info(f"Group {g}: coverage {coverage:.1%}, SIC>25%: {percent_above_25:.1f}% → skipped (not enough ice)")
        else:
            logger.info(f"Group {g}: coverage {coverage:.1%} → skipped (not enough coverage)")


    # result
    logger.info("\nFinal filtered features:")
    for f in kept_features:
        logger.info(f["properties"]["datetime"])


    for i in range(len(kept_features)):    
        acq_time = datetime.fromisoformat(
            kept_features[i]["properties"]["datetime"].replace("Z", "+00:00")
        )

        # set window that includes retrivals 2 hours before acquisition time
        time_from = (acq_time - timedelta(hours=2)).isoformat()
        time_to   = (acq_time + timedelta(hours=0)).isoformat()

        processing_payload = {
            "input": {
                "bounds": {
                    "bbox": bounding_box,
                },
                "data": [
                    {
                        "type": "sentinel-1-grd",
                        "dataFilter": {
                            "acquisitionMode": "EW",
                            "timeRange": {
                                "from": time_from,
                                "to": time_to
                            },
                        },
                        "processing": {
                            "orthorectify": True,
                            "output": {
                                "resampling": "BILINEAR"
                            },
                            # this controls how overlapping acquisitions are combined
                            "mosaicking": "MOST_RECENT"  # keeps the most recent at each pixel
                        }
                    }
                ]
            },
            "output": {
                "width": pixels[0],
                "height": pixels[1],
                "responses": [
                    {
                        "identifier": "default",
                        "format": {
                            "type": "image/tiff",
                            "parameters": {
                                "compression": "DEFLATE"
                            }
                        }
                    }
                ]
            },
            "evalscript": evalscript_raw
        }

        # create save path
        year, month, day = kept_features[i]["properties"]["datetime"].split("T")[0].split("-")
        save_path = f"{target_data_path}/{region}/{year}/{month}/{day}"
        subprocess.run(["mkdir", "-p", save_path])

        # build file name
        datehour, minutes = kept_features[i]["properties"]["datetime"].split(":")[0:2]
        file_name = datehour.replace("-", "") + minutes
        file_path = os.path.join(save_path, f"{file_name}.tiff")

        # skip if file already exists
        if os.path.exists(file_path):
            logger.info(f"File {file_path} already exists, skipping download.")
            continue

        # only download if not existing
        url = "https://sh.dataspace.copernicus.eu/api/v1/process"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = oauth.post(url, json=processing_payload, timeout=120)
                break  # success — exit retry loop

            except (RemoteDisconnected, ConnectionError, Timeout) as e:
                logger.warning(f"! Connection issue on attempt {attempt}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES:
                    logger.info(f"...Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"!!! Failed after {MAX_RETRIES} attempts. Skipping {file_name}.")
                    response = None

        if response and response.status_code == 200:
            with open(f'{save_path}/{file_name}.tiff', 'wb') as file:
                file.write(response.content)
            logger.info(f"SAR retrieval saved to '{save_path}/{file_name}.tiff'.")
        else:
            if response:
                logger.warning(f"Request for {file_name} failed with status {response.status_code}")
            # Some errors (like 502) don't return JSON
            try:
                if response is not None:
                    logger.warning(f"Error response: {response.json()}")
            except Exception:
                logger.warning("No JSON body returned.")
            logger.info(f"Skipping {file_name} and moving on.")

    # Update the start time for the next iteration
    current_start = current_end

