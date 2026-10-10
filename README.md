# GeoAI RoadGuard — Road Hazard Detection & Mapping

**GEOAI 03 | Computer Vision · GPS · GIS**

GeoAI RoadGuard is a Python application that processes dashcam or phone video alongside a timestamped GPS trajectory to identify potential road hazards and present the results as geospatial data. It includes a Streamlit interface, video annotation, hazard tables, GeoJSON/CSV export, and an interactive Leaflet-based map.

> **Important:** This is a decision-support prototype, not a certified emergency-response system. Geolocation is estimated from GPS, camera settings, and image geometry; validate results against known locations before operational use. Detection quality depends on the chosen model and its training data. The bundled colour-based detector is only for the synthetic demo clip and is not a real-world AI detector.

## Features

- Accepts MP4 video and timestamped GPX/CSV GPS trajectories.
- Supports video-only processing and map-only viewing of GPS/hazard CSV or GPX data.
- Offers a pretrained open-vocabulary detector option, a pothole model option, custom Ultralytics `.pt` weights, and a synthetic demo detector.
- Interpolates GPS positions against video timestamps and estimates vehicle heading.
- Projects detections to estimated latitude/longitude using camera geometry.
- Groups repeated detections and assigns hazard categories, confidence scores, and severity labels.
- Exports GeoJSON and CSV; produces an annotated MP4 with audio alerts.
- Displays hazard features and route information in an interactive map.

## Example workflow

1. Start the Streamlit app.
2. Choose **Video + GPS**, **Video only**, or **GPS / hazard CSV or GPX only**.
3. Upload your MP4 and, for geolocation, a matching GPX or CSV trajectory.
4. Select the detector and configure confidence, frame sampling, and camera/GPS settings.
5. Run analysis and review the annotated video, hazard table, map, and downloadable outputs.

## Requirements

- Python 3.10 or newer recommended.
- Internet access on first run if pretrained model weights need to be downloaded.
- Video file in MP4 format.
- For full geolocation, a timestamped GPX track or CSV with timestamps and coordinates.

## Run locally

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd geoai-road-hazards
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

### Try the built-in demo

Use **Run built-in demo** in the app to explore the end-to-end workflow with the bundled synthetic video and GPS track. The demo is useful for testing the pipeline and interface; it does not establish real-world hazard-detection performance.

## Input formats

### GPS CSV

A trajectory CSV should contain `timestamp`, `lat`, and `lon` columns. The app also accepts common aliases such as `time`, `latitude`, `longitude`, and `lng`.

```csv
timestamp,lat,lon
2026-01-01T10:00:00Z,18.5204,73.8567
2026-01-01T10:00:01Z,18.5205,73.8568
```

Use timestamps synchronized with the video, or configure the GPS-to-video offset in the app. GPX tracks should include timestamps on track points for synchronization.

### Hazard CSV (map-only mode)

```csv
lat,lon,type,severity,confidence
18.5204,73.8567,fallen_tree,high,0.91
```

## Detection models

The available detector choices have different purposes:

- **Auto / zero-shot AI:** uses an open-vocabulary model and text prompts for candidate hazards. Model downloads and availability depend on the installed Ultralytics version and network access. Verify detections on your own footage.
- **Pothole-only model:** uses the configured downloadable model and is limited to the classes it was trained to detect.
- **My own `.pt` file:** allows a compatible Ultralytics model trained on your selected classes.
- **Demo detector:** colour-based logic for the bundled synthetic demo clip only; it is not suitable for real road footage.

For robust field performance, evaluate and, where necessary, fine-tune a model using representative labelled footage for fallen trees, flooding, debris, power lines, landslides, potholes, and other target hazards. The [Road Damage Detector project](https://github.com/sekilab/RoadDamageDetector) is a useful reference for road-damage data, but it does not by itself cover every disaster-related hazard class.

## Outputs

- `hazards.geojson`: a GeoJSON FeatureCollection of mapped hazard features.
- `hazards.csv`: tabular hazard attributes.
- `annotated_hazards.mp4`: processed video with visual detections and alert sounds.
- Interactive map with hazard details and route context.

Feature properties may include an identifier, hazard type, severity, confidence/score, detection count, timestamp, video time, latitude, and longitude. Point and line geometries depend on the hazard grouping and processing path.

## How geolocation works

The pipeline aligns video times with GPS samples, interpolates the vehicle position, estimates heading from the trajectory, and uses the detection's image location with camera geometry to estimate a point on the road plane. Repeated detections can be grouped by proximity. The estimate is sensitive to GPS accuracy, time offset, camera height/pitch/field of view/yaw, terrain slope, and assumptions about the road surface. Do not interpret estimated coordinates as survey-grade or guaranteed-exact positions.

## Tests

Run the included geospatial and pipeline tests with:

```bash
python tests/test_geo.py
```

If tests require optional dependencies or fail in your environment, install the requirements and inspect the reported failure before relying on the outputs.

## Deployment

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create an app at [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select the repository, branch, and `app.py` as the entry point.
4. Confirm the Python dependencies and model-weight download strategy for the deployment environment.

### Docker

```bash
docker build -t geoai-roadguard .
docker run --rm -p 8501:8501 geoai-roadguard
```

The included `Dockerfile` provides a container-based option. Model downloads, memory usage, and platform compatibility should be tested in the target environment.

## Project structure

```text
app.py                  Streamlit user interface
engine/
  detectors.py           Detection backends and hazard prompts
  geo.py                 GPS parsing, interpolation, heading and projection
  hazards.py             Hazard categories, severity and GeoJSON helpers
  media.py               Video annotation and alert audio
  model_store.py         Model-weight setup
  pipeline.py            End-to-end processing pipeline
dashboard/               Interactive map assets
demo/                    Synthetic demonstration video and GPS data
tools/make_demo.py        Demo-data generator
tests/test_geo.py         Geospatial and pipeline tests
requirements.txt          Python dependencies
Dockerfile                Container configuration
```

## References

- [RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector)
- [comma2k19 dataset](https://github.com/commaai/comma2k19)
- [Ultralytics documentation](https://docs.ultralytics.com/)
- [Streamlit documentation](https://docs.streamlit.io/)

## License

Add a `LICENSE` file before publishing if you want to specify reuse terms. Check the licenses and terms for datasets, pretrained weights, and third-party components separately.
