
import streamlit as st
import cv2
import tempfile
import os
import json
import pandas as pd
import folium

from ultralytics import YOLO
from streamlit_folium import st_folium

# Keep video results visible after Streamlit reruns.
if "processed_video_path" not in st.session_state:
    st.session_state.processed_video_path = None
if "video_metrics" not in st.session_state:
    st.session_state.video_metrics = None
if "class_counts" not in st.session_state:
    st.session_state.class_counts = {}
if "demo_detection_events" not in st.session_state:
    st.session_state.demo_detection_events = []


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="GEOAI Hazard Mapping",
    page_icon="🗺️",
    layout="wide"
)

# ---------------- PREMIUM DASHBOARD THEME ----------------
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at 12% 0%, rgba(49,94,251,.10), transparent 28%), #f5f7fc; }
[data-testid="stHeader"] { background: rgba(245,247,252,.82); }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1440px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#101a35 0%,#17264b 100%); }
[data-testid="stSidebar"] * { color: #f3f6ff !important; }
.geoai-hero { padding: 30px 32px; border-radius: 24px; color: white; background: linear-gradient(120deg,#101a35 0%,#233c80 60%,#315efb 100%); box-shadow: 0 18px 42px rgba(25,48,110,.18); margin-bottom: 22px; position: relative; overflow: hidden; }
.geoai-eyebrow { text-transform:uppercase; letter-spacing:2px; font-size:12px; font-weight:800; color:#9feaf1; margin-bottom:9px; }
.geoai-title { font-size:clamp(27px,4vw,42px); line-height:1.12; font-weight:850; margin:0 0 10px 0; color:#fff; }
.geoai-subtitle { font-size:15px; line-height:1.65; max-width:780px; color:#e0e8ff; margin:0; }
.geoai-pills { display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }
.geoai-pill { border:1px solid rgba(255,255,255,.24); background:rgba(255,255,255,.10); padding:7px 11px; border-radius:999px; font-size:12px; font-weight:650; }
.section-banner { border-left:4px solid #315efb; padding:7px 0 7px 13px; margin: 8px 0 15px 0; }
.section-banner h3 { margin:0; color:#17264b; font-size:23px; font-weight:800; }
.section-banner p { margin:4px 0 0 0; color:#64718a; font-size:13px; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,#fff,#f7f9ff); border:1px solid #e2e8f5; padding:17px 18px; border-radius:17px; box-shadow:0 7px 22px rgba(25,48,110,.055); }
div[data-testid="stMetricLabel"] { color:#64718a; font-weight:650; }
div[data-testid="stMetricValue"] { color:#17264b; font-weight:800; }
div[data-testid="stFileUploader"] { background:#fff; border:1px dashed #b8c7e8; border-radius:16px; padding:8px; }
div.stButton > button, div.stDownloadButton > button { border-radius:12px; border:0; padding:.62rem 1rem; font-weight:750; background:linear-gradient(100deg,#315efb,#5146e5); color:white; box-shadow:0 6px 14px rgba(49,94,251,.18); transition:transform .15s ease, box-shadow .15s ease; }
div.stButton > button:hover, div.stDownloadButton > button:hover { color:white; transform:translateY(-1px); box-shadow:0 10px 22px rgba(49,94,251,.24); }
div[data-testid="stDataFrame"], div[data-testid="stTable"] { border:1px solid #e2e8f5; border-radius:14px; overflow:hidden; }
.stAlert { border-radius:13px; }
hr { border-color:#e1e7f3; }
footer { visibility:hidden; }
@media (max-width: 700px) { .geoai-hero {padding:22px 20px;} .block-container {padding-left:1rem;padding-right:1rem;} }
</style>
<div class="geoai-hero">
  <div class="geoai-eyebrow">GEOAI LABS · HAZARD INTELLIGENCE</div>
  <div class="geoai-title">Road Hazard Mapping</div>
  <p class="geoai-subtitle">Turn dashcam footage and GPS tracks into an interactive road-monitoring workspace. Review AI detections, inspect route data, and export geographic results.</p>
  <div class="geoai-pills"><span class="geoai-pill">✦ AI video analysis</span><span class="geoai-pill">⌖ GPS trajectory</span><span class="geoai-pill">▧ GeoJSON export</span><span class="geoai-pill">◉ Human review workflow</span></div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🛰️ GEOAI 03")
    st.caption("ROAD SAFETY CONSOLE")
    st.markdown("---")
    st.markdown("**WORKSPACE**")
    st.markdown("🎥  Video analysis")
    st.markdown("📍  GPS & route mapping")
    st.markdown("📑  Review reports")
    st.markdown("---")
    st.markdown("**PROTOTYPE STATUS**")
    st.success("App interface ready")
    st.info("AI detections need human verification. GPS matching is a demonstration when using sample data.")


# ---------------- AI VIDEO DETECTION ----------------

st.markdown("""<div class="section-banner"><h3>🎥 AI Road Video Analysis</h3><p>Upload a dashcam clip and inspect general-purpose object detections.</p></div>""", unsafe_allow_html=True)

uploaded_video = st.file_uploader(
    "Upload dashcam video",
    type=["mp4", "avi", "mov"],
    key="road_video"
)

confidence = st.slider(
    "Detection confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05
)

if uploaded_video is not None:
    st.video(uploaded_video)

    if st.button("🚀 Run AI Detection"):
        temp_input = None
        temp_output = None
        cap = None
        writer = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as f:
                f.write(uploaded_video.getvalue())
                temp_input = f.name

            st.info("Loading YOLO AI model...")
            model = YOLO("yolov8n.pt")

            cap = cv2.VideoCapture(temp_input)

            if not cap.isOpened():
                raise ValueError("Could not open the uploaded video.")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 20.0

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if width <= 0 or height <= 0:
                raise ValueError("Invalid video dimensions.")

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as f:
                temp_output = f.name

            writer = cv2.VideoWriter(
                temp_output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            if not writer.isOpened():
                raise ValueError("Could not create output video.")

            total_detections = 0
            class_counts = {}
            detection_events = []
            frames_processed_by_ai = 0
            frame_number = 0

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            progress = st.progress(0)
            status = st.empty()

            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                # Run AI on every third frame
                if frame_number % 3 == 0:
                    result = model.predict(
                        frame,
                        conf=confidence,
                        imgsz=640,
                        verbose=False
                    )[0]

                    total_detections += len(result.boxes)
                    frame_event_classes = []
                    for detected_box in result.boxes:
                        class_id = int(detected_box.cls[0].item())
                        class_name = model.names.get(class_id, str(class_id))
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        frame_event_classes.append(class_name)
                    detection_events.append({
                        "elapsed_seconds": round(frame_number / fps, 2),
                        "classes": frame_event_classes,
                    })
                    frame = result.plot()
                    frames_processed_by_ai += 1

                writer.write(frame)
                frame_number += 1

                if frame_number % 10 == 0:
                    status.write(
                        f"Processing frame {frame_number} "
                        f"of {total_frames}"
                    )

                    progress.progress(
                        min(frame_number / max(total_frames, 1), 1.0)
                    )

            cap.release()
            cap = None

            writer.release()
            writer = None

            progress.progress(1.0)
            status.success("Video analysis completed!")

            st.session_state.class_counts = class_counts
            st.session_state.demo_detection_events = detection_events
            st.session_state.processed_video_path = temp_output
            st.session_state.video_metrics = {
                "total_frames": frame_number,
                "ai_frames": frames_processed_by_ai,
                "detections": total_detections,
            }



        except Exception as e:
            st.error(f"Video processing failed: {e}")

        finally:
            if cap is not None:
                cap.release()

            if writer is not None:
                writer.release()

            # Remove the temporary input, but keep the processed output.
            if temp_input and os.path.exists(temp_input):
                try:
                    os.remove(temp_input)
                except OSError:
                    pass


# Show saved results outside the button block so they survive reruns.
saved_video = st.session_state.get("processed_video_path")
saved_metrics = st.session_state.get("video_metrics")

if saved_video and os.path.exists(saved_video) and saved_metrics:
    st.markdown("""<div class="section-banner"><h3>✅ Analysis overview</h3><p>Key processing metrics and the annotated video output.</p></div>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total video frames", saved_metrics["total_frames"])
    col2.metric("AI-analyzed frames", saved_metrics["ai_frames"])
    col3.metric("Object detections", saved_metrics["detections"])

    st.markdown("""<div class="section-banner"><h3>🎯 Annotated video</h3><p>Review the model output before interpreting any object as a hazard.</p></div>""", unsafe_allow_html=True)
    st.video(saved_video)

    with open(saved_video, "rb") as video_file:
        st.download_button(
            "⬇️ Download processed video",
            data=video_file.read(),
            file_name="geoai_detected_video.mp4",
            mime="video/mp4",
        )

    st.markdown("""<div class="section-banner"><h3>🚧 Road-safety review report</h3><p>Detection counts and suggested review categories—not confirmed danger levels.</p></div>""", unsafe_allow_html=True)
    st.caption(
        "Prototype screening only: these are detected objects, not confirmed hazards. "
        "Review the video before taking action."
    )

    counts = st.session_state.get("class_counts", {})
    if counts:
        st.markdown("**Detection summary by object class**")
        chart_data = pd.DataFrame(
            {"Object class": list(counts.keys()), "Count": list(counts.values())}
        ).set_index("Object class")
        st.bar_chart(chart_data)
    if counts:
        report_rows = []
        priority_classes = {
            "person", "bicycle", "motorcycle", "car", "bus", "truck"
        }
        for class_name, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
            if class_name in {"person", "bicycle", "motorcycle"}:
                priority = "High review priority"
                reason = "Road user detected; verify context in video"
            elif class_name in {"car", "bus", "truck"}:
                priority = "Review"
                reason = "Vehicle detected; not itself a hazard"
            else:
                priority = "Informational"
                reason = "General object detection; verify relevance"
            report_rows.append({
                "Detected class": class_name,
                "Detection count": count,
                "Review priority": priority,
                "Interpretation": reason,
            })
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True)
        st.download_button(
            "⬇️ Download review report (CSV)",
            data=pd.DataFrame(report_rows).to_csv(index=False).encode("utf-8"),
            file_name="geoai_road_safety_review.csv",
            mime="text/csv",
        )
    else:
        st.info("No object-class summary is available yet. Run AI detection again.")

    st.warning(
        "These are general-purpose YOLO object detections. "
        "They are not automatically confirmed road hazards."
    )


# ---------------- DASHBOARD GUIDE ----------------
with st.expander("ℹ️ How to interpret this prototype"):
    st.markdown(
        "- **AI detections:** general-purpose objects detected in sampled video frames.\\n"
        "- **Review priorities:** suggestions for a human to inspect the video; not measured danger levels.\\n"
        "- **Blue GPS route/markers:** vehicle positions from the uploaded CSV.\\n"
        "- **Confirmed hazards:** not yet automatically established by this prototype.\\n"
        "- **Next technical milestone:** synchronize real video timestamps with real GPS readings and validate hazard classes."
    )


# ---------------- GPS AND GIS MAPPING ----------------

st.divider()
st.markdown("""<div class="section-banner"><h3>📍 GPS trajectory & GIS mapping</h3><p>Validate timestamped coordinates, inspect the route, and export geographic data.</p></div>""", unsafe_allow_html=True)

st.write(
    "Upload a timestamped GPS CSV to visualize the vehicle route "
    "and export its coordinates as GeoJSON."
)

gps_file = st.file_uploader(
    "Upload GPS trajectory CSV",
    type=["csv"],
    key="gps_csv"
)

if gps_file is not None:
    try:
        gps = pd.read_csv(gps_file)

        # Standardize column names
        gps.columns = gps.columns.str.strip().str.lower()

        required = {"timestamp", "latitude", "longitude"}

        if not required.issubset(gps.columns):
            st.error(
                "CSV must contain these columns: "
                "timestamp, latitude, longitude"
            )

        else:
            gps["timestamp"] = pd.to_datetime(
                gps["timestamp"], errors="coerce", utc=True
            )

            gps["latitude"] = pd.to_numeric(
                gps["latitude"], errors="coerce"
            )

            gps["longitude"] = pd.to_numeric(
                gps["longitude"], errors="coerce"
            )

            gps = gps.dropna(
                subset=["timestamp", "latitude", "longitude"]
            )

            gps = gps[
                gps["latitude"].between(-90, 90)
                & gps["longitude"].between(-180, 180)
            ].sort_values("timestamp")

            if gps.empty:
                st.error("No valid GPS records found.")

            else:
                st.success(
                    f"Successfully loaded {len(gps)} GPS positions."
                )

                st.dataframe(gps.head(10), use_container_width=True)

                # Create GeoJSON features
                features = []

                for _, row in gps.iterrows():
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [
                                float(row["longitude"]),
                                float(row["latitude"])
                            ]
                        },
                        "properties": {
                            "timestamp": row["timestamp"].isoformat(),
                            "source": "GPS trajectory",
                            "status": "Vehicle position; hazard unverified"
                        }
                    })

                geojson = {
                    "type": "FeatureCollection",
                    "features": features
                }

                st.download_button(
                    "⬇️ Download GPS GeoJSON",
                    data=json.dumps(geojson, indent=2),
                    file_name="geoai_trajectory.geojson",
                    mime="application/geo+json"
                )

                # Create interactive map
                center = [
                    float(gps["latitude"].mean()),
                    float(gps["longitude"].mean())
                ]

                m = folium.Map(
                    location=center,
                    zoom_start=14,
                    tiles="OpenStreetMap"
                )

                route_points = list(
                    zip(gps["latitude"], gps["longitude"])
                )

                if len(route_points) >= 2:
                    folium.PolyLine(
                        route_points,
                        color="blue",
                        weight=4,
                        tooltip="Vehicle GPS trajectory"
                    ).add_to(m)

                for _, row in gps.iterrows():
                    folium.CircleMarker(
                        location=[
                            float(row["latitude"]),
                            float(row["longitude"])
                        ],
                        radius=5,
                        color="blue",
                        fill=True,
                        fill_opacity=0.8,
                        popup=(
                            f"GPS time: {row['timestamp'].isoformat()}"
                        )
                    ).add_to(m)

                # DEMONSTRATION ONLY: assume video starts at the first CSV timestamp.
                # This is not real synchronization because the CSV is sample data.
                demo_events = st.session_state.get("demo_detection_events", [])
                if demo_events:
                    st.markdown("### 🟠 Demonstration Detection-to-GPS Matching")
                    st.warning(
                        "DEMO MODE ONLY: the sample GPS CSV is illustrative. "
                        "For this preview, video elapsed time is aligned to the first CSV timestamp. "
                        "Orange markers are candidate review points, NOT confirmed hazard locations."
                    )
                    first_gps_time = gps["timestamp"].iloc[0]
                    gps_seconds = (
                        (gps["timestamp"] - first_gps_time).dt.total_seconds().to_numpy()
                    )
                    demo_features = []
                    for event in demo_events:
                        if not event["classes"] or len(gps_seconds) == 0:
                            continue
                        elapsed = float(event["elapsed_seconds"])
                        nearest_idx = int(abs(gps_seconds - elapsed).argmin())
                        gps_row = gps.iloc[nearest_idx]
                        class_summary = ", ".join(sorted(set(event["classes"])))
                        folium.Marker(
                            location=[float(gps_row["latitude"]), float(gps_row["longitude"])],
                            tooltip="DEMO candidate — verify video",
                            popup=(
                                "<b>DEMONSTRATION ONLY</b><br>"
                                f"Video elapsed time: {elapsed:.2f} s<br>"
                                f"Detected classes: {class_summary}<br>"
                                "Not a confirmed road hazard."
                            ),
                            icon=folium.Icon(color="orange", icon="info-sign"),
                        ).add_to(m)
                        demo_features.append({
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [
                                    float(gps_row["longitude"]),
                                    float(gps_row["latitude"]),
                                ],
                            },
                            "properties": {
                                "mode": "DEMONSTRATION ONLY",
                                "video_elapsed_seconds": elapsed,
                                "detected_classes": class_summary,
                                "gps_timestamp_used": gps_row["timestamp"].isoformat(),
                                "status": "Candidate review point; not a confirmed hazard",
                                "gps_source": "Illustrative sample CSV",
                            },
                        })
                    if demo_features:
                        st.download_button(
                            "⬇️ Download demonstration matching GeoJSON",
                            data=json.dumps({
                                "type": "FeatureCollection",
                                "features": demo_features,
                            }, indent=2),
                            file_name="geoai_demo_detection_gps_match.geojson",
                            mime="application/geo+json",
                        )
                    else:
                        st.info("No detected objects were recorded for demonstration matching.")

                st.markdown("""<div class="section-banner"><h3>🗺️ Interactive route map</h3><p>Blue markers are GPS positions; orange markers are demonstration matches.</p></div>""", unsafe_allow_html=True)

                st_folium(
                    m,
                    use_container_width=True,
                    height=500,
                    key="gps_map"
                )

                st.caption(
                    "Blue markers represent vehicle GPS positions, "
                    "not confirmed hazards. Actual hazard locations "
                    "require matching video detections to GPS timestamps "
                    "and estimating the camera's viewing geometry."
                )

    except Exception as e:
        st.error(f"GPS processing failed: {e}")


# ---------------- PROJECT STATUS ----------------

st.divider()
st.markdown("""<div class="section-banner"><h3>📊 Project pipeline</h3><p>Current capabilities and the next milestones toward a validated system.</p></div>""", unsafe_allow_html=True)

st.markdown(
    """
    - ✅ Video upload
    - ✅ General-purpose YOLO object detection
    - ✅ GPS CSV upload and validation
    - ✅ Interactive GPS route map
    - ✅ GeoJSON trajectory export
    - ⏳ Matching AI detections to GPS timestamps
    - ⏳ Validated road-hazard classification and geolocation
    """
)