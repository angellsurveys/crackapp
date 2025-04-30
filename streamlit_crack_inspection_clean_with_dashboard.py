
# === FINAL MERGED STREAMLIT SCRIPT: Clean, Dashboard, No Arrows ===

import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

st.set_page_config(page_title="Bridge Crack Inspection", layout="wide")
st.title("Bridge Crack Inspection - Angell Surveys Ltd. 🚧")

st.sidebar.header("Settings")
tile_folder = st.sidebar.text_input("Path to /tiles/ folder", "tiles")
client_name = st.sidebar.text_input("Client Name", "")
project_name = st.sidebar.text_input("Project Name", "Enter Project Name")
inspector_name = st.sidebar.text_input("Inspector Name", "Philip Angell")
max_crack_width = st.sidebar.slider("Max Crack Width (mm)", 0.1, 5.0, 5.0, 0.1)

def get_geotiff_info(tile_path):
    try:
        with tifffile.TiffFile(tile_path) as tif:
            tags = tif.pages[0].tags
            scale = tags['ModelPixelScaleTag'].value
            tiepoint = tags['ModelTiepointTag'].value
            x_origin = tiepoint[3]
            y_origin = tiepoint[4]
            pixel_size_x = scale[0]
            pixel_size_y = scale[1]
        return x_origin, y_origin, pixel_size_x, pixel_size_y
    except:
        return None, None, None, None

def detect_cracks_in_tile(tile_path, tile_id, output_annotated_folder, max_crack_width):
    x_origin, y_origin, pixel_size_x, pixel_size_y = get_geotiff_info(tile_path)
    if None in (x_origin, y_origin, pixel_size_x, pixel_size_y):
        return []

    GSD = pixel_size_x * 1000
    img = cv2.imread(tile_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 10, 60)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cracks = []
    overlay = img.copy()

    for cnt in contours:
        if len(cnt) < 5 or cv2.contourArea(cnt) < 10:
            continue
        length_px = cv2.arcLength(cnt, True)
        length_mm = length_px * GSD
        if length_mm < 100:
            continue
        try:
            ellipse = cv2.fitEllipse(cnt)
            width_mm = min(ellipse[1]) * GSD
            if width_mm >= max_crack_width:
                continue
            cx, cy = ellipse[0]
            x_global = x_origin + (cx * pixel_size_x)
            y_global = y_origin - (cy * pixel_size_y)
            label = f"{width_mm:.2f}mm"
            cv2.drawContours(overlay, [cnt], -1, (0, 0, 255), 1)
            cv2.putText(overlay, label, (int(cx), int(cy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cracks.append({
                "Tile ID": tile_id,
                "Global X (m)": round(x_global, 3),
                "Global Y (m)": round(y_global, 3),
                "Length (mm)": round(length_mm, 2),
                "Width (mm)": round(width_mm, 2)
            })
        except:
            continue

    annotated_path = os.path.join(output_annotated_folder, f"{tile_id}_annotated.jpg")
    resized = cv2.resize(overlay, (1600, 1600))
    cv2.imwrite(annotated_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return cracks

def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.drawString(50, 20, "Copyright © Angell Surveys Ltd. 2025")
    canvas.drawRightString(A4[0] - 50, 20, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()

def build_full_pdf(crack_data, project, inspector, client):
    output = "output_results"
    pdf_path = os.path.join(output, "full_crack_report.pdf")
    annotated = os.path.join(output, "annotated_tiles")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    if os.path.exists("logo.png"):
        logo = Image("logo.png", width=120, height=40)
        logo.hAlign = 'RIGHT'
        story.append(logo)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Angell Surveys Ltd.", styles["Heading1"]))
    story.append(Paragraph(f"Client: {client}", styles["Normal"]))
    story.append(Paragraph(f"Project: {project}", styles["Normal"]))
    story.append(Paragraph(f"Inspector: {inspector}", styles["Normal"]))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%d %B %Y')}", styles["Normal"]))
    story.append(Spacer(1, 24))

    df = pd.DataFrame(crack_data)
    story.append(Paragraph("<a name='ExecutiveSummary'/><b>Executive Summary</b>", styles["Heading2"]))
    story.append(Paragraph(f"Total Cracks: {len(df)}", styles["Normal"]))
    story.append(Paragraph(f"Avg Width: {df['Width (mm)'].mean():.2f} mm", styles["Normal"]))
    story.append(Paragraph(f"Widest: {df['Width (mm)'].max():.2f} mm", styles["Normal"]))
    story.append(Paragraph(f"Smallest: {df['Width (mm)'].min():.2f} mm", styles["Normal"]))
    story.append(PageBreak())

    story.append(Paragraph("<b>Master Crack Table</b>", styles["Heading2"]))
    for i, row in df.iterrows():
        story.append(Paragraph(
            f"<link href='#{row['Tile ID']}'><b>Crack {i+1}</b></link>: {row['Tile ID']} | X: {row['Global X (m)']} | Y: {row['Global Y (m)']} | Width: {row['Width (mm)']}mm", styles["BodyText"]
        ))
    story.append(PageBreak())

    for tile_id in df['Tile ID'].unique():
        story.append(Paragraph(f"<a name='{tile_id}'/><b>Tile: {tile_id}</b>", styles["Heading2"]))
        img_path = os.path.join(annotated, f"{tile_id}_annotated.jpg")
        if os.path.exists(img_path):
            story.append(Image(img_path, width=400, height=400))
        for i, row in df[df["Tile ID"] == tile_id].iterrows():
            story.append(Paragraph(
                f"Crack: X: {row['Global X (m)']} | Y: {row['Global Y (m)']} | Width: {row['Width (mm)']} mm", styles["Normal"]
            ))
        story.append(Spacer(1, 12))
        story.append(Paragraph("<link href='#ExecutiveSummary'>⬅️ Back to Executive Summary</link>", styles["Normal"]))
        story.append(PageBreak())

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    return pdf_path

# === MAIN LOGIC ===
if st.sidebar.button("Start Inspection"):
    output = "output_results"
    annotated_folder = os.path.join(output, "annotated_tiles")
    os.makedirs(output, exist_ok=True)
    os.makedirs(annotated_folder, exist_ok=True)

    all_cracks = []
    tile_files = sorted([f for f in os.listdir(tile_folder) if f.lower().endswith(".tif")])
    progress = st.progress(0)

    for i, tf in enumerate(tile_files):
        tile_path = os.path.join(tile_folder, tf)
        tile_id = tf.replace(".tif", "")
        cracks = detect_cracks_in_tile(tile_path, tile_id, annotated_folder, max_crack_width)
        all_cracks.extend(cracks)
        progress.progress((i+1)/len(tile_files))

    df = pd.DataFrame(all_cracks)
    csv_path = os.path.join(output, "full_crack_table.csv")
    df.to_csv(csv_path, index=False)

    pdf_path = build_full_pdf(all_cracks, project_name, inspector_name, client_name)

    st.success("✅ Inspection complete. Download your results below.")
    with open(csv_path, "rb") as f:
        st.download_button("📥 Download Crack Table (CSV)", f, file_name="full_crack_table.csv")
    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download Full Crack Report (PDF)", f, file_name="full_crack_report.pdf")

    # === DASHBOARD ===
    st.header("📊 Crack Analysis Dashboard")
    st.metric("Total Cracks", len(df))
    st.metric("Avg Width (mm)", f"{df['Width (mm)'].mean():.2f}")
    st.metric("Max Width (mm)", f"{df['Width (mm)'].max():.2f}")
    st.subheader("Cracks per Tile")
    st.bar_chart(df["Tile ID"].value_counts().sort_index())
    st.subheader("Crack Width Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["Width (mm)"], bins=20, color='skyblue', edgecolor='black')
    ax.set_xlabel("Crack Width (mm)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # === TILE VIEWER ===
    st.header("🖼️ Annotated Tile Viewer")
    tile_imgs = sorted([f for f in os.listdir(annotated_folder) if f.endswith(".jpg")])
    cols = st.columns(4)
    for i, img_file in enumerate(tile_imgs):
        img_path = os.path.join(annotated_folder, img_file)
        with cols[i % 4]:
            st.image(img_path, caption=img_file, use_column_width=True)
