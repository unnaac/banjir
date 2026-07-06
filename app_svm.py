"""
Aplikasi Streamlit - Peta Kerawanan Banjir (Model: SVM)
Dua mode: (1) peta choropleth seluruh kecamatan, (2) input manual + titik prediksi.
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import joblib
import folium
import json
import base64 # Ditambahkan untuk mengonversi peta ke iframe murni

st.set_page_config(page_title="Peta Kerawanan Banjir - SVM", page_icon="🌊", layout="wide")

# =========================================================
# FUNGSI RENDER PETA (ANTI-ERROR & 100% RESPONSIF)
# =========================================================
def render_map_responsive(m, height=550):
    """Merender peta Folium ke HTML murni tanpa bergantung pada st.components"""
    map_html = m.get_root().render()
    b64 = base64.b64encode(map_html.encode('utf-8')).decode('utf-8')
    # Menyuntikkan iframe murni dengan lebar 100%
    iframe_html = f'<iframe src="data:text/html;base64,{b64}" width="100%" height="{height}" style="border:none; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"></iframe>'
    st.markdown(iframe_html, unsafe_allow_html=True)

# =========================================================
# MUAT MODEL & DATA GEOSPASIAL
# =========================================================
MODEL_PATH = "model_svm_banjir_streamlit.joblib"
GEOJSON_PATH = "BandungRaya_merged.geojson"
KOLOM_NAMA_KEC_GEOJSON = "kecamata"                 
PRED_CSV_PATH = "prediksi_svm_per_kecamatan.csv"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]
classes = bundle["classes"]
stats = bundle["feature_stats"]

gdf = gpd.read_file(GEOJSON_PATH)
gdf["key_join"] = gdf[KOLOM_NAMA_KEC_GEOJSON].astype(str).str.strip().str.upper()

pred_df = pd.read_csv(PRED_CSV_PATH)
pred_df["key_join"] = pred_df["key_join"].astype(str).str.strip().str.upper()
gdf = gdf.merge(pred_df, on="key_join", how="left", suffixes=("", "_pred"))
if "kecamata_pred" in gdf.columns:
    gdf.drop(columns=["kecamata_pred"], inplace=True)
if "kab_kota_pred" in gdf.columns:
    gdf.drop(columns=["kab_kota_pred"], inplace=True)

WARNA = {
    "Rendah": "#2ecc71",
    "Sedang": "#f1c40f",
    "Tinggi": "#e67e22",
    "Sangat Tinggi": "#e74c3c",
}

def warna_kelas(kelas):
    return WARNA.get(kelas, "#999999")

# =========================================================
# ANTARMUKA
# =========================================================
st.markdown("""
    <style>
    .stProgress > div > div > div > div { background-color: #3498db; }
    </style>
""", unsafe_allow_html=True)

st.title("🌊 Peta Kerawanan Banjir — Model SVM")
st.markdown("Aplikasi prediksi kerawanan banjir berbasis **Support Vector Machine (SVM)** untuk wilayah Bandung Raya.")

tab1, tab2 = st.tabs(["Peta Keseluruhan", "Input Manual & Simulasi"])

bounds = gdf.total_bounds
lat_center = (bounds[1] + bounds[3]) / 2
lon_center = (bounds[0] + bounds[2]) / 2

# --- TAB 1: Choropleth semua kecamatan ---
with tab1:
    st.markdown("### Pemetaan Kerawanan Eksisting")
    st.markdown("Visualisasi prediksi tingkat kerawanan banjir untuk seluruh kecamatan berdasarkan data eksisting.")

    col_map_all, col_info_all = st.columns([2.2, 1])

    with col_map_all:
        m = folium.Map(location=[lat_center, lon_center], zoom_start=10)
        geojson_data = json.loads(gdf.to_json())

        for feature in geojson_data["features"]:
            status = feature["properties"].get("pred_kerawanan", "Rendah")
            feature["properties"]["style"] = {
                "fillColor": warna_kelas(status),
                "color": "black",
                "weight": 0.5,
                "fillOpacity": 0.7,
            }

        folium.GeoJson(
            geojson_data,
            tooltip=folium.GeoJsonTooltip(
                fields=["kecamata", "pred_kerawanan"],
                aliases=["Kecamatan:", "Prediksi Kerawanan:"],
            ),
        ).add_to(m)

        # Memanggil fungsi baru kita
        render_map_responsive(m, height=550)

    with col_info_all:
        st.markdown("#### Legenda & Ringkasan")
        st.write("Distribusi status kerawanan dari total wilayah:")
        
        if "pred_kerawanan" in gdf.columns:
            counts = gdf["pred_kerawanan"].value_counts().to_dict()
        else:
            counts = {}
            
        urutan_kelas = ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]
        
        for kls in urutan_kelas:
            jml = counts.get(kls, 0)
            warna = warna_kelas(kls)
            
            st.markdown(f"""
            <div style="
                border: 1px solid rgba(128, 128, 128, 0.2);
                border-left: 6px solid {warna}; 
                padding: 12px 16px; 
                margin-bottom: 12px; 
                border-radius: 6px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                background-color: transparent;">
                <span style="font-weight: 600; font-size: 16px;">{kls}</span>
                <span style="font-weight: 900; color: {warna}; font-size: 24px;">
                    {jml} <span style="font-size:14px; font-weight: normal; color: gray;">Kec</span>
                </span>
            </div>
            """, unsafe_allow_html=True)
            
        st.info("**Tips:** Arahkan kursor pada area peta untuk melihat detail nama kecamatan.")


# --- TAB 2: Input manual + titik prediksi ---
with tab2:
    st.markdown("### Simulasi Parameter")
    st.markdown("Ubah nilai parameter pada form di bawah untuk mensimulasikan dampaknya terhadap tingkat kerawanan banjir di kecamatan tertentu.")

    with st.form(key="form_input_manual", border=True):
        st.markdown("#### 1. Tentukan Lokasi")
        daftar_kec = sorted(gdf["kecamata"].dropna().unique())
        kec_pilih = st.selectbox(
            "Pilih Kecamatan untuk meletakkan titik koordinat simulasi pada peta", 
            daftar_kec
        )
        
        st.divider()

        st.markdown("#### 2. Sesuaikan Nilai Fitur")
        nilai = {}
        cols = st.columns(3)
        for i, f in enumerate(features):
            s = stats[f]
            nilai[f] = cols[i % 3].slider(
                label=f"**{f}**", 
                min_value=float(s["min"]), 
                max_value=float(s["max"]), 
                value=float(s["mean"]),
                help=f"Nilai rata-rata dataset: {s['mean']:.2f}"
            )

        st.write("") 
        submit_btn = st.form_submit_button("Jalankan Prediksi", type="primary", use_container_width=True)

    if submit_btn:
        X_input = pd.DataFrame([nilai])[features]
        hasil = model.predict(X_input)[0]
        proba = dict(zip(classes, model.predict_proba(X_input)[0].round(3)))

        st.markdown("---")
        st.markdown("### Hasil Analisis")

        col_map, col_results = st.columns([1.5, 1])

        # --- BAGIAN KIRI: PETA ---
        with col_map:
            st.markdown(f"**Titik Fokus:** Simulasi pada **Kecamatan {kec_pilih}**")
            
            row = gdf[gdf["kecamata"] == kec_pilih]
            geom = row.geometry.iloc[0]
            
            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                poly_bounds = geom.bounds 
                titik_y = (poly_bounds[1] + poly_bounds[3]) / 2
                titik_x = (poly_bounds[0] + poly_bounds[2]) / 2
            else:
                titik_y, titik_x = lat_center, lon_center

            m2 = folium.Map(location=[titik_y, titik_x], zoom_start=13)
                
            geojson_data_2 = json.loads(gdf.to_json())

            for feature in geojson_data_2["features"]:
                status = feature["properties"].get("pred_kerawanan", "Rendah")
                feature["properties"]["style"] = {
                    "fillColor": warna_kelas(status),
                    "color": "gray",
                    "weight": 0.3,
                    "fillOpacity": 0.25,
                }

            folium.GeoJson(geojson_data_2).add_to(m2)
            
            folium.CircleMarker(
                location=[titik_y, titik_x],
                radius=15,
                color="white",
                weight=2,
                fill=True,
                fill_color=warna_kelas(hasil),
                fill_opacity=0.9,
                popup=f"<b>{kec_pilih}</b><br>Hasil Prediksi Baru: {hasil}",
                tooltip="Klik untuk melihat detail"
            ).add_to(m2)

            # Memanggil fungsi baru kita
            render_map_responsive(m2, height=450)

        # --- BAGIAN KANAN: KARTU HASIL & PROBABILITAS ---
        with col_results:
            warna_hasil = warna_kelas(hasil)
            st.markdown(f"""
                <div style="
                    background-color: {warna_hasil}15; 
                    border: 2px solid {warna_hasil}; 
                    border-radius: 10px; 
                    padding: 25px; 
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                    margin-top: 30px;">
                    <p style="margin:0; font-size: 16px;">Tingkat Kerawanan:</p>
                    <h1 style="margin:0; color: {warna_hasil}; font-weight: 900; text-transform: uppercase;">{hasil}</h1>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("**Rincian Probabilitas Model (Tingkat Keyakinan):**")
            
            urutan_kelas = ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]
            
            for kls in urutan_kelas:
                if kls in proba:
                    prb = proba[kls]
                    col_teks, col_bar = st.columns([1.5, 4]) 
                    col_teks.write(f"**{kls}**")
                    col_bar.progress(float(prb), text=f"{prb*100:.1f}%")
