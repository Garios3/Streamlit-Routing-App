
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# Definimos el título y el layout de la página
st.set_page_config(page_title="Greedy Routing Map", layout="wide")

# Formatos generales
st.markdown(
    """
    <style>
    .title-container {
        background-image: url("https://w0.peakpx.com/wallpaper/581/454/HD-wallpaper-network-concept-darkness-points-and-lines-social-network-abstract-art-network.jpg");
        background-size: cover;
        background-position: center;
        padding: 40px 20px;
        border-radius: 12px;
        margin-bottom: 25px;
    }
    .title-text {
        color: white;
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
    }
    .subtitle-text {
        color: white;
        font-size: 18px;
        text-align: center;
        margin-top: 10px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
    }
    </style>

    <div class="title-container">
        <div class="title-text">Greedy Routing</div>
        <div class="subtitle-text">
            Upload locations • Compute routes • Visualize on map
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Inicialización de la sesión (auxiliares)
if "route" not in st.session_state:
    st.session_state.route = None
if "total_km" not in st.session_state:
    st.session_state.total_km = None
if "df_used" not in st.session_state:
    st.session_state.df_used = None

# Funciones
# Cálculo de la distancia (Harvesine)
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088 # Curvatura (fijo)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2 # Fórmula de cálculo de la distancia
    return 2 * R * np.arcsin(np.sqrt(a))

# Cálculo de la matriz de distancias en una matriz (pares de distancias)
def pairwise_distance_matrix(df, lat_col="lat", lon_col="lon"):
    n = len(df)
    D = np.zeros((n, n), dtype=float) # Rellenamos la matriz con ceros (matriz cuadrada)
    lats = df[lat_col].to_numpy()
    lons = df[lon_col].to_numpy()
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(lats[i], lons[i], lats[j], lons[j]) # Para cada par calculamos su distancia y llevamos a la matriz
            D[i, j] = d # Dejamos las distancias simétricas (i,j) = (j,i)
            D[j, i] = d
    return D

# Ruteo con algoritmo Greedy
def greedy_route(df, start_idx=0, return_to_start=False, lat_col="lat", lon_col="lon"):
    df = df.reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        return [], 0.0
    if not (0 <= start_idx < n):
        raise ValueError("start_idx must be between 0 and n-1.")

    D = pairwise_distance_matrix(df, lat_col=lat_col, lon_col=lon_col)

    unvisited = set(range(n)) # Nodos no visitados
    route = [start_idx]
    unvisited.remove(start_idx)
    total = 0.0
    current = start_idx

    while unvisited:
        nxt = min(unvisited, key=lambda j: D[current, j]) # Se rellena la ruta con el siguiente mínimo
        total += D[current, nxt]
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt

    if return_to_start and n > 1:
        total += D[current, start_idx]
        route.append(start_idx)

    return route, total

# Generación del mapa con Leaflet
def build_route_map(df, route, lat_col="lat", lon_col="lon", name_col=None):
    df = df.reset_index(drop=True)
    coords = [[df.loc[i, lat_col], df.loc[i, lon_col]] for i in route]
    center = [df[lat_col].mean(), df[lon_col].mean()]

    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

    # Añadimos los marcadores con visitas en el orden establecido
    for step, i in enumerate(route):
        label = f"{step}"
        if name_col and name_col in df.columns:
            label += f" - {df.loc[i, name_col]}"
        folium.Marker(
            location=[df.loc[i, lat_col], df.loc[i, lon_col]],
            popup=label,
            tooltip=label,
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size: 12px; 
                    font-weight: 700;
                    background: white;
                    border: 2px solid black;
                    border-radius: 999px;
                    width: 26px;
                    height: 26px;
                    display: flex;
                    align-items: center;
                    justify-content: center;">
                    {step}
                </div>
                """
            ),
        ).add_to(m)

    # Route polyline
    folium.PolyLine(coords, weight=4).add_to(m)

    # Fit bounds nicely
    m.fit_bounds(coords)
    return m
    
# ===================== UI =====================
# Eliminado ya que se cambia al principio
# st.title("Greedy Routing (Nearest-Neighbor) + Leaflet Map")

st.markdown(
    """
Carga un archivo **Excel** que contenga, al menos:
- `lat` (latitude)
- `lon` (longitude)

Opcional:
- `Nombre` (etiqueta para cada ubicación en el mapa)
"""
)

uploaded = st.file_uploader("Carga del archivo con las ubicaciones (.xlsx)", type=["xlsx"], key="uploader")

if uploaded is None:
    st.info("Carga un archivo .xlsx para empezar")
    st.stop()

# Read Excel
try:
    df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Could not read the Excel file: {e}")
    st.stop()

st.subheader("Visualización")
st.dataframe(df, use_container_width=True)

st.subheader("Mapeo de las columnas")
col_candidates = list(df.columns)

lat_col = st.selectbox(
    "Columna con Latitude",
    col_candidates,
    index=col_candidates.index("lat") if "lat" in col_candidates else 0,
    key="lat_col_select"
)

lon_col = st.selectbox(
    "Columna con Longitude",
    col_candidates,
    index=col_candidates.index("lon") if "lon" in col_candidates else 0,
    key="lon_col_select"
)

name_col = None
if "name" in col_candidates:
    use_name = st.checkbox("Usa una columna 'name' para las etiquetas", value=True, key="use_name_checkbox")
    if use_name:
        name_col = "name"
else:
    use_name = st.checkbox("Usa una columna para las etiquetas", value=False, key="use_label_checkbox")
    if use_name:
        name_col = st.selectbox("Etiqueta", col_candidates, key="label_col_select")

# Validación y limpieza
df2 = df.copy()

# Convert to numeric (coerce invalid to NaN)
df2[lat_col] = pd.to_numeric(df2[lat_col], errors="coerce")
df2[lon_col] = pd.to_numeric(df2[lon_col], errors="coerce")

df2 = df2.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

if len(df2) < 2:
    st.error("Need at least 2 valid locations after cleaning lat/lon.")
    st.stop()

st.subheader("Opciones de ruteo")

if name_col and name_col in df2.columns:
    start_label = st.selectbox(
        "Ubicación inicial",
        df2[name_col].astype(str).tolist(),
        key="start_location_select"
    )
    start_idx = int(df2.index[df2[name_col].astype(str) == start_label][0])
else:
    start_idx = st.number_input(
        "Start index (0-based)",
        min_value=0, max_value=len(df2)-1, value=0, step=1,
        key="start_idx_input"
    )

return_to_start = st.checkbox("Volver al inicio (loop cerrado)", value=False, key="return_to_start_checkbox")

compute = st.button("Obtener la ruta basado en un algoritmo Greedy", key="compute_button")

if compute:
    route, total_km = greedy_route(
        df2,
        start_idx=start_idx,
        return_to_start=return_to_start,
        lat_col=lat_col,
        lon_col=lon_col,
    )

    st.session_state.route = route
    st.session_state.total_km = total_km
    st.session_state.df_used = df2.copy()

# Ajuste para display persistente
if st.session_state.route is not None:
    st.success(f"Distancia total: {st.session_state.total_km:.2f} km")

    df_used = st.session_state.df_used
    route = st.session_state.route

    coords = [[df_used.loc[i, lat_col], df_used.loc[i, lon_col]] for i in route]

    m = folium.Map(tiles="OpenStreetMap")

    for step, i in enumerate(route):
        label = f"{step}"
        if name_col and name_col in df_used.columns:
            label += f" - {df_used.loc[i, name_col]}"

        folium.Marker(
            [df_used.loc[i, lat_col], df_used.loc[i, lon_col]],
            popup=label,
            tooltip=label
        ).add_to(m)

    folium.PolyLine(coords, weight=4).add_to(m)
    m.fit_bounds(coords)

    st_folium(m, width=900, height=600, key="route_map")




