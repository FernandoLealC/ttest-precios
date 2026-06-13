# ============================================================
#  t-Test Analizador de Precios — Ejercicio 15 RD3
#  Universidad Panamericana — IA para el Análisis Financiero
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
import io

# ── Configuración de página ──────────────────────────────────────────────
st.set_page_config(
    page_title="t-Test | Análisis de Precios",
    page_icon="📊",
    layout="wide"
)

# ── Estilos ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        background: linear-gradient(90deg, #001D3D, #003566);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .main-title h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-title p  { color: #aec6e8; margin: 0.3rem 0 0; font-size: 0.95rem; }
    .card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .verdict-yes {
        background: #e8f5e9;
        border-left: 5px solid #2e7d32;
        border-radius: 6px;
        padding: 1rem 1.4rem;
        color: #1b5e20;
        font-size: 1.05rem;
        font-weight: 600;
    }
    .verdict-no {
        background: #fff3e0;
        border-left: 5px solid #e65100;
        border-radius: 6px;
        padding: 1rem 1.4rem;
        color: #bf360c;
        font-size: 1.05rem;
        font-weight: 600;
    }
    .pvalue-big {
        font-size: 2.2rem;
        font-weight: 700;
    }
    .info-box {
        background: #e3f2fd;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        color: #0d47a1;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .orange-header {
        color: #F15B2B;
        font-weight: 700;
        font-size: 1.1rem;
        border-bottom: 2px solid #F15B2B;
        padding-bottom: 0.3rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-title">
    <h1>📊 t-Test — Analizador de Cambios de Precio</h1>
    <p>Ejercicio 15 RD3 · Universidad Panamericana · IA para el Análisis Financiero</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")

    variante = st.selectbox(
        "Variante del t-Test",
        options=["1 · Pareada (mismos productos, antes/después)",
                 "2 · Dos grupos — varianzas iguales",
                 "3 · Dos grupos — varianzas desiguales (Welch)"],
        help="Selecciona según cómo están organizados tus datos."
    )

    alpha = st.slider("Nivel de significancia (α)", 0.01, 0.10, 0.05, 0.01,
                      format="%.2f",
                      help="Umbral para decidir si la diferencia es significativa. Estándar = 0.05")

    st.markdown("---")
    st.markdown("### 📂 Cargar datos")

    archivo = st.file_uploader(
        "Arrastra tu CSV o Excel aquí",
        type=["csv", "xlsx"],
        help="El archivo debe tener 3 columnas: Producto, Grupo A, Grupo B"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color:#666;'>
    <b>Estructura esperada del archivo:</b><br><br>
    <b>Variante 1 (Pareada):</b><br>
    Producto | Margen_Antes_% | Margen_Despues_%<br><br>
    <b>Variante 2 (Var. iguales):</b><br>
    Producto | Margen_A_% | Margen_B_%<br><br>
    <b>Variante 3 (Welch):</b><br>
    Producto | Margen_Estandar_% | Margen_Premium_%
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# FUNCIONES PRINCIPALES
# ════════════════════════════════════════════════════════════════════════════

def leer_archivo(f):
    if f.name.endswith(".csv"):
        return pd.read_csv(f)
    return pd.read_excel(f)

def get_variante_num(v):
    return int(v[0])

def correr_ttest(df, variante_num, alpha):
    col_a = df.columns[1]
    col_b = df.columns[2]
    grupo_a = df[col_a].astype(float)
    grupo_b = df[col_b].astype(float)

    if variante_num == 1:
        t_stat, p_two = stats.ttest_rel(grupo_a, grupo_b)
        df_val = len(grupo_a) - 1
        corr = grupo_a.corr(grupo_b)
        extra = {"Correlación de Pearson": round(corr, 4)}
        pooled = None
    elif variante_num == 2:
        t_stat, p_two = stats.ttest_ind(grupo_a, grupo_b, equal_var=True)
        df_val = len(grupo_a) + len(grupo_b) - 2
        var_pool = ((len(grupo_a)-1)*grupo_a.var(ddof=1) +
                    (len(grupo_b)-1)*grupo_b.var(ddof=1)) / df_val
        extra = {"Varianza agrupada (pooled)": round(var_pool, 4)}
        pooled = var_pool
    else:
        t_stat, p_two = stats.ttest_ind(grupo_a, grupo_b, equal_var=False)
        df_val = len(grupo_a) + len(grupo_b) - 2
        extra = {}
        pooled = None

    t_crit = stats.t.ppf(1 - alpha/2, df_val)
    significativa = p_two < alpha

    return {
        "col_a": col_a, "col_b": col_b,
        "grupo_a": grupo_a, "grupo_b": grupo_b,
        "media_a": round(grupo_a.mean(), 4),
        "media_b": round(grupo_b.mean(), 4),
        "var_a":   round(grupo_a.var(ddof=1), 4),
        "var_b":   round(grupo_b.var(ddof=1), 4),
        "n":       len(grupo_a),
        "t_stat":  round(t_stat, 4),
        "p_two":   p_two,
        "t_crit":  round(t_crit, 4),
        "df":      df_val,
        "significativa": significativa,
        "extra":   extra,
    }

def interpretar(r, variante_num, alpha):
    dif = round(r["media_b"] - r["media_a"], 2)
    col_a = r["col_a"].replace("_", " ")
    col_b = r["col_b"].replace("_", " ")
    if r["significativa"]:
        if variante_num == 1:
            direccion = "subió" if dif > 0 else "bajó"
            return (f"El ajuste de precio {direccion} el margen en {abs(dif):.2f} pp. "
                    f"Con p-value = {r['p_two']:.6f} (< {alpha}), esta diferencia es estadísticamente real — "
                    f"no fue casualidad.")
        else:
            return (f"Los grupos '{col_a}' y '{col_b}' tienen márgenes estadísticamente distintos "
                    f"({r['media_a']}% vs {r['media_b']}%). "
                    f"Con p-value = {r['p_two']:.6f} (< {alpha}), la diferencia no es azar.")
    else:
        if variante_num == 3:
            return (f"Aunque las medias son diferentes ({r['media_a']}% vs {r['media_b']}%), "
                    f"la alta variabilidad de uno de los grupos hace que no podamos afirmar que "
                    f"la diferencia sea real. p-value = {r['p_two']:.4f} (> {alpha}). "
                    f"Se necesitan más datos o reducir la dispersión del grupo volátil.")
        return (f"No hay evidencia estadística suficiente para afirmar que los grupos son distintos. "
                f"p-value = {r['p_two']:.4f} (> {alpha}). La diferencia observada pudo ser azar.")

def grafico_comparacion(r, df):
    col_a, col_b = r["col_a"], r["col_b"]
    productos = df.iloc[:, 0].astype(str)
    grupo_a   = r["grupo_a"]
    grupo_b   = r["grupo_b"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=col_a.replace("_", " "),
        x=productos, y=grupo_a,
        marker_color="#001D3D",
        text=[f"{v:.1f}%" for v in grupo_a],
        textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name=col_b.replace("_", " "),
        x=productos, y=grupo_b,
        marker_color="#F15B2B",
        text=[f"{v:.1f}%" for v in grupo_b],
        textposition="outside"
    ))
    fig.update_layout(
        barmode="group",
        title="Comparación de márgenes por producto",
        xaxis_title="Producto",
        yaxis_title="Margen (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        font=dict(family="Arial", size=12),
        margin=dict(t=60, b=80)
    )
    fig.update_xaxes(tickangle=-30, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig

def grafico_distribucion(r):
    a = r["grupo_a"]
    b = r["grupo_b"]
    fig = go.Figure()
    for vals, name, color in [(a, r["col_a"], "#001D3D"), (b, r["col_b"], "#F15B2B")]:
        fig.add_trace(go.Box(
            y=vals, name=name.replace("_", " "),
            marker_color=color,
            boxmean=True,
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(size=6, opacity=0.6)
        ))
    fig.update_layout(
        title="Distribución de márgenes (Boxplot)",
        yaxis_title="Margen (%)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=380,
        font=dict(family="Arial", size=12),
        showlegend=True
    )
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig

def tabla_resultados_df(r, variante_num):
    filas = [
        ("Media Grupo A (%)", f"{r['media_a']}"),
        ("Media Grupo B (%)", f"{r['media_b']}"),
        ("Varianza Grupo A",  f"{r['var_a']}"),
        ("Varianza Grupo B",  f"{r['var_b']}"),
        ("Observaciones (n)", f"{r['n']}"),
    ]
    for k, v in r["extra"].items():
        filas.append((k, str(v)))
    filas += [
        ("Grados de libertad (df)", f"{r['df']}"),
        ("Estadístico t",           f"{r['t_stat']}"),
        ("P-value (dos colas)",     f"{r['p_two']:.10f}"),
        (f"t crítico (α={r.get('alpha', 0.05)}, dos colas)", f"{r['t_crit']}"),
    ]
    return pd.DataFrame(filas, columns=["Parámetro", "Valor"])

# ════════════════════════════════════════════════════════════════════════════
# PANTALLA PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

variante_num = get_variante_num(variante)

# ── Sin archivo: mostrar guía de uso ────────────────────────────────────
if archivo is None:
    st.markdown('<p class="orange-header">¿Cómo usar esta herramienta?</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
        <b>Paso 1 — Elige la variante</b><br><br>
        En el panel izquierdo selecciona cuál de las 3 pruebas t aplica a tu situación.<br><br>
        • <b>Pareada:</b> mismos productos antes/después<br>
        • <b>Var. iguales:</b> dos categorías similares<br>
        • <b>Welch:</b> un grupo muy volátil
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
        <b>Paso 2 — Carga tu archivo</b><br><br>
        Arrastra un CSV o Excel con exactamente 3 columnas:<br><br>
        • Columna 1: nombre del producto<br>
        • Columna 2: margen del grupo A<br>
        • Columna 3: margen del grupo B
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card">
        <b>Paso 3 — Lee el resultado</b><br><br>
        La herramienta calcula automáticamente:<br><br>
        • p-value y veredicto en español<br>
        • Estadístico t y t crítico<br>
        • Gráficas comparativas<br>
        • Tabla completa exportable
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="orange-header">Regla de oro del p-value</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.success("**p-value < 0.05 → Diferencia REAL**  \nLa diferencia no es casualidad. Puedes tomar decisiones con base en ella.")
    with col_b:
        st.warning("**p-value ≥ 0.05 → No significativa**  \nNo hay evidencia suficiente. La diferencia pudo ser azar.")

    st.markdown("---")
    st.markdown('<p class="orange-header">Archivos de prueba disponibles</p>', unsafe_allow_html=True)
    st.markdown("""
    Descarga cualquiera de estos archivos de prueba y arrástralo a la herramienta:

    | Archivo | Para qué variante |
    |---|---|
    | `ttest_1_pareada.csv` | Variante 1 — Pareada |
    | `ttest_2_var_iguales.csv` | Variante 2 — Varianzas iguales |
    | `ttest_3_var_desiguales.csv` | Variante 3 — Welch |
    """)
    st.stop()

# ── Con archivo: correr análisis ─────────────────────────────────────────
try:
    df = leer_archivo(archivo)
except Exception as e:
    st.error(f"Error al leer el archivo: {e}")
    st.stop()

if df.shape[1] < 3:
    st.error("El archivo debe tener al menos 3 columnas: Producto, Grupo A, Grupo B.")
    st.stop()

r = correr_ttest(df, variante_num, alpha)
r["alpha"] = alpha

# ── Nombre de variante ───────────────────────────────────────────────────
nombres_variante = {
    1: "Prueba t Pareada (Paired Two Sample for Means)",
    2: "Dos Muestras — Varianzas Iguales",
    3: "Dos Muestras — Varianzas Desiguales (Welch)"
}
st.markdown(f'<p class="orange-header">📋 {nombres_variante[variante_num]}</p>', unsafe_allow_html=True)
st.caption(f"Archivo: **{archivo.name}** · {r['n']} productos · α = {alpha}")

# ── Métricas rápidas ─────────────────────────────────────────────────────
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Media Grupo A", f"{r['media_a']}%")
mc2.metric("Media Grupo B", f"{r['media_b']}%", delta=f"{round(r['media_b']-r['media_a'],2)} pp")
mc3.metric("Estadístico t", f"{r['t_stat']}")
mc4.metric("t crítico (α)", f"{r['t_crit']}")

st.markdown("---")

# ── Veredicto central ────────────────────────────────────────────────────
col_verd, col_pval = st.columns([3, 1])
with col_verd:
    interpretacion = interpretar(r, variante_num, alpha)
    if r["significativa"]:
        st.markdown(f'<div class="verdict-yes">✅ DIFERENCIA SIGNIFICATIVA<br><span style="font-weight:400;font-size:0.95rem">{interpretacion}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict-no">⚠️ NO SIGNIFICATIVA<br><span style="font-weight:400;font-size:0.95rem">{interpretacion}</span></div>', unsafe_allow_html=True)

with col_pval:
    color_pval = "#1b5e20" if r["significativa"] else "#bf360c"
    p_display = f"{r['p_two']:.6f}" if r['p_two'] > 0.0001 else f"{r['p_two']:.2e}"
    st.markdown(f"""
    <div style="text-align:center; padding:1rem; background:#f8f8f8; border-radius:10px; border:1px solid #ddd">
        <div style="font-size:0.85rem; color:#666; margin-bottom:4px">p-value (dos colas)</div>
        <div class="pvalue-big" style="color:{color_pval}">{p_display}</div>
        <div style="font-size:0.8rem; color:#999; margin-top:4px">umbral α = {alpha}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Gráficas ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Comparación por producto", "📦 Distribución (Boxplot)", "📋 Tabla de resultados"])

with tab1:
    st.plotly_chart(grafico_comparacion(r, df), use_container_width=True)

with tab2:
    st.plotly_chart(grafico_distribucion(r), use_container_width=True)
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(f"""
        <div class="info-box">
        <b>Varianza Grupo A:</b> {r['var_a']}<br>
        <b>Varianza Grupo B:</b> {r['var_b']}<br>
        <b>Razón de varianzas:</b> {round(max(r['var_a'],r['var_b'])/max(min(r['var_a'],r['var_b']),0.0001),2)}x
        </div>
        """, unsafe_allow_html=True)
        if variante_num == 3 and max(r['var_a'],r['var_b'])/max(min(r['var_a'],r['var_b']),0.0001) > 4:
            st.info("💡 La razón de varianzas es alta (>4x), lo que confirma que la variante Welch es la correcta para estos datos.")
    with col_info2:
        if "Correlación de Pearson" in r["extra"]:
            corr_val = r["extra"]["Correlación de Pearson"]
            st.markdown(f"""
            <div class="info-box">
            <b>Correlación de Pearson:</b> {corr_val}<br>
            {'Alta correlación — los productos se mueven juntos. Ideal para la prueba pareada.' if corr_val > 0.7 else 'Correlación moderada — verificar si la prueba pareada es apropiada.'}
            </div>
            """, unsafe_allow_html=True)

with tab3:
    df_res = tabla_resultados_df(r, variante_num)
    st.dataframe(
        df_res.style.apply(
            lambda row: ["background-color: #fff9c4; font-weight: bold" if "P-value" in row["Parámetro"] else "" for _ in row],
            axis=1
        ),
        use_container_width=True, hide_index=True, height=380
    )

    # Botón exportar
    csv_export = df_res.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar resultados como CSV",
        data=csv_export,
        file_name=f"ttest_resultados_{archivo.name}",
        mime="text/csv"
    )

st.markdown("---")

# ── Datos crudos ─────────────────────────────────────────────────────────
with st.expander("🔍 Ver datos originales cargados"):
    st.dataframe(df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#999; font-size:0.8rem; margin-top:2rem; padding-top:1rem; border-top:1px solid #eee">
    Universidad Panamericana · IA para el Análisis Financiero · Ejercicio 15 RD3
</div>
""", unsafe_allow_html=True)
