# ============================================================
#  t-Test Analizador de Cambios de Precio — Ejercicio 15 RD3
#  Universidad Panamericana — IA para el Análisis Financiero
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import io

st.set_page_config(
    page_title="t-Test | Análisis de Precios",
    page_icon="📊",
    layout="wide"
)

# ── CSS — compatible modo oscuro y claro ─────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #001D3D, #003566);
        color: white !important;
        padding: 1.4rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white !important; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: #aec6e8 !important; margin: 0.3rem 0 0; font-size: 0.9rem; }

    .step-card {
        border: 1px solid #F15B2B;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.5rem;
        height: 100%;
    }
    .step-card .step-title {
        color: #F15B2B;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.6rem;
    }

    .verdict-yes {
        background-color: #1b5e20;
        border-left: 5px solid #69f0ae;
        border-radius: 6px;
        padding: 1rem 1.4rem;
        color: #e8f5e9 !important;
        font-size: 1rem;
    }
    .verdict-yes b { color: #b9f6ca !important; font-size: 1.1rem; }

    .verdict-no {
        background-color: #bf360c;
        border-left: 5px solid #ffccbc;
        border-radius: 6px;
        padding: 1rem 1.4rem;
        color: #fbe9e7 !important;
        font-size: 1rem;
    }
    .verdict-no b { color: #ffccbc !important; font-size: 1.1rem; }

    .pval-box {
        border: 1px solid #444;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .pval-label { font-size: 0.82rem; color: #aaa; margin-bottom: 4px; }
    .pval-num-green { font-size: 2rem; font-weight: 700; color: #69f0ae; }
    .pval-num-orange { font-size: 2rem; font-weight: 700; color: #ffab40; }
    .pval-alpha { font-size: 0.78rem; color: #888; margin-top: 4px; }

    .orange-header {
        color: #F15B2B;
        font-weight: 700;
        font-size: 1.05rem;
        border-bottom: 2px solid #F15B2B;
        padding-bottom: 0.3rem;
        margin-bottom: 0.8rem;
    }
    .info-pill {
        display: inline-block;
        background: #1a3a5c;
        color: #aec6e8;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
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
        options=[
            "1 · Pareada (mismos productos, antes/después)",
            "2 · Dos grupos — varianzas iguales",
            "3 · Dos grupos — varianzas desiguales (Welch)"
        ]
    )

    alpha = st.slider(
        "Nivel de significancia (α)",
        min_value=0.01, max_value=0.10,
        value=0.05, step=0.01, format="%.2f"
    )

    st.markdown("---")
    st.markdown("### 📂 Cargar datos")

    # UN SOLO archivo a la vez — evita el bug de múltiples archivos
    archivo = st.file_uploader(
        "Arrastra tu CSV o Excel aquí",
        type=["csv", "xlsx"],
        accept_multiple_files=False
    )

    st.markdown("---")
    st.markdown("**Estructura esperada:**")
    st.markdown("""
    <div style='font-size:0.8rem; color:#aaa; line-height:1.7'>
    <b style='color:#F15B2B'>Variante 1 (Pareada):</b><br>
    Producto | Margen_Antes_% | Margen_Despues_%<br><br>
    <b style='color:#F15B2B'>Variante 2 (Var. iguales):</b><br>
    Producto | Margen_A_% | Margen_B_%<br><br>
    <b style='color:#F15B2B'>Variante 3 (Welch):</b><br>
    Producto | Margen_Estandar_% | Margen_Premium_%
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# DATOS DE PRUEBA EMBEBIDOS
# ════════════════════════════════════════════════════════════════════════════
CSV_PAREADA = """Producto,Margen_Antes_%,Margen_Despues_%
Agua mineral 1L,18.5,24.2
Refresco cola 600ml,22.0,27.5
Jugo naranja 1L,15.3,20.8
Leche entera 1L,20.1,25.6
Yogurt natural 200g,17.8,23.1
Pan integral 500g,14.2,19.7
Aceite oliva 500ml,28.5,34.0
Arroz blanco 1kg,12.4,17.9
Frijol negro 1kg,11.8,17.3
Avena 500g,16.3,21.8"""

CSV_VAR_IGUALES = """Producto,Margen_Lacteos_%,Margen_Bebidas_%
Producto 1,25.2,18.3
Producto 2,26.8,19.7
Producto 3,24.5,17.8
Producto 4,27.1,20.2
Producto 5,25.9,18.9
Producto 6,26.3,19.4
Producto 7,24.8,18.1
Producto 8,27.5,20.5
Producto 9,25.6,19.0
Producto 10,26.0,18.6"""

CSV_VAR_DESIGUALES = """Producto,Margen_Estandar_%,Margen_Premium_%
Producto 1,22.1,38.5
Producto 2,21.8,12.0
Producto 3,22.4,44.2
Producto 4,22.0,18.7
Producto 5,21.9,41.0
Producto 6,22.3,15.3
Producto 7,22.1,42.8
Producto 8,21.7,10.5
Producto 9,22.2,39.6
Producto 10,22.0,22.4"""

# ════════════════════════════════════════════════════════════════════════════
# FUNCIONES
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
    elif variante_num == 2:
        t_stat, p_two = stats.ttest_ind(grupo_a, grupo_b, equal_var=True)
        df_val = len(grupo_a) + len(grupo_b) - 2
        var_pool = ((len(grupo_a)-1)*grupo_a.var(ddof=1) +
                    (len(grupo_b)-1)*grupo_b.var(ddof=1)) / df_val
        extra = {"Varianza agrupada (pooled)": round(var_pool, 4)}
    else:
        t_stat, p_two = stats.ttest_ind(grupo_a, grupo_b, equal_var=False)
        df_val = round(len(grupo_a) + len(grupo_b) - 2)
        extra = {}

    t_crit = stats.t.ppf(1 - alpha/2, df_val)

    return {
        "col_a": col_a, "col_b": col_b,
        "grupo_a": grupo_a, "grupo_b": grupo_b,
        "media_a": round(grupo_a.mean(), 2),
        "media_b": round(grupo_b.mean(), 2),
        "var_a":   round(grupo_a.var(ddof=1), 4),
        "var_b":   round(grupo_b.var(ddof=1), 4),
        "n":       len(grupo_a),
        "t_stat":  round(t_stat, 4),
        "p_two":   p_two,
        "t_crit":  round(t_crit, 4),
        "df":      df_val,
        "significativa": p_two < alpha,
        "extra":   extra,
        "alpha":   alpha,
    }

def interpretar(r, variante_num):
    dif = round(r["media_b"] - r["media_a"], 2)
    col_a = r["col_a"].replace("_", " ")
    col_b = r["col_b"].replace("_", " ")
    p_fmt = f"{r['p_two']:.6f}" if r['p_two'] > 0.0001 else f"{r['p_two']:.2e}"

    if r["significativa"]:
        if variante_num == 1:
            dir_ = "subió" if dif > 0 else "bajó"
            return (f"El ajuste de precio {dir_} el margen en {abs(dif):.2f} pp. "
                    f"Con p-value = {p_fmt} (< {r['alpha']}), esta diferencia es "
                    f"estadísticamente real — no fue casualidad.")
        else:
            return (f"Los grupos '{col_a}' y '{col_b}' tienen márgenes estadísticamente distintos "
                    f"({r['media_a']}% vs {r['media_b']}%). "
                    f"Con p-value = {p_fmt} (< {r['alpha']}), la diferencia no es azar.")
    else:
        if variante_num == 3:
            return (f"Aunque las medias difieren ({r['media_a']}% vs {r['media_b']}%), "
                    f"la alta variabilidad de uno de los grupos impide afirmar que la "
                    f"diferencia sea real. p-value = {p_fmt} (> {r['alpha']}). "
                    f"Se necesitan más datos o reducir la dispersión.")
        return (f"No hay evidencia estadística suficiente. "
                f"p-value = {p_fmt} (> {r['alpha']}). La diferencia observada pudo ser azar.")

def grafico_barras(r, df):
    productos = df.iloc[:, 0].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=r["col_a"].replace("_", " "),
        x=productos, y=r["grupo_a"],
        marker_color="#1565C0",
        text=[f"{v:.1f}%" for v in r["grupo_a"]],
        textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name=r["col_b"].replace("_", " "),
        x=productos, y=r["grupo_b"],
        marker_color="#F15B2B",
        text=[f"{v:.1f}%" for v in r["grupo_b"]],
        textposition="outside"
    ))
    fig.update_layout(
        barmode="group",
        title="Comparación de márgenes por producto",
        xaxis_title="Producto",
        yaxis_title="Margen (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12, color="#cccccc"),
        height=420,
        margin=dict(t=60, b=80)
    )
    fig.update_xaxes(tickangle=-30, showgrid=False, color="#aaa")
    fig.update_yaxes(showgrid=True, gridcolor="#333", color="#aaa")
    return fig

def grafico_boxplot(r):
    fig = go.Figure()
    for vals, name, color in [
        (r["grupo_a"], r["col_a"], "#1565C0"),
        (r["grupo_b"], r["col_b"], "#F15B2B")
    ]:
        fig.add_trace(go.Box(
            y=vals, name=name.replace("_", " "),
            marker_color=color,
            boxmean=True,
            jitter=0.3, pointpos=-1.8,
            marker=dict(size=7, opacity=0.7)
        ))
    fig.update_layout(
        title="Distribución de márgenes por grupo",
        yaxis_title="Margen (%)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12, color="#cccccc"),
        height=400
    )
    fig.update_yaxes(showgrid=True, gridcolor="#333", color="#aaa")
    return fig

def tabla_resultados(r):
    filas = [
        ("Media Grupo A (%)",        f"{r['media_a']}"),
        ("Media Grupo B (%)",        f"{r['media_b']}"),
        ("Diferencia de medias",     f"{round(r['media_b']-r['media_a'],2)} pp"),
        ("Varianza Grupo A",         f"{r['var_a']}"),
        ("Varianza Grupo B",         f"{r['var_b']}"),
        ("Observaciones (n)",        f"{r['n']}"),
    ]
    for k, v in r["extra"].items():
        filas.append((k, str(v)))
    filas += [
        ("Grados de libertad (df)",  f"{r['df']}"),
        ("Estadístico t",            f"{r['t_stat']}"),
        ("t crítico (dos colas)",    f"{r['t_crit']}"),
        ("|t| > t crítico",          "Sí ✓" if abs(r['t_stat']) > r['t_crit'] else "No ✗"),
        ("P-value (dos colas)",      f"{r['p_two']:.10f}"),
        ("Nivel α",                  f"{r['alpha']}"),
        ("Conclusión",               "SIGNIFICATIVA ✅" if r['significativa'] else "NO SIGNIFICATIVA ⚠️"),
    ]
    return pd.DataFrame(filas, columns=["Parámetro", "Valor"])

# ════════════════════════════════════════════════════════════════════════════
# PANTALLA SIN ARCHIVO — bienvenida + descargas
# ════════════════════════════════════════════════════════════════════════════
if archivo is None:
    st.markdown('<p class="orange-header">¿Cómo usar esta herramienta?</p>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, num, titulo, cuerpo in [
        (c1, "1", "Elige la variante",
         "Selecciona en el panel izquierdo cuál de las 3 pruebas t aplica a tu situación.<br><br>"
         "• <b>Pareada:</b> mismos productos antes/después<br>"
         "• <b>Var. iguales:</b> dos categorías similares<br>"
         "• <b>Welch:</b> un grupo muy volátil"),
        (c2, "2", "Carga tu archivo",
         "Arrastra un CSV o Excel con exactamente 3 columnas:<br><br>"
         "• Columna 1: nombre del producto<br>"
         "• Columna 2: margen del grupo A<br>"
         "• Columna 3: margen del grupo B"),
        (c3, "3", "Lee el resultado",
         "La herramienta calcula automáticamente:<br><br>"
         "• p-value y veredicto en español<br>"
         "• Estadístico t y t crítico<br>"
         "• Gráficas comparativas<br>"
         "• Tabla completa exportable"),
    ]:
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-title">Paso {num} — {titulo}</div>
                <div style="font-size:0.9rem; line-height:1.6">{cuerpo}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="orange-header">Regla de oro del p-value</p>',
                unsafe_allow_html=True)
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.success("**p-value < 0.05 → Diferencia REAL**  \nLa diferencia no es casualidad. Puedes tomar decisiones con base en ella.")
    with col_r2:
        st.warning("**p-value ≥ 0.05 → No significativa**  \nNo hay evidencia suficiente. La diferencia pudo ser azar.")

    st.markdown("---")
    st.markdown('<p class="orange-header">📥 Archivos de prueba — descárgalos y arrástralos</p>',
                unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.caption("Variante 1 — Pareada")
        st.download_button(
            "⬇️ ttest_1_pareada.csv",
            data=CSV_PAREADA.encode("utf-8"),
            file_name="ttest_1_pareada.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d2:
        st.caption("Variante 2 — Varianzas iguales")
        st.download_button(
            "⬇️ ttest_2_var_iguales.csv",
            data=CSV_VAR_IGUALES.encode("utf-8"),
            file_name="ttest_2_var_iguales.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d3:
        st.caption("Variante 3 — Welch")
        st.download_button(
            "⬇️ ttest_3_var_desiguales.csv",
            data=CSV_VAR_DESIGUALES.encode("utf-8"),
            file_name="ttest_3_var_desiguales.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("""
    <div style="margin-top:1rem; padding:0.8rem 1.2rem; background:#1a2a3a;
                border-radius:8px; font-size:0.85rem; color:#aec6e8;">
        💡 <b>Tip:</b> Descarga cualquier archivo, ábrelo en Excel, cambia los números
        con tus propios datos, guárdalo y arrástralo aquí. La herramienta funciona
        con cualquier base que tenga esa estructura de 3 columnas.
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# ANÁLISIS — con archivo cargado
# ════════════════════════════════════════════════════════════════════════════
try:
    df = leer_archivo(archivo)
except Exception as e:
    st.error(f"❌ Error al leer el archivo: {e}")
    st.stop()

if df.shape[1] < 3:
    st.error("El archivo debe tener al menos 3 columnas: Producto, Grupo A, Grupo B.")
    st.stop()

variante_num = get_variante_num(variante)

try:
    r = correr_ttest(df, variante_num, alpha)
except Exception as e:
    st.error(f"❌ Error al correr el t-Test: {e}. Verifica que las columnas 2 y 3 sean numéricas.")
    st.stop()

# ── Nombre variante ──────────────────────────────────────────────────────
nombres = {
    1: "Prueba t Pareada (Paired Two Sample for Means)",
    2: "Dos Muestras — Varianzas Iguales",
    3: "Dos Muestras — Varianzas Desiguales (Welch)"
}
st.markdown(f'<p class="orange-header">📋 {nombres[variante_num]}</p>',
            unsafe_allow_html=True)
st.markdown(f'<span class="info-pill">📁 {archivo.name} &nbsp;·&nbsp; {r["n"]} productos &nbsp;·&nbsp; α = {alpha}</span>',
            unsafe_allow_html=True)

# ── 4 métricas ───────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Media Grupo A", f"{r['media_a']}%")
m2.metric("Media Grupo B", f"{r['media_b']}%",
          delta=f"{round(r['media_b']-r['media_a'],2)} pp")
m3.metric("Estadístico t", f"{r['t_stat']}")
m4.metric("t crítico (α)", f"{r['t_crit']}")

st.markdown("---")

# ── Veredicto + p-value ──────────────────────────────────────────────────
cv, cp = st.columns([3, 1])
interpretacion = interpretar(r, variante_num)

with cv:
    if r["significativa"]:
        st.markdown(f"""
        <div class="verdict-yes">
            <b>✅ DIFERENCIA SIGNIFICATIVA</b><br>
            <span style="font-weight:400">{interpretacion}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-no">
            <b>⚠️ NO SIGNIFICATIVA</b><br>
            <span style="font-weight:400">{interpretacion}</span>
        </div>""", unsafe_allow_html=True)

with cp:
    p_display = f"{r['p_two']:.6f}" if r['p_two'] > 0.0001 else f"{r['p_two']:.2e}"
    cls = "pval-num-green" if r["significativa"] else "pval-num-orange"
    st.markdown(f"""
    <div class="pval-box">
        <div class="pval-label">p-value (dos colas)</div>
        <div class="{cls}">{p_display}</div>
        <div class="pval-alpha">umbral α = {alpha}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Pestañas ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Comparación por producto",
    "📦 Distribución (Boxplot)",
    "📋 Tabla de resultados"
])

with tab1:
    st.plotly_chart(grafico_barras(r, df), use_container_width=True)

with tab2:
    st.plotly_chart(grafico_boxplot(r), use_container_width=True)
    i1, i2 = st.columns(2)
    with i1:
        razon = round(max(r['var_a'], r['var_b']) / max(min(r['var_a'], r['var_b']), 0.0001), 2)
        st.info(f"**Varianza Grupo A:** {r['var_a']}  \n"
                f"**Varianza Grupo B:** {r['var_b']}  \n"
                f"**Razón de varianzas:** {razon}x")
        if variante_num == 3 and razon > 4:
            st.success("✓ Razón > 4x confirma que Welch es la variante correcta.")
    with i2:
        if "Correlación de Pearson" in r["extra"]:
            corr = r["extra"]["Correlación de Pearson"]
            nivel = "Alta" if corr > 0.7 else "Moderada" if corr > 0.4 else "Baja"
            st.info(f"**Correlación de Pearson:** {corr}  \n"
                    f"**Nivel:** {nivel}  \n"
                    f"{'Ideal para prueba pareada.' if corr > 0.7 else 'Verificar si la prueba pareada es apropiada.'}")

with tab3:
    df_res = tabla_resultados(r)
    st.dataframe(df_res, use_container_width=True, hide_index=True, height=420)
    st.download_button(
        "⬇️ Descargar resultados como CSV",
        data=df_res.to_csv(index=False).encode("utf-8"),
        file_name=f"ttest_resultados_{archivo.name.replace('.xlsx','.csv')}",
        mime="text/csv"
    )

# ── Datos cargados ───────────────────────────────────────────────────────
with st.expander("🔍 Ver datos originales"):
    st.dataframe(df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#666; font-size:0.78rem;
            margin-top:2rem; padding-top:1rem; border-top:1px solid #333">
    Universidad Panamericana · IA para el Análisis Financiero · Ejercicio 15 RD3
</div>
""", unsafe_allow_html=True)
