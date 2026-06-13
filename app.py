# ============================================================
#  t-Test Analizador de Cambios de Precio — Ejercicio 15 RD3
#  Universidad Panamericana — IA para el Análisis Financiero
#  Versión con detección automática de variante
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go

st.set_page_config(
    page_title="t-Test | Análisis de Precios",
    page_icon="📊",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #001D3D, #003566);
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
    height: 100%;
}
.step-title { color: #F15B2B; font-weight: 700; font-size: 1rem; margin-bottom: 0.6rem; }

.verdict-yes {
    background-color: #1b5e20;
    border-left: 5px solid #69f0ae;
    border-radius: 6px;
    padding: 1rem 1.4rem;
    color: #e8f5e9 !important;
}
.verdict-yes b { color: #b9f6ca !important; font-size: 1.05rem; }

.verdict-no {
    background-color: #bf360c;
    border-left: 5px solid #ffccbc;
    border-radius: 6px;
    padding: 1rem 1.4rem;
    color: #fbe9e7 !important;
}
.verdict-no b { color: #ffccbc !important; font-size: 1.05rem; }

.badge-auto {
    display: inline-block;
    background: #003566;
    color: #aec6e8;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    margin-bottom: 0.8rem;
    border: 1px solid #F15B2B;
}
.pval-box {
    border: 1px solid #444;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.pval-label { font-size: 0.82rem; color: #aaa; margin-bottom: 4px; }
.pval-green { font-size: 2rem; font-weight: 700; color: #69f0ae; }
.pval-orange { font-size: 2rem; font-weight: 700; color: #ffab40; }
.pval-alpha { font-size: 0.78rem; color: #888; margin-top: 4px; }

.orange-header {
    color: #F15B2B;
    font-weight: 700;
    font-size: 1.05rem;
    border-bottom: 2px solid #F15B2B;
    padding-bottom: 0.3rem;
    margin-bottom: 0.8rem;
}
.detection-box {
    background: #0d2137;
    border: 1px solid #1565C0;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-size: 0.85rem;
    color: #90caf9;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 t-Test — Analizador de Cambios de Precio</h1>
    <p>Ejercicio 15 RD3 · Universidad Panamericana · IA para el Análisis Financiero · Detección automática de variante</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")

    alpha = st.slider(
        "Nivel de significancia (α)",
        min_value=0.01, max_value=0.10,
        value=0.05, step=0.01, format="%.2f",
        help="Estándar = 0.05. La app detecta la variante automáticamente."
    )

    st.markdown("---")
    st.markdown("### 📂 Cargar datos")

    archivo = st.file_uploader(
        "Arrastra tu CSV o Excel aquí",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
        help="3 columnas: Producto · Grupo A · Grupo B"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.82rem; color:#aaa; line-height:1.8'>
    <b style='color:#F15B2B'>La app detecta automáticamente:</b><br><br>
    🔵 <b style='color:#90caf9'>Variante 1 — Pareada</b><br>
    Si los nombres de columna contienen:<br>
    <i>antes, before, pre, despues, after, post</i><br><br>
    🟡 <b style='color:#ffcc80'>Variante 3 — Welch</b><br>
    Si el F-Test detecta varianzas distintas<br>
    (p-value F &lt; α)<br><br>
    🟢 <b style='color:#a5d6a7'>Variante 2 — Var. iguales</b><br>
    En todos los demás casos
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
# DETECCIÓN AUTOMÁTICA DE VARIANTE
# ════════════════════════════════════════════════════════════════════════════
PALABRAS_ANTES   = ["antes", "before", "pre", "inicial", "base", "original"]
PALABRAS_DESPUES = ["despues", "después", "after", "post", "final", "nuevo", "nueva"]

def detectar_variante(df, alpha):
    """
    Retorna (variante_num, razon, detalle_ftest)
    Paso 1: revisa nombres de columnas → Pareada
    Paso 2: corre F-Test automático → Welch o Iguales
    """
    col_a = df.columns[1].lower()
    col_b = df.columns[2].lower()

    tiene_antes   = any(p in col_a for p in PALABRAS_ANTES)
    tiene_despues = any(p in col_b for p in PALABRAS_DESPUES)

    if tiene_antes and tiene_despues:
        return 1, "Columnas detectadas con palabras clave **antes/después**", None

    # F-Test para decidir entre variante 2 y 3
    grupo_a = df.iloc[:, 1].astype(float)
    grupo_b = df.iloc[:, 2].astype(float)
    var_a = grupo_a.var(ddof=1)
    var_b = grupo_b.var(ddof=1)

    if var_a == 0 or var_b == 0:
        return 2, "Una varianza es cero — se asumen varianzas iguales", None

    f_stat = var_a / var_b if var_a >= var_b else var_b / var_a
    df1 = len(grupo_a) - 1
    df2 = len(grupo_b) - 1
    p_ftest = 2 * min(
        stats.f.cdf(f_stat, df1, df2),
        1 - stats.f.cdf(f_stat, df1, df2)
    )

    razon_var = round(max(var_a, var_b) / min(var_a, var_b), 1)
    detalle = {
        "f_stat": round(f_stat, 4),
        "p_ftest": p_ftest,
        "var_a": round(var_a, 4),
        "var_b": round(var_b, 4),
        "razon": razon_var
    }

    if p_ftest < alpha:
        return 3, f"F-Test detectó varianzas distintas (razón {razon_var}x, p={p_ftest:.4f})", detalle
    else:
        return 2, f"F-Test confirmó varianzas similares (razón {razon_var}x, p={p_ftest:.4f})", detalle

# ════════════════════════════════════════════════════════════════════════════
# T-TEST
# ════════════════════════════════════════════════════════════════════════════
def correr_ttest(df, variante_num, alpha):
    col_a   = df.columns[1]
    col_b   = df.columns[2]
    grupo_a = df[col_a].astype(float)
    grupo_b = df[col_b].astype(float)

    if variante_num == 1:
        t_stat, p_two = stats.ttest_rel(grupo_a, grupo_b)
        df_val = len(grupo_a) - 1
        extra  = {"Correlación de Pearson": round(grupo_a.corr(grupo_b), 4)}
    elif variante_num == 2:
        t_stat, p_two = stats.ttest_ind(grupo_a, grupo_b, equal_var=True)
        df_val  = len(grupo_a) + len(grupo_b) - 2
        var_p   = ((len(grupo_a)-1)*grupo_a.var(ddof=1) +
                   (len(grupo_b)-1)*grupo_b.var(ddof=1)) / df_val
        extra   = {"Varianza agrupada (pooled)": round(var_p, 4)}
    else:
        t_stat, p_two = stats.ttest_ind(grupo_a, grupo_b, equal_var=False)
        df_val = len(grupo_a) + len(grupo_b) - 2
        extra  = {}

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
    dif  = round(r["media_b"] - r["media_a"], 2)
    p_fmt = f"{r['p_two']:.6f}" if r['p_two'] > 0.0001 else f"{r['p_two']:.2e}"
    col_a = r["col_a"].replace("_", " ")
    col_b = r["col_b"].replace("_", " ")

    if r["significativa"]:
        if variante_num == 1:
            dir_ = "subió" if dif > 0 else "bajó"
            return (f"El ajuste de precio {dir_} el margen en {abs(dif):.2f} pp. "
                    f"Con p-value = {p_fmt} (< {r['alpha']}), esta diferencia es "
                    f"estadísticamente real — no fue casualidad.")
        return (f"'{col_a}' y '{col_b}' tienen márgenes estadísticamente distintos "
                f"({r['media_a']}% vs {r['media_b']}%). "
                f"Con p-value = {p_fmt} (< {r['alpha']}), la diferencia no es azar.")
    else:
        if variante_num == 3:
            return (f"Aunque las medias difieren ({r['media_a']}% vs {r['media_b']}%), "
                    f"la alta variabilidad de un grupo impide afirmar diferencia real. "
                    f"p-value = {p_fmt} (> {r['alpha']}). Se necesitan más datos.")
        return (f"No hay evidencia estadística suficiente. "
                f"p-value = {p_fmt} (> {r['alpha']}). La diferencia pudo ser azar.")

# ════════════════════════════════════════════════════════════════════════════
# GRÁFICAS
# ════════════════════════════════════════════════════════════════════════════
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
        xaxis_title="Producto", yaxis_title="Margen (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12, color="#cccccc"),
        height=420, margin=dict(t=60, b=80)
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
            marker_color=color, boxmean=True,
            jitter=0.3, pointpos=-1.8,
            marker=dict(size=7, opacity=0.7)
        ))
    fig.update_layout(
        title="Distribución de márgenes por grupo",
        yaxis_title="Margen (%)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12, color="#cccccc"),
        height=400
    )
    fig.update_yaxes(showgrid=True, gridcolor="#333", color="#aaa")
    return fig

def tabla_resultados(r, variante_num):
    nombres_variante = {
        1: "Prueba t Pareada",
        2: "Dos Muestras — Varianzas Iguales",
        3: "Dos Muestras — Varianzas Desiguales (Welch)"
    }
    filas = [
        ("Variante aplicada",         nombres_variante[variante_num]),
        ("Media Grupo A (%)",         f"{r['media_a']}"),
        ("Media Grupo B (%)",         f"{r['media_b']}"),
        ("Diferencia de medias",      f"{round(r['media_b']-r['media_a'],2)} pp"),
        ("Varianza Grupo A",          f"{r['var_a']}"),
        ("Varianza Grupo B",          f"{r['var_b']}"),
        ("Observaciones (n)",         f"{r['n']}"),
    ]
    for k, v in r["extra"].items():
        filas.append((k, str(v)))
    filas += [
        ("Grados de libertad (df)",   f"{r['df']}"),
        ("Estadístico t",             f"{r['t_stat']}"),
        ("t crítico (dos colas)",     f"{r['t_crit']}"),
        ("|t| > t crítico",           "Sí ✓" if abs(r['t_stat']) > r['t_crit'] else "No ✗"),
        ("P-value (dos colas)",       f"{r['p_two']:.10f}"),
        ("Nivel α",                   f"{r['alpha']}"),
        ("Conclusión",                "SIGNIFICATIVA ✅" if r['significativa'] else "NO SIGNIFICATIVA ⚠️"),
    ]
    return pd.DataFrame(filas, columns=["Parámetro", "Valor"])

# ════════════════════════════════════════════════════════════════════════════
# PANTALLA SIN ARCHIVO
# ════════════════════════════════════════════════════════════════════════════
if archivo is None:
    st.markdown('<p class="orange-header">¿Cómo usar esta herramienta?</p>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    pasos = [
        ("1", "Ajusta el α (opcional)",
         "El nivel de significancia está en el panel izquierdo.<br>"
         "El valor estándar es <b>0.05</b> — no necesitas cambiarlo salvo que tu profesor indique otro."),
        ("2", "Carga tu archivo",
         "Arrastra un CSV o Excel con <b>3 columnas</b>:<br><br>"
         "• Columna 1: nombre del producto<br>"
         "• Columna 2: margen del grupo A<br>"
         "• Columna 3: margen del grupo B"),
        ("3", "Lee el resultado automático",
         "La app detecta sola cuál variante aplicar y muestra:<br><br>"
         "• Variante detectada y por qué<br>"
         "• Veredicto en español<br>"
         "• p-value, gráficas y tabla exportable"),
    ]
    for col, (num, titulo, cuerpo) in zip([c1, c2, c3], pasos):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-title">Paso {num} — {titulo}</div>
                <div style="font-size:0.9rem; line-height:1.7">{cuerpo}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="orange-header">Regla de oro del p-value</p>',
                unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    with r1:
        st.success("**p-value < 0.05 → Diferencia REAL**  \nConfía en la diferencia y toma decisiones con base en ella.")
    with r2:
        st.warning("**p-value ≥ 0.05 → No significativa**  \nLa diferencia pudo ser azar. Necesitas más datos.")

    st.markdown("---")
    st.markdown('<p class="orange-header">📥 Archivos de prueba — descárgalos y arrástralos</p>',
                unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    archivos_prueba = [
        (d1, "Variante 1 — Pareada",          "⬇️ ttest_1_pareada.csv",
         CSV_PAREADA, "ttest_1_pareada.csv"),
        (d2, "Variante 2 — Varianzas iguales", "⬇️ ttest_2_var_iguales.csv",
         CSV_VAR_IGUALES, "ttest_2_var_iguales.csv"),
        (d3, "Variante 3 — Welch",             "⬇️ ttest_3_var_desiguales.csv",
         CSV_VAR_DESIGUALES, "ttest_3_var_desiguales.csv"),
    ]
    for col, caption, label, data, fname in archivos_prueba:
        with col:
            st.caption(caption)
            st.download_button(label, data=data.encode("utf-8"),
                               file_name=fname, mime="text/csv",
                               use_container_width=True)

    st.markdown("""
    <div style="margin-top:1rem; padding:0.8rem 1.2rem; background:#0d2137;
                border-radius:8px; font-size:0.85rem; color:#90caf9; border:1px solid #1565C0">
        💡 <b>Tip:</b> Descarga cualquier archivo, cámbia los números con tus propios datos,
        guárdalo y arrástralo. La app detecta automáticamente qué prueba aplicar.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# ANÁLISIS CON ARCHIVO
# ════════════════════════════════════════════════════════════════════════════
try:
    df = pd.read_csv(archivo) if archivo.name.endswith(".csv") else pd.read_excel(archivo)
except Exception as e:
    st.error(f"❌ Error al leer el archivo: {e}")
    st.stop()

if df.shape[1] < 3:
    st.error("El archivo necesita al menos 3 columnas: Producto · Grupo A · Grupo B")
    st.stop()

# ── Detección automática ─────────────────────────────────────────────────
try:
    variante_num, razon_deteccion, detalle_ftest = detectar_variante(df, alpha)
except Exception as e:
    st.error(f"Error en detección automática: {e}")
    st.stop()

# ── Mostrar variante detectada ───────────────────────────────────────────
nombres_variante = {
    1: "Variante 1 · Prueba t Pareada (Paired Two Sample for Means)",
    2: "Variante 2 · Dos Muestras — Varianzas Iguales",
    3: "Variante 3 · Dos Muestras — Varianzas Desiguales (Welch)"
}
colores_variante = {1: "#1D9E75", 2: "#185FA5", 3: "#BA7517"}
iconos_variante  = {1: "🔵", 2: "🟢", 3: "🟡"}

st.markdown(f"""
<div class="detection-box">
    {iconos_variante[variante_num]} <b>Variante detectada automáticamente:</b>
    {nombres_variante[variante_num]}<br>
    <span style="color:#64b5f6">↳ Razón: {razon_deteccion}</span>
    {f'<br><span style="color:#888; font-size:0.8rem">F-Test: F={detalle_ftest["f_stat"]}, p={detalle_ftest["p_ftest"]:.4f}, razón de varianzas={detalle_ftest["razon"]}x</span>' if detalle_ftest else ''}
</div>
""", unsafe_allow_html=True)

# ── Correr t-Test ────────────────────────────────────────────────────────
try:
    r = correr_ttest(df, variante_num, alpha)
except Exception as e:
    st.error(f"Error al correr el t-Test: {e}. Verifica que las columnas 2 y 3 sean numéricas.")
    st.stop()

st.markdown(f'<p style="color:#F15B2B; font-weight:700; font-size:1.05rem; '
            f'border-bottom:2px solid #F15B2B; padding-bottom:0.3rem">'
            f'📋 {nombres_variante[variante_num]}</p>', unsafe_allow_html=True)
st.caption(f"📁 {archivo.name}  ·  {r['n']} productos  ·  α = {alpha}")

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
            <span style="font-weight:400; font-size:0.95rem">{interpretacion}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-no">
            <b>⚠️ NO SIGNIFICATIVA</b><br>
            <span style="font-weight:400; font-size:0.95rem">{interpretacion}</span>
        </div>""", unsafe_allow_html=True)

with cp:
    p_display = f"{r['p_two']:.6f}" if r['p_two'] > 0.0001 else f"{r['p_two']:.2e}"
    cls = "pval-green" if r["significativa"] else "pval-orange"
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
                    f"**Nivel:** {nivel} — "
                    f"{'ideal para prueba pareada.' if corr > 0.7 else 'verificar si la prueba pareada es apropiada.'}")

with tab3:
    df_res = tabla_resultados(r, variante_num)
    st.dataframe(df_res, use_container_width=True, hide_index=True, height=440)
    st.download_button(
        "⬇️ Descargar resultados como CSV",
        data=df_res.to_csv(index=False).encode("utf-8"),
        file_name=f"ttest_resultado_{archivo.name.replace('.xlsx','.csv')}",
        mime="text/csv"
    )

with st.expander("🔍 Ver datos originales cargados"):
    st.dataframe(df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#555; font-size:0.78rem;
            margin-top:2rem; padding-top:1rem; border-top:1px solid #333">
    Universidad Panamericana · IA para el Análisis Financiero · Ejercicio 15 RD3
</div>
""", unsafe_allow_html=True)
