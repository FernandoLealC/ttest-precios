# ============================================================
#  t-Test Analizador de Cambios de Precio — Ejercicio 15 RD3
#  Universidad Panamericana — IA para el Análisis Financiero
#  v3 — Análisis múltiple (hasta 3 archivos) + comentarios
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
    padding: 1.4rem 2rem; border-radius: 10px; margin-bottom: 1.5rem;
}
.main-header h1 { color: white !important; margin: 0; font-size: 1.7rem; }
.main-header p  { color: #aec6e8 !important; margin: 0.3rem 0 0; font-size: 0.88rem; }

.file-card {
    border-radius: 10px; padding: 1.2rem 1.4rem; margin-bottom: 1.4rem;
}
.file-card-1 { border: 2px solid #0F6E8E; }
.file-card-2 { border: 2px solid #F15B2B; }
.file-card-3 { border: 2px solid #1B7F4F; }

.file-header-1 { color: #0F6E8E; font-weight:700; font-size:1rem; border-bottom:2px solid #0F6E8E; padding-bottom:0.3rem; margin-bottom:0.8rem; }
.file-header-2 { color: #F15B2B; font-weight:700; font-size:1rem; border-bottom:2px solid #F15B2B; padding-bottom:0.3rem; margin-bottom:0.8rem; }
.file-header-3 { color: #1B7F4F; font-weight:700; font-size:1rem; border-bottom:2px solid #1B7F4F; padding-bottom:0.3rem; margin-bottom:0.8rem; }

.verdict-yes {
    background-color:#1b5e20; border-left:5px solid #69f0ae;
    border-radius:6px; padding:0.8rem 1.2rem; color:#e8f5e9 !important;
}
.verdict-yes b { color:#b9f6ca !important; }
.verdict-no {
    background-color:#bf360c; border-left:5px solid #ffccbc;
    border-radius:6px; padding:0.8rem 1.2rem; color:#fbe9e7 !important;
}
.verdict-no b { color:#ffccbc !important; }

.pval-box { border:1px solid #444; border-radius:10px; padding:0.9rem; text-align:center; }
.pval-label { font-size:0.78rem; color:#aaa; margin-bottom:3px; }
.pval-green  { font-size:1.8rem; font-weight:700; color:#69f0ae; }
.pval-orange { font-size:1.8rem; font-weight:700; color:#ffab40; }
.pval-alpha  { font-size:0.75rem; color:#888; margin-top:3px; }

.comment-box {
    background:#0d2137; border:1px solid #1565C0;
    border-radius:8px; padding:0.9rem 1.2rem;
    font-size:0.88rem; color:#90caf9; margin-top:0.8rem;
}
.comment-box b { color:#64b5f6; }

.detection-box {
    background:#0a1a2a; border:1px solid #334; border-radius:8px;
    padding:0.6rem 1rem; font-size:0.82rem; color:#90caf9; margin-bottom:0.8rem;
}
.orange-header {
    color:#F15B2B; font-weight:700; font-size:1.05rem;
    border-bottom:2px solid #F15B2B; padding-bottom:0.3rem; margin-bottom:0.8rem;
}
.step-card { border:1px solid #F15B2B; border-radius:10px; padding:1.1rem 1.3rem; height:100%; }
.step-title { color:#F15B2B; font-weight:700; font-size:0.95rem; margin-bottom:0.5rem; }

.summary-row {
    display:flex; gap:10px; margin-bottom:0.5rem; align-items:center;
}
.summary-badge {
    font-size:11px; font-weight:600; padding:3px 10px;
    border-radius:20px; white-space:nowrap; flex-shrink:0;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 t-Test — Analizador de Cambios de Precio</h1>
    <p>Ejercicio 15 RD3 · Universidad Panamericana · IA para el Análisis Financiero · Hasta 3 análisis simultáneos</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")
    alpha = st.slider("Nivel de significancia (α)", 0.01, 0.10, 0.05, 0.01,
                      format="%.2f", help="Estándar = 0.05")
    st.markdown("---")
    st.markdown("### 📂 Cargar archivos")
    st.caption("Puedes cargar hasta 3 archivos a la vez")

    archivos = st.file_uploader(
        "Arrastra tus CSV o Excel aquí",
        type=["csv","xlsx"],
        accept_multiple_files=True,
        help="Máximo 3 archivos. Cada uno debe tener 3 columnas."
    )

    if archivos and len(archivos) > 3:
        st.warning("Solo se procesan los primeros 3 archivos.")
        archivos = archivos[:3]

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem;color:#aaa;line-height:1.8'>
    <b style='color:#F15B2B'>Estructura de cada archivo:</b><br>
    Columna 1: Producto / ítem<br>
    Columna 2: Valor grupo A<br>
    Columna 3: Valor grupo B<br><br>
    <b style='color:#F15B2B'>Detección automática:</b><br>
    🔵 Pareada → palabras antes/después<br>
    🟡 Welch → varianzas muy distintas<br>
    🟢 Var. iguales → resto de casos
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

# ── 3 Bases nuevas de práctica ────────────────────────────────────────────
CSV_PRACTICA_1 = """Sucursal,Ventas_Pre_Campaña_MXN,Ventas_Post_Campaña_MXN
Sucursal Norte,142500,158900
Sucursal Sur,98700,121400
Sucursal Centro,205000,218600
Sucursal Oriente,87300,105200
Sucursal Poniente,163400,177800
Sucursal Aeropuerto,312000,345600
Sucursal Periferia,74200,89500
Sucursal Express 1,56800,71300
Sucursal Express 2,61500,78900
Sucursal Plaza,189000,203400"""

CSV_PRACTICA_2 = """Servicio,Tiempo_Respuesta_Operador_A_min,Tiempo_Respuesta_Operador_B_min
Asistencia vial,18.5,17.2
Grúa pesada,42.3,41.8
Cerrajería,25.1,24.9
Plomería urgente,38.7,37.5
Electricidad,31.2,30.8
Vidrios,28.4,27.6
Médico en casa,55.3,54.1
Ambulancia,12.8,12.5
Asistencia hogar,33.6,32.9
Fumigación,47.2,46.8"""

CSV_PRACTICA_3 = """Producto,Costo_Proveedor_Consolidado_MXN,Costo_Proveedor_Nuevo_MXN
Refacción motor A,1250,980
Refacción motor B,875,1420
Kit frenos,2340,1890
Aceite sintético 4L,385,510
Filtro aire,145,88
Batería 12V,1890,2340
Llanta 195/65R15,1650,1240
Cable diagnóstico,780,1150
Sensor ABS,2100,890
Bomba agua,1430,1980"""

# ════════════════════════════════════════════════════════════════════════════
# FUNCIONES CORE
# ════════════════════════════════════════════════════════════════════════════
PALABRAS_ANTES   = ["antes","before","pre","inicial","base","original","previo"]
PALABRAS_DESPUES = ["despues","después","after","post","final","nuevo","nueva","posterior"]

def detectar_variante(df, alpha):
    col_a = df.columns[1].lower()
    col_b = df.columns[2].lower()
    tiene_antes   = any(p in col_a for p in PALABRAS_ANTES)
    tiene_despues = any(p in col_b for p in PALABRAS_DESPUES)
    if tiene_antes and tiene_despues:
        return 1, "Columnas con palabras clave **antes/después** detectadas", None
    grupo_a = df.iloc[:,1].astype(float)
    grupo_b = df.iloc[:,2].astype(float)
    var_a, var_b = grupo_a.var(ddof=1), grupo_b.var(ddof=1)
    if var_a == 0 or var_b == 0:
        return 2, "Una varianza es cero — se asumen varianzas iguales", None
    f_stat = var_a/var_b if var_a >= var_b else var_b/var_a
    df1, df2 = len(grupo_a)-1, len(grupo_b)-1
    p_f = 2 * min(stats.f.cdf(f_stat,df1,df2), 1-stats.f.cdf(f_stat,df1,df2))
    razon = round(max(var_a,var_b)/min(var_a,var_b),1)
    det = {"f_stat":round(f_stat,4),"p_ftest":p_f,"var_a":round(var_a,4),"var_b":round(var_b,4),"razon":razon}
    if p_f < alpha:
        return 3, f"F-Test detectó varianzas muy distintas (razón {razon}x, p={p_f:.4f})", det
    return 2, f"F-Test confirmó varianzas similares (razón {razon}x, p={p_f:.4f})", det

def correr_ttest(df, variante_num, alpha):
    col_a   = df.columns[1]
    col_b   = df.columns[2]
    grupo_a = df[col_a].astype(float)
    grupo_b = df[col_b].astype(float)
    if variante_num == 1:
        t_stat, p_two = stats.ttest_rel(grupo_a, grupo_b)
        df_val = len(grupo_a)-1
        extra  = {"Correlación de Pearson": round(grupo_a.corr(grupo_b),4)}
    elif variante_num == 2:
        t_stat, p_two = stats.ttest_ind(grupo_a, grupo_b, equal_var=True)
        df_val  = len(grupo_a)+len(grupo_b)-2
        vp = ((len(grupo_a)-1)*grupo_a.var(ddof=1)+(len(grupo_b)-1)*grupo_b.var(ddof=1))/df_val
        extra   = {"Varianza agrupada (pooled)": round(vp,4)}
    else:
        t_stat, p_two = stats.ttest_ind(grupo_a, grupo_b, equal_var=False)
        df_val  = len(grupo_a)+len(grupo_b)-2
        extra   = {}
    t_crit = stats.t.ppf(1-alpha/2, df_val)
    return {
        "col_a":col_a,"col_b":col_b,"grupo_a":grupo_a,"grupo_b":grupo_b,
        "media_a":round(grupo_a.mean(),2),"media_b":round(grupo_b.mean(),2),
        "var_a":round(grupo_a.var(ddof=1),4),"var_b":round(grupo_b.var(ddof=1),4),
        "n":len(grupo_a),"t_stat":round(t_stat,4),"p_two":p_two,
        "t_crit":round(t_crit,4),"df":df_val,
        "significativa":p_two<alpha,"extra":extra,"alpha":alpha,
    }

def interpretar(r, variante_num):
    dif  = round(r["media_b"]-r["media_a"],2)
    p_fmt = f"{r['p_two']:.6f}" if r['p_two']>0.0001 else f"{r['p_two']:.2e}"
    col_a = r["col_a"].replace("_"," ")
    col_b = r["col_b"].replace("_"," ")
    if r["significativa"]:
        if variante_num == 1:
            dir_ = "subió" if dif>0 else "bajó"
            return (f"El cambio {dir_} el valor promedio en {abs(dif):.2f} unidades. "
                    f"Con p-value = {p_fmt} (< {r['alpha']}), esta diferencia es "
                    f"estadísticamente real — no fue casualidad.")
        return (f"'{col_a}' y '{col_b}' tienen valores estadísticamente distintos "
                f"({r['media_a']} vs {r['media_b']}). "
                f"p-value = {p_fmt} (< {r['alpha']}): la diferencia no es azar.")
    else:
        if variante_num == 3:
            return (f"Aunque las medias difieren ({r['media_a']} vs {r['media_b']}), "
                    f"la alta variabilidad de un grupo impide confirmarlo. "
                    f"p-value = {p_fmt} (> {r['alpha']}). Necesitas más datos o reducir dispersión.")
        return (f"No hay evidencia estadística suficiente para afirmar diferencia real. "
                f"p-value = {p_fmt} (> {r['alpha']}). La diferencia observada pudo ser azar.")

def comentario_negocio(r, variante_num, nombre_archivo):
    """Genera un comentario de negocio breve y accionable según el resultado."""
    dif  = round(r["media_b"]-r["media_a"],2)
    sig  = r["significativa"]
    col_a = r["col_a"].replace("_"," ")
    col_b = r["col_b"].replace("_"," ")

    if variante_num == 1:
        if sig:
            if dif > 0:
                return (f"**Acción recomendada:** El cambio tuvo un impacto positivo real de {abs(dif):.2f} unidades "
                        f"en promedio. Considera aplicar la misma estrategia al resto de la cartera "
                        f"y documenta el momento del cambio para replicarlo.")
            else:
                return (f"**Alerta:** El cambio redujo el valor promedio en {abs(dif):.2f} unidades. "
                        f"La caída es estadísticamente real — revisa si fue un efecto esperado "
                        f"o si hay que corregir la estrategia.")
        else:
            return (f"**Observación:** No se puede afirmar que el cambio haya tenido efecto real. "
                    f"El negocio puede estar operando con variaciones naturales. "
                    f"Amplía la muestra a más períodos antes de tomar decisiones.")

    elif variante_num == 2:
        if sig:
            mayor = col_a if r["media_a"] > r["media_b"] else col_b
            return (f"**Acción recomendada:** '{mayor}' tiene rendimiento estadísticamente superior. "
                    f"Considera reasignar recursos o inversión hacia este segmento. "
                    f"La diferencia de {abs(dif):.2f} unidades es real y consistente.")
        else:
            return (f"**Observación:** Ambos grupos tienen rendimiento estadísticamente equivalente. "
                    f"No hay base para priorizar uno sobre el otro con estos datos. "
                    f"Evalúa otros criterios como volumen, costo operativo o potencial de crecimiento.")

    else:  # Welch
        if sig:
            mayor = col_a if r["media_a"] > r["media_b"] else col_b
            return (f"**Acción recomendada:** Aunque hay alta variabilidad, la diferencia es real. "
                    f"'{mayor}' tiene mejor rendimiento promedio. "
                    f"Trabaja en reducir la volatilidad del grupo menos consistente antes de escalar.")
        else:
            return (f"**Observación:** La enorme diferencia de variabilidad entre grupos "
                    f"(razón {round(max(r['var_a'],r['var_b'])/max(min(r['var_a'],r['var_b']),0.001),1)}x) "
                    f"absorbe cualquier diferencia de medias. "
                    f"Primero estabiliza el grupo volátil, luego compara de nuevo.")

def grafico_barras(r, df, color_a, color_b):
    productos = df.iloc[:,0].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Bar(name=r["col_a"].replace("_"," "), x=productos, y=r["grupo_a"],
        marker_color=color_a, text=[f"{v:.1f}" for v in r["grupo_a"]], textposition="outside"))
    fig.add_trace(go.Bar(name=r["col_b"].replace("_"," "), x=productos, y=r["grupo_b"],
        marker_color=color_b, text=[f"{v:.1f}" for v in r["grupo_b"]], textposition="outside"))
    fig.update_layout(
        barmode="group", title="Comparación por ítem",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial",size=11,color="#ccc"),
        height=360, margin=dict(t=50,b=70),
        legend=dict(orientation="h",y=1.1,x=1,xanchor="right")
    )
    fig.update_xaxes(tickangle=-30, showgrid=False, color="#aaa")
    fig.update_yaxes(showgrid=True, gridcolor="#333", color="#aaa")
    return fig

def grafico_boxplot(r, color_a, color_b):
    fig = go.Figure()
    for vals, name, color in [(r["grupo_a"],r["col_a"],color_a),(r["grupo_b"],r["col_b"],color_b)]:
        fig.add_trace(go.Box(y=vals, name=name.replace("_"," "), marker_color=color,
            boxmean=True, jitter=0.3, pointpos=-1.8, marker=dict(size=6,opacity=0.7)))
    fig.update_layout(
        title="Distribución de valores",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial",size=11,color="#ccc"), height=340
    )
    fig.update_yaxes(showgrid=True, gridcolor="#333", color="#aaa")
    return fig

def tabla_res(r, variante_num):
    nombres = {1:"Prueba t Pareada",2:"Dos Muestras — Varianzas Iguales",3:"Welch (Varianzas Desiguales)"}
    filas = [
        ("Variante aplicada", nombres[variante_num]),
        ("Media Grupo A", f"{r['media_a']}"),
        ("Media Grupo B", f"{r['media_b']}"),
        ("Diferencia de medias", f"{round(r['media_b']-r['media_a'],2)}"),
        ("Varianza Grupo A", f"{r['var_a']}"),
        ("Varianza Grupo B", f"{r['var_b']}"),
        ("Observaciones (n)", f"{r['n']}"),
    ]
    for k,v in r["extra"].items():
        filas.append((k, str(v)))
    filas += [
        ("Grados de libertad", f"{r['df']}"),
        ("Estadístico t", f"{r['t_stat']}"),
        ("t crítico (dos colas)", f"{r['t_crit']}"),
        ("|t| > t crítico", "Sí ✓" if abs(r['t_stat'])>r['t_crit'] else "No ✗"),
        ("P-value (dos colas)", f"{r['p_two']:.10f}"),
        ("Conclusión", "SIGNIFICATIVA ✅" if r['significativa'] else "NO SIGNIFICATIVA ⚠️"),
    ]
    return pd.DataFrame(filas, columns=["Parámetro","Valor"])

# ════════════════════════════════════════════════════════════════════════════
# PANTALLA SIN ARCHIVOS
# ════════════════════════════════════════════════════════════════════════════
if not archivos:
    st.markdown('<p class="orange-header">¿Cómo usar esta herramienta?</p>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col,(num,titulo,cuerpo) in zip([c1,c2,c3],[
        ("1","Ajusta el α (opcional)","El valor estándar es <b>0.05</b>. Solo cámbialo si tu profesor indica otro."),
        ("2","Carga 1, 2 o 3 archivos","Arrastra todos los archivos a la vez. Cada uno necesita 3 columnas: ítem · grupo A · grupo B."),
        ("3","Lee los 3 resultados","La app detecta la variante de cada archivo automáticamente y muestra veredicto + comentario de negocio."),
    ]):
        with col:
            st.markdown(f'<div class="step-card"><div class="step-title">Paso {num} — {titulo}</div><div style="font-size:0.88rem;line-height:1.7">{cuerpo}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="orange-header">📥 Archivos de prueba y práctica — descarga y arrastra</p>', unsafe_allow_html=True)

    st.caption("**Archivos del ejercicio RD3 (bases originales)**")
    d1,d2,d3 = st.columns(3)
    with d1:
        st.caption("Variante 1 — Pareada")
        st.download_button("⬇️ ttest_1_pareada.csv", CSV_PAREADA.encode(), "ttest_1_pareada.csv", "text/csv", use_container_width=True)
    with d2:
        st.caption("Variante 2 — Varianzas iguales")
        st.download_button("⬇️ ttest_2_var_iguales.csv", CSV_VAR_IGUALES.encode(), "ttest_2_var_iguales.csv", "text/csv", use_container_width=True)
    with d3:
        st.caption("Variante 3 — Welch")
        st.download_button("⬇️ ttest_3_var_desiguales.csv", CSV_VAR_DESIGUALES.encode(), "ttest_3_var_desiguales.csv", "text/csv", use_container_width=True)

    st.markdown("---")
    st.caption("**Bases adicionales de práctica (casos empresariales diferentes)**")
    p1,p2,p3 = st.columns(3)
    with p1:
        st.caption("Práctica A — Campaña de ventas por sucursal")
        st.download_button("⬇️ practica_A_sucursales.csv", CSV_PRACTICA_1.encode(), "practica_A_sucursales.csv", "text/csv", use_container_width=True)
    with p2:
        st.caption("Práctica B — Tiempo de respuesta operadores")
        st.download_button("⬇️ practica_B_operadores.csv", CSV_PRACTICA_2.encode(), "practica_B_operadores.csv", "text/csv", use_container_width=True)
    with p3:
        st.caption("Práctica C — Costo proveedores consolidados vs nuevos")
        st.download_button("⬇️ practica_C_proveedores.csv", CSV_PRACTICA_3.encode(), "practica_C_proveedores.csv", "text/csv", use_container_width=True)

    st.markdown("""
    <div style="margin-top:1rem;padding:0.8rem 1.2rem;background:#0d2137;
                border-radius:8px;font-size:0.85rem;color:#90caf9;border:1px solid #1565C0">
        💡 <b>Tip:</b> Para probar los 3 análisis simultáneos descarga los 3 archivos de práctica,
        selecciónalos todos juntos al arrastrar al uploader. Cada uno producirá su propio análisis independiente.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# RESUMEN RÁPIDO si hay más de 1 archivo
# ════════════════════════════════════════════════════════════════════════════
COLORES = [("#1565C0","#0F6E8E","1"),("#C0392B","#F15B2B","2"),("#1B7F4F","#27AE60","3")]
NOMBRES_VARIANTE = {1:"Pareada",2:"Var. Iguales",3:"Welch"}

if len(archivos) > 1:
    st.markdown(f'<p class="orange-header">📋 Resumen — {len(archivos)} archivos analizados</p>', unsafe_allow_html=True)
    cols_sum = st.columns(len(archivos))
    resultados_previos = []

    for idx, archivo in enumerate(archivos):
        try:
            df_s = pd.read_csv(archivo) if archivo.name.endswith(".csv") else pd.read_excel(archivo)
            archivo.seek(0)
            var_num_s, _, _ = detectar_variante(df_s, alpha)
            r_s = correr_ttest(df_s, var_num_s, alpha)
            resultados_previos.append((df_s, var_num_s, r_s))
            with cols_sum[idx]:
                color_sig = "#1b5e20" if r_s["significativa"] else "#7f3300"
                bg_sig    = "#e8f5e9" if r_s["significativa"] else "#fff3e0"
                txt_sig   = "✅ SIGNIFICATIVA" if r_s["significativa"] else "⚠️ NO SIGNIFICATIVA"
                p_fmt = f"{r_s['p_two']:.4f}" if r_s['p_two']>0.0001 else f"{r_s['p_two']:.2e}"
                st.markdown(f"""
                <div style="background:{bg_sig};border-radius:8px;padding:12px;text-align:center">
                  <div style="font-size:11px;color:#666;margin-bottom:4px">{archivo.name}</div>
                  <div style="font-size:13px;font-weight:600;color:{color_sig}">{txt_sig}</div>
                  <div style="font-size:11px;color:#888;margin-top:4px">
                    {NOMBRES_VARIANTE[var_num_s]} · p = {p_fmt}
                  </div>
                </div>""", unsafe_allow_html=True)
        except:
            resultados_previos.append(None)
            with cols_sum[idx]:
                st.error(f"Error al leer {archivo.name}")

    st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# ANÁLISIS INDIVIDUAL POR ARCHIVO
# ════════════════════════════════════════════════════════════════════════════
COLOR_PAIRS = [("#1565C0","#0F6E8E"),("#C0392B","#F15B2B"),("#1B7F4F","#27AE60")]
CARD_CLASSES = ["1","2","3"]

for idx, archivo in enumerate(archivos):
    ca, cb = COLOR_PAIRS[idx]
    cc = CARD_CLASSES[idx]

    st.markdown(f'<div class="file-card file-card-{cc}">', unsafe_allow_html=True)
    st.markdown(f'<p class="file-header-{cc}">📁 Archivo {idx+1} — {archivo.name}</p>', unsafe_allow_html=True)

    try:
        archivo.seek(0)
        df = pd.read_csv(archivo) if archivo.name.endswith(".csv") else pd.read_excel(archivo)
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        continue

    if df.shape[1] < 3:
        st.error("El archivo necesita al menos 3 columnas: Ítem · Grupo A · Grupo B")
        st.markdown("</div>", unsafe_allow_html=True)
        continue

    try:
        variante_num, razon_det, detalle_f = detectar_variante(df, alpha)
        r = correr_ttest(df, variante_num, alpha)
    except Exception as e:
        st.error(f"Error en el análisis: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        continue

    # Detección
    NOMBRES_V = {1:"Variante 1 · Prueba t Pareada",2:"Variante 2 · Dos Muestras — Varianzas Iguales",3:"Variante 3 · Welch"}
    ftest_txt = ""
    if detalle_f:
        ftest_txt = f' · F-Test: F={detalle_f["f_stat"]}, p={detalle_f["p_ftest"]:.4f}'
    st.markdown(f"""
    <div class="detection-box">
        🔍 <b>Variante detectada:</b> {NOMBRES_V[variante_num]}<br>
        <span style="color:#64b5f6;font-size:0.78rem">↳ {razon_det}{ftest_txt}</span>
    </div>""", unsafe_allow_html=True)

    # 4 métricas
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Media Grupo A", f"{r['media_a']}")
    m2.metric("Media Grupo B", f"{r['media_b']}", delta=f"{round(r['media_b']-r['media_a'],2)}")
    m3.metric("Estadístico t", f"{r['t_stat']}")
    m4.metric("t crítico", f"{r['t_crit']}")

    # Veredicto + p-value
    cv, cp = st.columns([3,1])
    interp = interpretar(r, variante_num)
    with cv:
        if r["significativa"]:
            st.markdown(f'<div class="verdict-yes"><b>✅ DIFERENCIA SIGNIFICATIVA</b><br><span style="font-weight:400;font-size:0.92rem">{interp}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-no"><b>⚠️ NO SIGNIFICATIVA</b><br><span style="font-weight:400;font-size:0.92rem">{interp}</span></div>', unsafe_allow_html=True)
    with cp:
        p_display = f"{r['p_two']:.6f}" if r['p_two']>0.0001 else f"{r['p_two']:.2e}"
        cls = "pval-green" if r["significativa"] else "pval-orange"
        st.markdown(f"""<div class="pval-box">
            <div class="pval-label">p-value (dos colas)</div>
            <div class="{cls}">{p_display}</div>
            <div class="pval-alpha">umbral α = {alpha}</div>
        </div>""", unsafe_allow_html=True)

    # Comentario de negocio
    comentario = comentario_negocio(r, variante_num, archivo.name)
    st.markdown(f'<div class="comment-box">💼 <b>Comentario de negocio:</b><br>{comentario}</div>', unsafe_allow_html=True)

    # Gráficas y tabla en tabs
    t1,t2,t3 = st.tabs(["📊 Comparación","📦 Distribución","📋 Tabla"])
    with t1:
        st.plotly_chart(grafico_barras(r, df, ca, cb), use_container_width=True)
    with t2:
        st.plotly_chart(grafico_boxplot(r, ca, cb), use_container_width=True)
        i1,i2 = st.columns(2)
        with i1:
            razon = round(max(r['var_a'],r['var_b'])/max(min(r['var_a'],r['var_b']),0.0001),2)
            st.info(f"**Var. A:** {r['var_a']}  ·  **Var. B:** {r['var_b']}  ·  **Razón:** {razon}x")
            if variante_num==3 and razon>4:
                st.success("✓ Razón >4x confirma que Welch es la variante correcta.")
        with i2:
            if "Correlación de Pearson" in r["extra"]:
                corr = r["extra"]["Correlación de Pearson"]
                nivel = "Alta" if corr>0.7 else "Moderada" if corr>0.4 else "Baja"
                st.info(f"**Correlación Pearson:** {corr}  ({nivel})")
    with t3:
        df_res = tabla_res(r, variante_num)
        st.dataframe(df_res, use_container_width=True, hide_index=True, height=420)
        st.download_button(
            f"⬇️ Descargar resultados — {archivo.name}",
            data=df_res.to_csv(index=False).encode("utf-8"),
            file_name=f"ttest_resultado_{idx+1}_{archivo.name.replace('.xlsx','.csv')}",
            mime="text/csv"
        )

    with st.expander("🔍 Ver datos originales"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)
    if idx < len(archivos)-1:
        st.markdown("---")

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#555;font-size:0.75rem;
            margin-top:2rem;padding-top:1rem;border-top:1px solid #333">
    Universidad Panamericana · IA para el Análisis Financiero · Ejercicio 15 RD3
</div>
""", unsafe_allow_html=True)
