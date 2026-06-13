# 📊 t-Test — Analizador de Cambios de Precio

**Ejercicio 15 · RD3 · Universidad Panamericana**  
Curso: IA para el Análisis Financiero

---

## ¿Qué hace esta herramienta?

Permite cargar un archivo CSV o Excel con márgenes de productos y determinar automáticamente si los cambios de precio generaron diferencias estadísticamente significativas.

Soporta las **3 variantes del t-Test**:

| Variante | Cuándo usarla |
|---|---|
| 1 · Pareada | Los mismos productos medidos antes y después del ajuste de precio |
| 2 · Varianzas iguales | Dos categorías distintas con dispersión de margen parecida |
| 3 · Welch | Dos grupos donde uno es mucho más volátil que el otro |

---

## Cómo usar la herramienta

**Paso 1** — Selecciona la variante del t-Test en el panel izquierdo  
**Paso 2** — Arrastra tu archivo CSV o Excel  
**Paso 3** — Lee el veredicto y las gráficas

---

## Estructura del archivo de entrada

El archivo debe tener exactamente **3 columnas**:

```
Producto | Grupo A | Grupo B
```

### Archivos de prueba incluidos en el repositorio

| Archivo | Variante |
|---|---|
| `ttest_1_pareada.csv` | Variante 1 — Pareada |
| `ttest_2_var_iguales.csv` | Variante 2 — Varianzas iguales |
| `ttest_3_var_desiguales.csv` | Variante 3 — Welch |

---

## Instalación local

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/ttest-precios.git
cd ttest-precios

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Correr la app
streamlit run app.py
```

---

## Despliegue en Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Conecta tu cuenta de GitHub
3. Selecciona este repositorio
4. Archivo principal: `app.py`
5. Clic en **Deploy**

---

## Archivos del repositorio

```
ttest-precios/
│
├── app.py                        ← Aplicación principal
├── requirements.txt              ← Dependencias
├── README.md                     ← Este archivo
│
├── ttest_1_pareada.csv           ← Datos de prueba · Variante 1
├── ttest_2_var_iguales.csv       ← Datos de prueba · Variante 2
└── ttest_3_var_desiguales.csv    ← Datos de prueba · Variante 3
```

---

## Resultado que genera

- ✅ / ⚠️ Veredicto en español (Significativa / No significativa)
- p-value destacado en grande
- Métricas: media, varianza, t Stat, t crítico, grados de libertad
- Gráfica de barras comparativa por producto
- Boxplot de distribución de márgenes
- Tabla completa exportable como CSV

---

*Universidad Panamericana · IA para el Análisis Financiero · Sesión 10 · Junio 2026*
