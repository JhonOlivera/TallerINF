"""
Análisis de datos - Global YouTube Statistics
Descriptivo -> Diagnóstico -> Predictivo -> Prescriptivo

Cómo ejecutarlo en VS Code:
  1. Pon este archivo y "Global YouTube Statistics.csv" en la misma carpeta.
  2. Instala las librerías (en la terminal de VS Code):
       pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
  3. Ejecuta:  python analisis_youtube.py
     (o usa "Run Cell" en cada bloque # %% con la extensión de Python/Jupyter)
  Las gráficas se guardan en la carpeta "graficas/" y también se abren en pantalla
  (cierra cada ventana para que siga con la siguiente).
"""

# %% [markdown]
# # Análisis de datos — Global YouTube Statistics
# **Analítica:** Descriptiva → Diagnóstica → Predictiva → Prescriptiva
#
# Dataset: 995 canales más grandes de YouTube (suscriptores, vistas, categoría, país, ingresos estimados, año de creación, etc.).

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')
import os

# Carpeta del script (así encuentra el CSV aunque ejecutes desde otra carpeta)
BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE, 'graficas'), exist_ok=True)
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 20)
contador = 0

def mostrar(nombre):
    """Guarda la gráfica en graficas/ y la muestra."""
    global contador
    contador += 1
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, 'graficas', f'{contador:02d}_{nombre}.png'), dpi=120)
    plt.show()
    plt.close('all')

sns.set_theme(style='whitegrid')
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

# %% [markdown]
# ## 0. Carga y limpieza

# %%
df = pd.read_csv(os.path.join(BASE, 'Global YouTube Statistics.csv'), encoding='latin-1')
df.columns = df.columns.str.replace('ï»¿', '', regex=False).str.strip()
print('Filas y columnas originales:', df.shape)

# Limpieza
df = df[df['video views'] > 0]                       # canales con 0 vistas (datos erróneos)
df = df[(df['created_year'] >= 2005) | df['created_year'].isna()]  # YouTube nace en 2005 (hay un 1970)
df['category'] = df['category'].fillna('Sin categoría')
df['Country'] = df['Country'].fillna('Desconocido')

# Variables derivadas
df['antiguedad'] = 2023 - df['created_year']                  # el dataset es de 2023
df['vistas_por_sub'] = df['video views'] / df['subscribers']
df['vistas_por_video'] = df['video views'] / df['uploads'].replace(0, np.nan)
df['ingreso_anual_medio'] = (df['lowest_yearly_earnings'] + df['highest_yearly_earnings']) / 2

print('Filas tras limpieza:', df.shape)
print(df.isna().sum()[lambda s: s > 0])

# %% [markdown]
# ---
# # 1. ANÁLISIS DESCRIPTIVO (EDA)
# ## 1.1 Resumen de variables numéricas

# %%
num = ['subscribers', 'video views', 'uploads', 'video_views_for_the_last_30_days',
       'ingreso_anual_medio', 'created_year']
desc = df[num].describe().T
desc['asimetria'] = df[num].skew()
print(desc)

# %% [markdown]
# ### Hallazgo numérico 1 — Suscriptores y vistas: muy concentrados y fuertemente relacionados

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(df['subscribers'] / 1e6, bins=40, ax=ax[0])
ax[0].set(title='Distribución de suscriptores', xlabel='Suscriptores (millones)')
sns.scatterplot(data=df, x='subscribers', y='video views', alpha=.5, ax=ax[1])
ax[1].set(xscale='log', yscale='log', title='Suscriptores vs vistas (escala log)')
mostrar('suscriptores_vistas')

print(f"Media suscriptores:   {df['subscribers'].mean()/1e6:.1f} M")
print(f"Mediana suscriptores: {df['subscribers'].median()/1e6:.1f} M")
top10 = df.nlargest(int(len(df)*.10), 'subscribers')['subscribers'].sum() / df['subscribers'].sum()
print(f"El 10% de canales más grandes concentra el {top10:.1%} de los suscriptores")
print(f"Correlación Spearman suscriptores-vistas: {df['subscribers'].corr(df['video views'], method='spearman'):.2f}")

# %% [markdown]
# **Hallazgo N1.** Suscriptores y vistas tienen distribución con **sesgo positivo fuerte** (asimetría > 5): la media (22,6 M) es mayor que la mediana (17,7 M) porque unos pocos gigantes (T-Series, MrBeast…) jalan el promedio. Además, ambas variables están **positivamente relacionadas** (Spearman ≈ 0,59): a más suscriptores, más vistas, pero la relación no es perfecta — hay canales con las mismas suscripciones y vistas muy distintas.

# %% [markdown]
# ### Hallazgo numérico 2 — Los ingresos son extremadamente desiguales

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(df['ingreso_anual_medio'] / 1e6, bins=50, ax=ax[0])
ax[0].set(title='Ingreso anual estimado (punto medio)', xlabel='USD (millones)')
sns.boxplot(x=df['ingreso_anual_medio'] / 1e6, ax=ax[1])
ax[1].set(title='Boxplot ingreso anual', xlabel='USD (millones)')
mostrar('ingresos')

ing = df['ingreso_anual_medio']
print(f"Media:   {ing.mean()/1e6:.2f} M USD")
print(f"Mediana: {ing.median()/1e6:.2f} M USD")
print(f"Canales con ingreso 0: {(ing == 0).sum()} ({(ing == 0).mean():.1%})")
q1, q3 = ing.quantile([.25, .75])
print(f"Outliers (> Q3 + 1.5·IQR): {(ing > q3 + 1.5*(q3-q1)).sum()}")
print(f"El 10% que más gana se lleva el {df.nlargest(int(len(df)*.10), 'ingreso_anual_medio')['ingreso_anual_medio'].sum()/ing.sum():.1%} de los ingresos")

# %% [markdown]
# **Hallazgo N2.** Los ingresos anuales estimados son **muy desiguales**: media 3,8 M USD vs mediana 1,38 M USD, ~98 outliers y el **10% que más gana se queda con ~54% de todos los ingresos**. Además, 74 canales (7,5%) aparecen con ingreso 0 pese a estar en el top.

# %% [markdown]
# ### Hallazgo categórico 1 — Entertainment y Music dominan, y Music tiene más vistas por canal

# %%
cat = df['category'].value_counts()
print((cat / len(df) * 100).round(1).head(8).astype(str) + ' %')

top_cat = cat.head(8).index
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
cat.head(8).plot.barh(ax=ax[0]); ax[0].invert_yaxis(); ax[0].set(title='Canales por categoría (top 8)', xlabel='N.º canales')
(df[df['category'].isin(top_cat)].groupby('category')['video views'].median() / 1e9).sort_values().plot.barh(ax=ax[1])
ax[1].set(title='Mediana de vistas por categoría', xlabel='Vistas (miles de millones)')
mostrar('categorias')

# %% [markdown]
# **Hallazgo C1.** **Entertainment (24%) y Music (20%)** suman casi la mitad de los canales top. Pero Entertainment tiene más canales, mientras que **Music tiene la mediana de vistas más alta** de las categorías grandes (~12 mil millones por canal).

# %% [markdown]
# ### Hallazgo categórico 2 — Estados Unidos e India concentran más de la mitad de los canales

# %%
pais = df.loc[df['Country'] != 'Desconocido', 'Country'].value_counts()
print((pais / pais.sum() * 100).round(1).head(8).astype(str) + ' %')
print(f"\nEE. UU. + India = {(pais.iloc[:2].sum() / pais.sum()):.1%} de los canales con país conocido")

fig, ax = plt.subplots(figsize=(9, 4.5))
pais.head(10).plot.bar(ax=ax); ax.set(title='Canales por país (top 10)', ylabel='N.º canales')
plt.xticks(rotation=45, ha='right'); mostrar('paises')

# %% [markdown]
# **Hallazgo C2.** **Estados Unidos (36%) e India (19%)** concentran ~55% de los canales con país conocido; el tercero, Brasil, apenas tiene 7%. (Hay ~12% de canales sin país registrado, que se excluyen de este conteo.)

# %% [markdown]
# ---
# # 2. ANÁLISIS DIAGNÓSTICO — ¿por qué ocurre?
# Se buscan **otras variables** que expliquen cada hallazgo.

# %% [markdown]
# ### 2.1 ¿Por qué hay canales con muchas más vistas? (explica Hallazgo N1)

# %%
vars_ = ['video views', 'subscribers', 'uploads', 'antiguedad', 'video_views_for_the_last_30_days', 'ingreso_anual_medio']
plt.figure(figsize=(8, 6))
sns.heatmap(df[vars_].corr(method='spearman'), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlación de Spearman'); mostrar('correlacion')

# Antigüedad: canales más viejos acumulan más vistas
df['grupo_antig'] = pd.cut(df['antiguedad'], [0, 6, 10, 14, 20], labels=['≤6 años', '7-10', '11-14', '15+'])
print((df.groupby('grupo_antig', observed=True)['video views'].median() / 1e9).round(2).rename('mediana vistas (mil M)'))

# %% [markdown]
# **Diagnóstico N1.** Las vistas se explican principalmente por los **suscriptores**. La **cantidad de videos (uploads) casi no influye** (correlación ≈ 0): subir más videos no garantiza más vistas. La **antigüedad** sí ayuda: los canales de 15+ años tienen una mediana de ~10 mil M de vistas vs ~7–8 mil M en los más jóvenes, porque han tenido más tiempo para acumular. La dispersión que queda se explica por el tipo de contenido (ver 2.3).

# %% [markdown]
# ### 2.2 ¿Por qué los ingresos son tan desiguales? (explica Hallazgo N2)

# %%
print(f"Spearman ingreso vs vistas últimos 30 días: {df['ingreso_anual_medio'].corr(df['video_views_for_the_last_30_days'], method='spearman'):.2f}")
print(f"Spearman ingreso vs suscriptores:          {df['ingreso_anual_medio'].corr(df['subscribers'], method='spearman'):.2f}")

cero = df['ingreso_anual_medio'] == 0
print('\nVistas últimos 30 días (mediana):')
print(f"  canales con ingreso 0:  {df.loc[cero, 'video_views_for_the_last_30_days'].median():,.0f}")
print(f"  canales con ingreso >0: {df.loc[~cero, 'video_views_for_the_last_30_days'].median():,.0f}")

d = df[(df['ingreso_anual_medio'] > 0) & (df['video_views_for_the_last_30_days'] > 0)]
sns.scatterplot(data=d, x='video_views_for_the_last_30_days', y='ingreso_anual_medio', alpha=.5)
plt.xscale('log'); plt.yscale('log'); plt.title('Vistas últimos 30 días vs ingreso anual'); mostrar('ingreso_vs_vistas30')

# %% [markdown]
# **Diagnóstico N2.** El ingreso depende casi por completo de las **vistas de los últimos 30 días** (Spearman ≈ 0,94), **no de los suscriptores** (≈ 0,27). Es decir, se gana por actividad reciente, no por tamaño histórico. Esto explica la desigualdad: pocos canales generan muchísimas vistas mensuales. Los canales con ingreso 0 **sí tienen vistas recientes** similares al resto, así que su 0 no se explica por inactividad sino probablemente por **no estar monetizados** o por datos faltantes de la fuente (limitación del dataset).

# %% [markdown]
# ### 2.3 ¿Por qué Music tiene más vistas con menos canales que Entertainment? (explica Hallazgo C1)

# %%
tabla = (df[df['category'].isin(top_cat)]
         .groupby('category')
         .agg(canales=('Youtuber', 'count'),
              uploads_mediana=('uploads', 'median'),
              vistas_por_video=('vistas_por_video', 'median'),
              vistas_por_sub=('vistas_por_sub', 'median'),
              antiguedad_media=('antiguedad', 'mean'))
         .sort_values('vistas_por_video', ascending=False))
print(tabla)

# %% [markdown]
# **Diagnóstico C1.** Music tiene **la menor cantidad de videos por canal (mediana ~250)** pero **las más vistas por video (~34 M)** y por suscriptor, y son los canales **más antiguos** (12,5 años en promedio): la música se re-escucha muchas veces y es consumida globalmente sin barrera de idioma. Entertainment, en cambio, domina por **cantidad de canales**, con más videos y menos vistas por video (~5 M): estrategia de volumen.

# %% [markdown]
# ### 2.4 ¿Por qué EE. UU. e India dominan? (explica Hallazgo C2)

# %%
paises = (df[df['Country'] != 'Desconocido']
          .groupby('Country')
          .agg(canales=('Youtuber', 'count'), poblacion=('Population', 'first'),
               urbana=('Urban_population', 'first'), educacion=('Gross tertiary education enrollment (%)', 'first'),
               ingreso_mediana=('ingreso_anual_medio', 'median'))
          .dropna())
print(f"Spearman n.º canales vs población:        {paises['canales'].corr(paises['poblacion'], method='spearman'):.2f}")
print(f"Spearman n.º canales vs población urbana: {paises['canales'].corr(paises['urbana'], method='spearman'):.2f}")
print(f"Spearman n.º canales vs educación sup.:   {paises['canales'].corr(paises['educacion'], method='spearman'):.2f}")
print()
print(pd.crosstab(df['Country'], df['category']).loc[['United States', 'India'], ['Entertainment', 'Music', 'People & Blogs', 'Gaming', 'Education']])
print(paises.sort_values('canales', ascending=False).head(6))

# %% [markdown]
# **Diagnóstico C2.** El número de canales top por país se relaciona con la **población urbana** (Spearman ≈ 0,63) y la **población total** (≈ 0,57), pero poco con la educación superior (≈ 0,22). EE. UU. combina mucha población urbana con alta conectividad y mercado publicitario; India aporta un **público gigantesco** (1.366 M habitantes). Ambos se apoyan sobre todo en Entertainment y Music. Curiosamente, la mediana de ingreso de los canales indios es mayor que la de EE. UU. porque sus canales top tienen volúmenes de vistas muy altos.

# %% [markdown]
# ---
# # 3. ANÁLISIS PREDICTIVO
# - Hallazgos N1 y N2 → **regresión lineal** (variables numéricas, escala log).
# - Hallazgos C1 y C2 → usan `created_year`, por eso se modelan como **serie de tiempo** (Holt / suavizamiento exponencial) del n.º de canales top creados por año.

# %% [markdown]
# ### 3.1 Predicción de vistas totales (Hallazgo N1)

# %%
m1 = df[(df['uploads'] > 0) & df['antiguedad'].notna()]
X = pd.concat([np.log10(m1[['subscribers', 'uploads']]),
               m1['antiguedad'],
               pd.get_dummies(m1['category'], prefix='cat', drop_first=True).astype(int)], axis=1)
y = np.log10(m1['video views'])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
reg1 = LinearRegression().fit(Xtr, ytr)
print(f"R² prueba: {r2_score(yte, reg1.predict(Xte)):.3f}")
coef = pd.Series(reg1.coef_, index=X.columns)
print(coef[['subscribers', 'uploads', 'antiguedad']].round(3))
print('\nEfecto de la categoría (vs la base), top 5:')
print(coef.filter(like='cat_').sort_values(ascending=False).head(5).round(3))

# Predicción para un canal nuevo
def predecir_vistas(subs, uploads, antig, categoria):
    fila = pd.DataFrame(0, index=[0], columns=X.columns)
    fila['subscribers'], fila['uploads'], fila['antiguedad'] = np.log10(subs), np.log10(uploads), antig
    if f'cat_{categoria}' in fila: fila[f'cat_{categoria}'] = 1
    return 10**reg1.predict(fila)[0]

for c in ['Music', 'Entertainment', 'Gaming']:
    print(f"30 M subs, 1000 videos, 10 años, {c:13s} → {predecir_vistas(30e6, 1000, 10, c)/1e9:.1f} mil M vistas")

# %% [markdown]
# **Predicción N1.** El modelo (log-log) con suscriptores, uploads, antigüedad y categoría explica ~26% de la variación de las vistas (R²). El coeficiente de suscriptores ≈ 1,2 indica que **+10% de suscriptores ≈ +12% de vistas**. Con los mismos 30 M de suscriptores, un canal de **Music** se proyecta con más vistas que uno de Entertainment o Gaming. El R² moderado indica que faltan variables (calidad del contenido, viralidad, idioma) que el dataset no tiene.

# %% [markdown]
# ### 3.2 Predicción de ingresos anuales (Hallazgo N2)

# %%
m2 = df[(df['ingreso_anual_medio'] > 0) & (df['video_views_for_the_last_30_days'] > 0)]
X2 = np.log10(m2[['video_views_for_the_last_30_days']])
y2 = np.log10(m2['ingreso_anual_medio'])
Xtr, Xte, ytr, yte = train_test_split(X2, y2, test_size=.2, random_state=42)
reg2 = LinearRegression().fit(Xtr, ytr)
pred = reg2.predict(Xte)
print(f"R² prueba: {r2_score(yte, pred):.3f}")
print(f"Error absoluto medio: {mean_absolute_error(10**yte, 10**pred)/1e6:.2f} M USD")
print(f"Elasticidad: +1% vistas mensuales → +{reg2.coef_[0]:.2f}% ingreso")

for v in [10e6, 100e6, 500e6]:
    p = 10**reg2.predict(pd.DataFrame({'video_views_for_the_last_30_days': [np.log10(v)]}))[0]
    print(f"  {v/1e6:>5.0f} M vistas/mes → {p/1e6:.2f} M USD/año")

# %% [markdown]
# **Predicción N2.** El modelo da R² ≈ 1 y elasticidad 1: **el ingreso crece en proporción exacta a las vistas mensuales** (~0,26 M USD/año por cada 10 M vistas/mes). Ojo: esto es casi perfecto porque la fuente (Social Blade) **calcula** el ingreso a partir de las vistas con un rango de CPM, así que el modelo reproduce esa fórmula. Sirve para estimar, pero no captura diferencias reales de CPM por país o nicho.

# %% [markdown]
# ### 3.3 Serie de tiempo — canales top creados por año por categoría (Hallazgo C1)

# %%
def pronostico(serie, pasos=4):
    modelo = ExponentialSmoothing(serie.values.astype(float), trend='add', damped_trend=True).fit()
    return pd.Series(modelo.forecast(pasos)).clip(lower=0)

# 2022 está incompleto (datos recogidos en 2023 y un canal tarda años en llegar al top) → se excluye
anios = range(2005, 2022)
fig, ax = plt.subplots(figsize=(11, 4.5))
for c in ['Entertainment', 'Music', 'People & Blogs', 'Gaming']:
    s = df[df['category'] == c].groupby('created_year').size().reindex(anios, fill_value=0)
    s.index = pd.Index(list(anios))
    f = pronostico(s); f.index = range(2022, 2026)
    l, = ax.plot(s.index, s.values, marker='o', label=c)
    ax.plot(f.index, f.values, '--', color=l.get_color())
    print(f"{c:15s} pronóstico 2022-2025: {f.round(1).tolist()}")
ax.axvline(2021.5, color='gray', ls=':'); ax.set(title='Canales top creados por año (— real, -- pronóstico)', xlabel='Año de creación', ylabel='N.º canales')
ax.legend(); mostrar('serie_categorias')

# %% [markdown]
# **Predicción C1 (serie de tiempo).** Se usó **suavizamiento exponencial de Holt con tendencia amortiguada** sobre el n.º de canales top creados por año (2005-2021). Entertainment y People & Blogs se proyectan **estables (~7–8 canales/año)**, mientras que **Music y Gaming tienden a ~0**: hace años que casi no nacen canales musicales que lleguen al top. Esto sugiere que el dominio de Music es **histórico** (canales viejos) y que el crecimiento futuro viene de Entertainment y People & Blogs.
#
# ⚠️ Limitación: hay **sesgo de supervivencia** — un canal creado en 2020 ha tenido poco tiempo para entrar al top, por eso los años recientes salen bajos en todas las series.

# %% [markdown]
# ### 3.4 Serie de tiempo — canales top creados por año por país (Hallazgo C2)

# %%
fig, ax = plt.subplots(figsize=(11, 4.5))
for p in ['United States', 'India', 'Brazil']:
    s = df[df['Country'] == p].groupby('created_year').size().reindex(anios, fill_value=0)
    s.index = pd.Index(list(anios))
    f = pronostico(s); f.index = range(2022, 2026)
    l, = ax.plot(s.index, s.values, marker='o', label=p)
    ax.plot(f.index, f.values, '--', color=l.get_color())
    print(f"{p:14s} pronóstico 2022-2025: {f.round(1).tolist()}")

# Participación de India en los canales creados cada año
part = (df[df['Country'] == 'India'].groupby('created_year').size() /
        df[df['Country'] != 'Desconocido'].groupby('created_year').size()).reindex(anios).fillna(0)
print('\nParticipación de India por periodo de creación:')
print(part.groupby(pd.cut(part.index, [2004, 2010, 2015, 2021], labels=['2005-2010', '2011-2015', '2016-2021'])).mean().map('{:.1%}'.format))
ax.axvline(2021.5, color='gray', ls=':'); ax.set(title='Canales top creados por año y país', xlabel='Año de creación', ylabel='N.º canales')
ax.legend(); mostrar('serie_paises')

# %% [markdown]
# **Predicción C2 (serie de tiempo).** EE. UU. se proyecta **a la baja** (de ~10 a ~7 canales/año), mientras **India se mantiene estable (~6/año)**. La participación de India entre los canales creados cada periodo **sube de ~13% (2005-2010) a ~25% (2016-2021)**: si la tendencia sigue, la brecha EE. UU.–India se seguirá cerrando. Brasil y el resto se mantienen marginales.

# %% [markdown]
# ---
# # 4. ANÁLISIS PRESCRIPTIVO — ¿qué se debería hacer?

# %%
# Simulación con el modelo de ingresos: ¿qué pasa si un canal mediano sube sus vistas mensuales?
base = m2['video_views_for_the_last_30_days'].median()
ing_base = 10**reg2.predict(pd.DataFrame({'video_views_for_the_last_30_days': [np.log10(base)]}))[0]
for aumento in [0.10, 0.25, 0.50]:
    v = base * (1 + aumento)
    p = 10**reg2.predict(pd.DataFrame({'video_views_for_the_last_30_days': [np.log10(v)]}))[0]
    print(f"+{aumento:.0%} vistas mensuales → ingreso {p/1e6:.2f} M USD (+{(p-ing_base)/1e6:.2f} M)")

# %% [markdown]
# **Prescripción.** Con base en lo anterior, para un creador o una marca que quiera crecer/invertir en YouTube:
#
# 1. **Priorizar vistas recientes, no solo suscriptores** (N2): el ingreso es proporcional a las vistas de los últimos 30 días; según la simulación, +25% de vistas mensuales en un canal mediano ≈ +0,4 M USD/año. Mantener constancia de publicación y contenido "evergreen" que siga dando vistas.
# 2. **Calidad sobre cantidad** (N1): subir más videos casi no aumenta las vistas; conviene invertir en menos videos pero mejor producidos y pensados para re-visualización (lo que hace Music, ~34 M vistas por video).
# 3. **Elegir nicho según el futuro, no el pasado** (C1): Music domina por canales antiguos, pero casi no entran nuevos al top; **Entertainment y People & Blogs** son las categorías donde siguen apareciendo canales nuevos exitosos.
# 4. **Apuntar a India y a audiencias de alta población urbana** (C2): India gana participación año a año; contenido en inglés/hindi o subtitulado amplía el mercado. Para una marca, patrocinar canales indios puede dar más alcance por el mismo presupuesto.
# 5. **Revisar la monetización**: 7,5% de canales top figuran con ingreso 0 pese a tener vistas → verificar que la monetización esté activa y diversificar ingresos (patrocinios, membresías).
#
# **Limitaciones generales:** datos de un solo corte (2023), ingresos estimados (no reales), sesgo de supervivencia en el año de creación y ~12% de canales sin país.
