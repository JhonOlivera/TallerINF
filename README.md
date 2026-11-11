# Análisis de datos — Global YouTube Statistics

Análisis en Python de los 995 canales más grandes de YouTube, en cuatro etapas: **descriptiva, diagnóstica, predictiva y prescriptiva**.

## Dataset

[Global YouTube Statistics 2023](https://www.kaggle.com/datasets/nelgiriyewithana/global-youtube-statistics-2023) (Kaggle). Tiene 995 canales y 28 variables: suscriptores, vistas, categoría, país, ingresos estimados, año de creación y datos del país.

## Estructura

```
├── analisis_youtube.py              # Script con todo el análisis
├── Global YouTube Statistics.csv    # Dataset
├── requirements.txt                 # Librerías necesarias
└── graficas/                        # Gráficas generadas por el script
```

## Cómo ejecutarlo

1. Clonar el repositorio:

```bash
git clone https://github.com/JhonOlivera/TallerINF.git
cd TallerINF
```

2. Instalar las librerías:

```bash
pip install -r requirements.txt
```

3. Ejecutar el script:

```bash
python analisis_youtube.py
```

Los resultados salen en la terminal y las gráficas se guardan en `graficas/`.

## Metodología

| Etapa | Qué se hizo | Técnica |
| --- | --- | --- |
| Descriptiva | 2 hallazgos con variables numéricas y 2 con variables categóricas (EDA) | Estadística descriptiva, histogramas, boxplots, barras |
| Diagnóstica | Explicación de cada hallazgo con otras variables | Correlación de Spearman, agrupaciones |
| Predictiva | Predicción de cada hallazgo | Regresión lineal (log-log) y series de tiempo (Holt con tendencia amortiguada) |
| Prescriptiva | Recomendaciones basadas en los resultados | Simulación de escenarios con el modelo de ingresos |

## Hallazgos principales

**Descriptivo**
- Los suscriptores y las vistas están muy concentrados en pocos canales y van de la mano (Spearman = 0,59).
- Los ingresos son muy desiguales: el 10% de canales que más gana se lleva el 54% de los ingresos.
- Entertainment (24%) y Music (20%) son las categorías con más canales.
- EE. UU. (36%) e India (19%) tienen el 55% de los canales con país conocido.

**Diagnóstico**
- Subir más videos casi no aumenta las vistas. Las vistas dependen de los suscriptores, la antigüedad y el tipo de contenido.
- El ingreso depende de las vistas de los últimos 30 días (Spearman = 0,94), no de los suscriptores.
- Music sube pocos videos, pero cada uno tiene muchísimas vistas (unos 34 M por video).
- El número de canales de un país se relaciona con su población urbana (Spearman = 0,63).

**Predictivo**
- Un 10% más de suscriptores da cerca de un 12% más de vistas.
- El ingreso crece en la misma proporción que las vistas mensuales.
- Se pronostica que casi no entrarán canales nuevos de Music al top. Entertainment y People & Blogs se mantienen estables.
- India sube su participación entre los canales nuevos: de 12,6% (2005–2010) a 25,4% (2016–2021).

**Prescriptivo**
1. Priorizar las vistas recientes sobre los suscriptores.
2. Calidad sobre cantidad de videos.
3. Elegir nichos que siguen creciendo (Entertainment, People & Blogs).
4. Apuntar al mercado de India y a países con mucha población urbana.
5. Revisar la monetización: el 7,5% de los canales top aparece con ingreso 0.

## Limitaciones

- Los datos son de un solo corte (2023).
- Los ingresos son estimados por la fuente, no reales.
- El año de creación tiene sesgo de supervivencia: los canales recientes han tenido poco tiempo para llegar al top.
- Cerca del 12% de los canales no tiene país registrado.

## Tecnologías

Python · pandas · NumPy · Matplotlib · seaborn · scikit-learn · statsmodels

## Autor

Jhon Edwin Olivera Duarte — Ingeniería de Sistemas, Universidad de Ibagué
