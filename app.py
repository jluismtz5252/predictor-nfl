# =====================================================================
# PREDICTOR NFL - APP (Streamlit)
# Ejecutar:  streamlit run app.py
# Datos: nflverse (partidos, spreads y reportes oficiales de lesiones), se descargan solos
# =====================================================================
import pandas as pd
import numpy as np
import streamlit as st
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Predictor NFL", page_icon="🏈", layout="wide")

URL_PARTIDOS = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
URL_LESIONES = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{}.csv"

VARS_SPREAD = ["spread_local"]
VARS_TODO = ["spread_local", "dif_pf", "dif_pc", "dif_gano", "dif_bajas",
             "local_qb_fuera", "visit_qb_fuera", "dif_descanso", "neutral"]
VARS_MARCADOR = ["spread_local", "over_under", "local_pf_ult5", "local_pc_ult5",
                 "visit_pf_ult5", "visit_pc_ult5", "dif_bajas", "local_qb_fuera", "visit_qb_fuera"]
MODELOS = {
    "Logística - solo spread": "log_spread",
    "Logística - spread+forma+lesiones": "log_todo",
    "Random Forest - spread+forma+lesiones": "rf_todo",
}
NOMBRES_VAR = {"spread_local": "Spread del local", "dif_pf": "Dif. puntos a favor (últ. 5)",
               "dif_pc": "Dif. puntos en contra (últ. 5)", "dif_gano": "Dif. % victorias (últ. 5)",
               "dif_bajas": "Dif. bajas por lesión", "local_qb_fuera": "QB titular local fuera",
               "visit_qb_fuera": "QB titular visitante fuera", "dif_descanso": "Dif. días de descanso",
               "neutral": "Cancha neutral"}


# =====================================================================
# 1. DATOS Y VARIABLES (se guardan en caché 6 horas)
# =====================================================================
@st.cache_data(ttl=6 * 3600, show_spinner="Descargando partidos y lesiones de nflverse...")
def cargar_datos():
    df = pd.read_csv(URL_PARTIDOS)
    df = df[df["season"] >= 2009].copy()
    df["gameday"] = pd.to_datetime(df["gameday"])
    df = df.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    df["jugado"] = df["home_score"].notna()

    lesiones = []
    for temporada in range(2009, int(df["season"].max()) + 1):
        try:
            lesiones.append(pd.read_csv(URL_LESIONES.format(temporada)))
        except Exception:
            pass
    lesiones = pd.concat(lesiones)
    lesiones = lesiones[lesiones["report_status"].isin(["Out", "Doubtful"])]

    # Variables del partido
    df["spread_local"] = -df["spread_line"]
    df["over_under"] = df["total_line"]
    df["neutral"] = (df["location"] == "Neutral").astype(int)
    df["dif_descanso"] = df["home_rest"] - df["away_rest"]
    df["gana_local"] = (df["home_score"] > df["away_score"]).astype(int)

    # Tabla larga: una fila por equipo por partido
    loc = df[["game_id", "season", "week", "gameday", "home_team", "home_score", "away_score", "home_qb_id"]].copy()
    loc.columns = ["game_id", "season", "week", "fecha", "equipo", "pf", "pc", "qb_id"]
    loc["rol"] = "local"
    vis = df[["game_id", "season", "week", "gameday", "away_team", "away_score", "home_score", "away_qb_id"]].copy()
    vis.columns = ["game_id", "season", "week", "fecha", "equipo", "pf", "pc", "qb_id"]
    vis["rol"] = "visit"
    largo = pd.concat([loc, vis]).sort_values(["equipo", "fecha"]).reset_index(drop=True)
    largo["gano"] = np.where(largo["pf"].notna(), (largo["pf"] > largo["pc"]).astype(float), np.nan)

    # Forma: últimos 5 partidos ANTES de cada juego
    for col in ["pf", "pc", "gano"]:
        largo[col + "_ult5"] = largo.groupby("equipo")[col].transform(
            lambda s: s.shift(1).rolling(5, min_periods=3).mean())

    # Lesiones: número de bajas y si el QB titular está Fuera/Dudoso
    bajas = lesiones.groupby(["season", "week", "team"]).size().rename("bajas").reset_index()
    largo = largo.merge(bajas, left_on=["season", "week", "equipo"], right_on=["season", "week", "team"], how="left")
    largo["bajas"] = largo["bajas"].fillna(0)
    largo = largo.sort_values(["equipo", "fecha"]).reset_index(drop=True)
    largo["qb_titular"] = largo.groupby("equipo")["qb_id"].shift(1)
    qbs_fuera = set(zip(lesiones["season"], lesiones["week"], lesiones["team"], lesiones["gsis_id"]))
    largo["qb_fuera"] = [int((s, w, e, q) in qbs_fuera) for s, w, e, q in
                         zip(largo["season"], largo["week"], largo["equipo"], largo["qb_titular"])]

    cols = ["pf_ult5", "pc_ult5", "gano_ult5", "bajas", "qb_fuera"]
    f_loc = largo[largo["rol"] == "local"].set_index("game_id")[cols].add_prefix("local_")
    f_vis = largo[largo["rol"] == "visit"].set_index("game_id")[cols].add_prefix("visit_")
    df = df.join(f_loc, on="game_id").join(f_vis, on="game_id")

    df["dif_pf"] = df["local_pf_ult5"] - df["visit_pf_ult5"]
    df["dif_pc"] = df["local_pc_ult5"] - df["visit_pc_ult5"]
    df["dif_gano"] = df["local_gano_ult5"] - df["visit_gano_ult5"]
    df["dif_bajas"] = df["local_bajas"] - df["visit_bajas"]

    semanas_con_reporte = set(zip(lesiones["season"], lesiones["week"]))
    return df, semanas_con_reporte


def partidos_jugados(df):
    datos = df[df["season"] >= 2010].dropna(subset=VARS_TODO + VARS_MARCADOR)
    return datos[datos["jugado"] & (datos["home_score"] != datos["away_score"])]


# =====================================================================
# 2. MODELOS
# =====================================================================
def crear_modelo(nombre):
    if nombre.startswith("Random Forest"):
        return RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=20, random_state=0)
    return LogisticRegression(max_iter=1000)


def entrenar_y_predecir(train, test, nombres):
    salida = {}
    for nombre in nombres:
        variables = VARS_SPREAD if "solo spread" in nombre else VARS_TODO
        scaler = StandardScaler()
        modelo = crear_modelo(nombre)
        modelo.fit(scaler.fit_transform(train[variables]), train["gana_local"])
        Xte = scaler.transform(test[variables])
        salida[nombre] = (modelo.predict(Xte), modelo.predict_proba(Xte)[:, 1])
    sc_r = StandardScaler()
    Xr = sc_r.fit_transform(train[VARS_MARCADOR])
    reg_l = LinearRegression().fit(Xr, train["home_score"])
    reg_v = LinearRegression().fit(Xr, train["away_score"])
    Xr_te = sc_r.transform(test[VARS_MARCADOR])
    salida["marcador"] = (reg_l.predict(Xr_te), reg_v.predict(Xr_te))
    return salida


@st.cache_data(ttl=6 * 3600, show_spinner="Probando los modelos con semanas anteriores...")
def backtest(df, temporadas):
    jugados = partidos_jugados(df)
    filas = []
    for temporada in temporadas:
        for semana in sorted(jugados[jugados["season"] == temporada]["week"].unique()):
            train = jugados[(jugados["season"] < temporada) |
                            ((jugados["season"] == temporada) & (jugados["week"] < semana))]
            test = jugados[(jugados["season"] == temporada) & (jugados["week"] == semana)]
            pred = entrenar_y_predecir(train, test, list(MODELOS))
            fila = {"temporada": temporada, "semana": int(semana), "partidos": len(test),
                    "favorito": int(((test["spread_local"] <= 0).astype(int) == test["gana_local"]).sum())}
            for nombre, campo in MODELOS.items():
                fila[campo] = int((pred[nombre][0] == test["gana_local"]).sum())
            fila["mae_marcador"] = round((mean_absolute_error(test["home_score"], pred["marcador"][0]) +
                                          mean_absolute_error(test["away_score"], pred["marcador"][1])) / 2, 2)
            filas.append(fila)
    return pd.DataFrame(filas)


def resumen_backtest(bt):
    filas = []
    for nombre, campo in [("Elegir al favorito", "favorito")] + list(MODELOS.items()):
        fila = {"Modelo": nombre, "Aciertos": int(bt[campo].sum()), "Partidos": int(bt["partidos"].sum()),
                "Accuracy %": round(bt[campo].sum() / bt["partidos"].sum() * 100, 1)}
        for t in sorted(bt["temporada"].unique()):
            sub = bt[bt["temporada"] == t]
            fila["Acc " + str(t) + " %"] = round(sub[campo].sum() / sub["partidos"].sum() * 100, 1)
        filas.append(fila)
    return pd.DataFrame(filas)


RANGOS = [-0.1, 2.5, 3.5, 6.5, 9.5, 13.5, 40]
ETIQUETAS = ["0 a 2.5", "3 a 3.5", "4 a 6.5", "7 a 9.5", "10 a 13.5", "14 o más"]


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def sorpresas_por_spread(df):
    """% de veces que pierde el favorito según el tamaño del spread (desde 2010)."""
    j = partidos_jugados(df).copy()
    j["falla"] = ((j["spread_local"] <= 0).astype(int) != j["gana_local"]).astype(int)
    j["rango"] = pd.cut(j["spread_local"].abs(), RANGOS, labels=ETIQUETAS)
    j["inicio"] = np.where(j["week"] <= 4, "Semanas 1-4", "Semana 5 en adelante")
    t = j.groupby("rango", observed=False)["falla"].agg(["count", "mean"]).reset_index()
    t.columns = ["Spread del favorito", "Partidos", "% que pierde el favorito"]
    t["% que pierde el favorito"] = (t["% que pierde el favorito"] * 100).round(1)
    t2 = (j.pivot_table(index="rango", columns="inicio", values="falla", aggfunc="mean", observed=False) * 100).round(1)
    return t, t2.reset_index().rename(columns={"rango": "Spread del favorito"})


def nivel_confianza(abs_spread):
    if abs_spread >= 7:
        return "🟢 Confianza alta"
    if abs_spread >= 4:
        return "🟡 Confianza media"
    return "🔴 Confianza baja (casi volado)"


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def factores_logit(df):
    jugados = partidos_jugados(df)
    X = jugados[VARS_TODO]
    X = (X - X.mean()) / X.std()
    r = sm.Logit(jugados["gana_local"], sm.add_constant(X)).fit(disp=0)
    tabla = pd.DataFrame({"Variable": [NOMBRES_VAR[v] for v in VARS_TODO],
                          "Coeficiente": [round(r.params[v], 4) for v in VARS_TODO],
                          "p-value": [round(r.pvalues[v], 4) for v in VARS_TODO]})
    tabla["¿Significativa?"] = np.where(tabla["p-value"] < 0.05, "Sí", "No")
    qb = jugados.groupby("local_qb_fuera")["gana_local"].agg(["count", "mean"]).reset_index()
    return tabla, qb, len(jugados)


# =====================================================================
# 3. INTERFAZ
# =====================================================================
df, semanas_con_reporte = cargar_datos()
temp_actual = int(df["season"].max())
temporadas_prueba = [temp_actual - 1, temp_actual]
bt = backtest(df, temporadas_prueba)
resumen = resumen_backtest(bt)
mejor_modelo = resumen[resumen["Modelo"] != "Elegir al favorito"].sort_values("Accuracy %", ascending=False).iloc[0]["Modelo"]

tabla_sorpresas, tabla_sorpresas_inicio = sorpresas_por_spread(df)

# --- Barra lateral ---
st.sidebar.title("🏈 Predictor NFL")
temporada = st.sidebar.selectbox("Temporada", sorted(df["season"].unique(), reverse=True)[:6])
semanas = sorted(df[df["season"] == temporada]["week"].unique())
pendientes = df[(df["season"] == temporada) & (~df["jugado"])]["week"]
semana_default = int(pendientes.min()) if len(pendientes) else int(max(semanas))
semana = st.sidebar.selectbox("Semana", semanas, index=semanas.index(semana_default))
modelo_sel = st.sidebar.selectbox("Modelo", list(MODELOS), index=list(MODELOS).index(mejor_modelo),
                                  help="Por defecto, el que más acertó en la prueba con semanas anteriores.")
st.sidebar.caption("Datos: nflverse. Se actualizan solos cada 6 horas.")
if st.sidebar.button("🔄 Actualizar datos ahora"):
    st.cache_data.clear()
    st.rerun()

tab1, tab2, tab3 = st.tabs(["📅 Pronósticos", "🧪 Prueba con semanas anteriores", "🩼 Lesiones y dinero"])

# --- Pestaña 1: pronósticos de la semana ---
with tab1:
    st.header(f"Temporada {temporada} · Semana {semana}")
    jugados = partidos_jugados(df)
    # Se entrena solo con partidos ANTERIORES a la semana elegida
    train = jugados[(jugados["season"] < temporada) | ((jugados["season"] == temporada) & (jugados["week"] < semana))]
    semana_df = df[(df["season"] == temporada) & (df["week"] == semana)].copy()
    sin_linea = semana_df[semana_df["spread_local"].isna()]
    semana_df = semana_df.dropna(subset=["spread_local", "over_under"])
    relleno = ["local_bajas", "visit_bajas", "dif_bajas"]
    semana_df[relleno] = semana_df[relleno].fillna(0)
    semana_df = semana_df.dropna(subset=VARS_TODO + VARS_MARCADOR)

    if (temporada, semana) not in semanas_con_reporte:
        st.warning("Todavía no sale el reporte de lesiones de esta semana (sale de miércoles a viernes). "
                   "Las bajas aparecen en 0 hasta entonces.")
    if len(sin_linea):
        st.info("Sin línea de apuestas todavía: " + ", ".join(sin_linea["away_team"] + " @ " + sin_linea["home_team"]))

    if len(semana_df) == 0:
        st.write("No hay partidos con línea para esta semana.")
    else:
        pred = entrenar_y_predecir(train, semana_df, [modelo_sel])
        gana, prob_local = pred[modelo_sel]
        pts_l, pts_v = pred["marcador"]
        semana_df["pred_gana_local"] = gana
        semana_df["prob_local"] = prob_local * 100
        semana_df["pts_l"] = np.round(pts_l).astype(int)
        semana_df["pts_v"] = np.round(pts_v).astype(int)

        ya_jugada = semana_df["jugado"].all()
        if ya_jugada:
            aciertos = int((semana_df["pred_gana_local"] == semana_df["gana_local"]).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Aciertos del modelo", f"{aciertos} de {len(semana_df)}",
                      f"{aciertos / len(semana_df) * 100:.1f}%")
            fav = int(((semana_df["spread_local"] <= 0).astype(int) == semana_df["gana_local"]).sum())
            c2.metric("Elegir al favorito", f"{fav} de {len(semana_df)}")
            err = (np.abs(semana_df["pts_l"] - semana_df["home_score"]).mean() +
                   np.abs(semana_df["pts_v"] - semana_df["away_score"]).mean()) / 2
            c3.metric("Error del marcador", f"±{err:.1f} pts")
            st.caption("Semana ya jugada: el modelo se entrenó solo con partidos anteriores, como si fuera ese momento.")

        columnas = st.columns(3)
        for i, (_, p) in enumerate(semana_df.iterrows()):
            gana_l = p["pred_gana_local"] == 1
            ganador = p["home_team"] if gana_l else p["away_team"]
            prob = p["prob_local"] if gana_l else 100 - p["prob_local"]
            cerrado = (40 <= p["prob_local"] <= 60) or ((p["pts_l"] > p["pts_v"]) != gana_l) or (p["pts_l"] == p["pts_v"])
            with columnas[i % 3].container(border=True):
                st.subheader(f"{p['away_team']} @ {p['home_team']}")
                st.caption(f"{p['gameday'].date()} · Spread local {p['spread_local']:+g} · O/U {p['over_under']:g}")
                st.markdown(f"**Ganador: {ganador}** ({prob:.1f}%)")
                st.progress(min(max(prob / 100, 0.0), 1.0))
                rango = pd.cut([abs(p["spread_local"])], RANGOS, labels=ETIQUETAS)[0]
                pct_sorpresa = tabla_sorpresas.loc[tabla_sorpresas["Spread del favorito"] == rango,
                                                   "% que pierde el favorito"].iloc[0]
                st.caption(f"{nivel_confianza(abs(p['spread_local']))} · Con este spread, el favorito "
                           f"ha perdido el {pct_sorpresa}% de las veces desde 2010")
                st.write(f"Marcador estimado: {p['away_team']} **{p['pts_v']}** – **{p['pts_l']}** {p['home_team']}")
                st.caption(f"Bajas (Fuera/Dudoso): {p['home_team']} {int(p['local_bajas'])} · "
                           f"{p['away_team']} {int(p['visit_bajas'])}")
                if p["local_qb_fuera"] == 1:
                    st.error(f"🚑 QB titular de {p['home_team']} fuera/dudoso")
                if p["visit_qb_fuera"] == 1:
                    st.error(f"🚑 QB titular de {p['away_team']} fuera/dudoso")
                if cerrado:
                    st.warning("Partido cerrado: revisa lesiones de último momento y % de dinero")
                dinero = st.number_input("💰 % del dinero al local (opcional)", 0, 100, value=None,
                                         key=f"dinero_{p['game_id']}", placeholder="De Action Network")
                if dinero is not None and (dinero >= 70 or dinero <= 30):
                    lado = p["home_team"] if dinero >= 70 else p["away_team"]
                    st.info(f"El dinero está muy cargado a {lado}: tómalo con cuidado.")
                if p["jugado"]:
                    real = p["home_team"] if p["gana_local"] == 1 else p["away_team"]
                    icono = "✅" if real == ganador else "❌"
                    st.write(f"{icono} Resultado real: {p['away_team']} {int(p['away_score'])} – "
                             f"{int(p['home_score'])} {p['home_team']}")

# --- Pestaña 2: prueba con semanas anteriores ---
with tab2:
    st.header("¿Qué tan bueno es el modelo?")
    st.caption("Cada semana se predijo usando solo los partidos jugados antes de ella.")
    fila_mejor = resumen[resumen["Modelo"] == mejor_modelo].iloc[0]
    fila_fav = resumen[resumen["Modelo"] == "Elegir al favorito"].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Mejor modelo: {mejor_modelo}", f"{fila_mejor['Accuracy %']}%",
              f"{fila_mejor['Aciertos']} de {fila_mejor['Partidos']} partidos", delta_color="off")
    c2.metric("Elegir siempre al favorito", f"{fila_fav['Accuracy %']}%")
    c3.metric("Error promedio del marcador", f"±{bt['mae_marcador'].mean():.1f} pts")

    st.subheader("Aciertos por semana")
    graf = bt.copy()
    graf["Semana"] = graf["temporada"].astype(str) + " S" + graf["semana"].astype(str).str.zfill(2)
    graf["Accuracy %"] = graf[MODELOS[mejor_modelo]] / graf["partidos"] * 100
    st.bar_chart(graf, x="Semana", y="Accuracy %")

    st.subheader("Comparación de modelos")
    st.dataframe(resumen, hide_index=True, width="stretch")
    with st.expander("Ver aciertos semana por semana"):
        st.dataframe(bt, hide_index=True, width="stretch")

# --- Pestaña 3: lesiones y dinero ---
with tab3:
    tabla, qb, n = factores_logit(df)
    st.header("¿Las lesiones y el dinero cambian el pronóstico?")
    pct_juega = qb.loc[qb["local_qb_fuera"] == 0, "mean"].iloc[0] * 100
    pct_fuera = qb.loc[qb["local_qb_fuera"] == 1, "mean"].iloc[0] * 100
    st.write(f"Cuando el QB titular del local está Fuera o Dudoso, el local gana el **{pct_fuera:.1f}%** "
             f"de las veces, contra **{pct_juega:.1f}%** cuando sí juega. Pero en la regresión logística, "
             f"junto al spread, las lesiones dejan de ser significativas: la línea de apuestas ya las descontó.")
    st.dataframe(tabla, hide_index=True, width="stretch")
    st.caption(f"Logit con {n:,} partidos desde 2010, variables estandarizadas.")
    st.subheader("¿Cuándo falla el modelo?")
    st.write("Lo que más explica las sorpresas es el tamaño del spread: entre más parejo el partido, más falla. "
             "Al inicio de la temporada (semanas 1-4) fallan más los partidos parejos, porque todavía hay poca información.")
    st.dataframe(tabla_sorpresas, hide_index=True, width="stretch")
    st.dataframe(tabla_sorpresas_inicio, hide_index=True, width="stretch")
    st.caption("Se revisaron también partido divisional, clima (viento y frío), techo, horario (jueves, lunes, noche), "
               "días de descanso, QB fuera y forma reciente: una vez que se toma en cuenta el spread, "
               "ninguno cambia de forma consistente quién gana.")
    st.subheader("Dinero apostado (public betting)")
    st.write("No hay historial gratuito para entrenar el modelo con él. Para la semana actual se puede revisar en "
             "[actionnetwork.com/nfl/public-money](https://www.actionnetwork.com/nfl/public-money) "
             "y escribirlo en cada partido de la pestaña de pronósticos.")
    st.caption("Herramienta de análisis: acertar cerca de dos de cada tres partidos no basta para ganarle a las casas de apuestas.")
