import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import networkx as nx
from itertools import combinations
from math import radians, sin, cos, sqrt, atan2

st.set_page_config(layout="wide", page_title="СППР: Оптимизация маршрутной сети")
st.title("Система анализа и оптимизации маршрутной сети общественного транспорта")


# Функция расчета расстояния по формуле гаверсинуса (в км)
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


@st.cache_data
def load_data():
    df = pd.read_csv('utils/transport_data.csv')
    df['RouteNumbers'] = df['RouteNumbers'].fillna('')
    return df


df = load_data()

# Боковая панель
st.sidebar.header("Фильтры и параметры модели")
districts = ["Все районы округа"] + sorted(df['District'].dropna().unique().tolist())
selected_district = st.sidebar.selectbox("Район:", districts)

filtered_df = df if selected_district == "Все районы округа" else df[df['District'] == selected_district].copy()

# Подготовка словаря маршрутов (сохраняем словари, а не объекты Series)
route_stops_map = {}
for record in df.to_dict('records'):
    raw_routes = str(record.get('RouteNumbers', ''))
    for r in raw_routes.split(';'):
        r = r.strip()
        if r and r.lower() not in ['none', 'nan', '']:
            route_stops_map.setdefault(r, []).append(record)

valid_routes = {r: stops for r, stops in route_stops_map.items() if len(stops) >= 3}


tab1, tab2, tab3, tab4 = st.tabs([
    "1. Пространственный анализ и ТПУ",
    "2. Трассировка и извилистость маршрутов",
    "3. Дублирование сети (Жаккар)",
    "4. Сводная оптимизация и отчет"
])

# --- ВКЛАДКА 1: ПРОСТРАНСТВЕННЫЙ АНАЛИЗ И «БЕЛЫЕ ПЯТНА» ---
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Активных остановок", len(filtered_df))
    c2.metric("Пиковый поток (чел/ч)", f"{filtered_df['peak_passengers_per_hour'].sum():,}".replace(",", " "))

    # Остановки далеко от метро (>800 м), но с высоким потоком
    remote_high_demand = filtered_df[(filtered_df['dist_to_metro_m'] > 800) & (
                filtered_df['peak_passengers_per_hour'] > filtered_df['peak_passengers_per_hour'].median())]
    c3.metric("Удаленных узлов спроса (>800м)", len(remote_high_demand))
    c4.metric("Среднее расстояние до метро", f"{int(filtered_df['dist_to_metro_m'].mean())} м")

    map_col, chart_col = st.columns([1.35, 1])
    with map_col:
        st.subheader("Тепловая карта спроса и зоны удаленного спроса")
        show_remote = st.checkbox("Выделить синим удаленные от метро узлы с высоким спросом (кандидаты на экспресс)",
                                  value=True)

        m = folium.Map(location=[filtered_df['lat'].mean(), filtered_df['lon'].mean()], zoom_start=12)

        css = folium.Element("<style>.leaflet-attribution-flag { display: none !important; }</style>")
        m.get_root().html.add_child(css)

        HeatMap([[r['lat'], r['lon'], r['peak_passengers_per_hour']] for _, r in filtered_df.iterrows()], radius=16,
                blur=14).add_to(m)

        # Топ-10 перегруженных ТПУ
        for _, row in filtered_df.nlargest(10, 'peak_passengers_per_hour').iterrows():
            folium.CircleMarker(
                [row['lat'], row['lon']], radius=7, color='red', fill=True,
                popup=f"<b>[Перегрузка ТПУ] {row['Name']}</b><br>Поток: {row['peak_passengers_per_hour']} пасс/ч"
            ).add_to(m)

        if show_remote:
            for _, row in remote_high_demand.nlargest(15, 'peak_passengers_per_hour').iterrows():
                folium.CircleMarker(
                    [row['lat'], row['lon']], radius=6, color='blue', fill=True,
                    popup=f"<b>[Удаленный узел] {row['Name']}</b><br>До метро: {row['dist_to_metro_m']} м<br>Поток: {row['peak_passengers_per_hour']} пасс/ч"
                ).add_to(m)

        st_folium(m, width=750, height=480)

    with chart_col:
        st.subheader("Зависимость потока от удаленности метро")
        fig_scatter = px.scatter(
            filtered_df, x='dist_to_metro_m', y='peak_passengers_per_hour',
            color='District', hover_data=['Name', 'nearest_metro'],
            labels={'dist_to_metro_m': 'Расстояние до метро (м)', 'peak_passengers_per_hour': 'Пассажиров в час'}
        )
        fig_scatter.update_layout(height=480, showlegend=False)
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- ВКЛАДКА 2: ТРАССИРОВКА И КОЭФФИЦИЕНТ ИЗВИЛИСТОСТИ ---
with tab2:
    st.subheader("Анализ геометрии маршрута и построение графа связности")
    st.markdown(
        "**Коэффициент непрямолинейности (извилистости)** рассчитывается как отношение суммы перегонов между остановками к расстоянию по прямой между конечными точками. Значение **> 1.6** сигнализирует о необходимости спрямления трассы.")

    sel_route = st.selectbox("Выберите маршрут для визуализации трассы:", sorted(valid_routes.keys()))
    r_stops = pd.DataFrame(valid_routes[sel_route])

    # Берем одно направление (например, 'к центру' или первое доступное), чтобы линия не прыгала туда-обратно
    directions = r_stops['Direction'].dropna().unique()
    if len(directions) > 0:
        sel_dir = st.radio("Направление движения:", directions, horizontal=True)
        r_stops = r_stops[r_stops['Direction'] == sel_dir].copy()

    # Упорядочиваем остановки методом ближайшего соседа (построение цепи маршрута)
    if len(r_stops) >= 2:
        coords = r_stops[['lat', 'lon']].values
        # Находим самую крайнюю точку (минимальная широта) как старт
        start_idx = np.argmin(coords[:, 0])
        ordered_indices = [start_idx]
        unvisited = set(range(len(coords))) - {start_idx}

        total_path_km = 0.0
        while unvisited:
            curr = ordered_indices[-1]
            nearest = min(unvisited, key=lambda idx: haversine_km(coords[curr][0], coords[curr][1], coords[idx][0],
                                                                  coords[idx][1]))
            total_path_km += haversine_km(coords[curr][0], coords[curr][1], coords[nearest][0], coords[nearest][1])
            ordered_indices.append(nearest)
            unvisited.remove(nearest)

        ordered_stops = r_stops.iloc[ordered_indices]
        straight_km = haversine_km(
            ordered_stops.iloc[0]['lat'], ordered_stops.iloc[0]['lon'],
            ordered_stops.iloc[-1]['lat'], ordered_stops.iloc[-1]['lon']
        )
        tortuosity = total_path_km / max(0.1, straight_km)

        m1, m2, m3 = st.columns(3)
        m1.metric("Длина трассы (оценка)", f"{total_path_km:.2f} км")
        m2.metric("Расстояние по прямой", f"{straight_km:.2f} км")
        m3.metric("Коэф. непрямолинейности", f"{tortuosity:.2f}",
                  delta="Требует спрямления" if tortuosity > 1.6 else "Оптимально", delta_color="inverse")

        m_route = folium.Map(location=[ordered_stops['lat'].mean(), ordered_stops['lon'].mean()], zoom_start=13)

        css = folium.Element("<style>.leaflet-attribution-flag { display: none !important; }</style>")
        m_route.get_root().html.add_child(css)

        line_coords = list(zip(ordered_stops['lat'], ordered_stops['lon']))
        folium.PolyLine(line_coords, color="blue", weight=5, opacity=0.8).add_to(m_route)

        for i, (_, s_row) in enumerate(ordered_stops.iterrows()):
            folium.CircleMarker(
                [s_row['lat'], s_row['lon']], radius=5, color="darkblue", fill=True,
                popup=f"{i + 1}. {s_row['Name']} ({s_row['peak_passengers_per_hour']} пасс/ч)"
            ).add_to(m_route)
        st_folium(m_route, width=1100, height=450)

# --- ВКЛАДКА 3: ДУБЛИРОВАНИЕ СЕТИ ---
with tab3:
    st.subheader("Поиск дублирующих маршрутов (Индекс Жаккара)")
    threshold = st.slider("Минимальный порог дублирования (%)", 30, 90, 55)

    route_id_sets = {r: set(pd.DataFrame(stops)['ID']) for r, stops in valid_routes.items()}
    duplicates = []
    for r1, r2 in combinations(route_id_sets.keys(), 2):
        s1, s2 = route_id_sets[r1], route_id_sets[r2]
        sim = (len(s1 & s2) / len(s1 | s2)) * 100
        if sim >= threshold:
            duplicates.append({
                "Маршрут 1": r1, "Маршрут 2": r2,
                "Общих остановок": len(s1 & s2),
                "Совпадение (%)": round(sim, 1),
                "Управленческое решение": "Объединение маршрутов" if sim > 75 else "Тактовое расписание (сдвиг на пол-интервала)"
            })
    dup_df = pd.DataFrame(duplicates).sort_values(by="Совпадение (%)", ascending=False)
    st.dataframe(dup_df, use_container_width=True)

# --- ВКЛАДКА 4: ИНТЕРАКТИВНЫЙ СИМУЛЯТОР ОПТИМИЗАЦИИ ---
with tab4:
    st.subheader("Симулятор оптимизации интервалов движения (What-If анализ)")
    st.markdown(
        "Выберите маршрут и измените параметры выпуска, чтобы оценить наполняемость салонов и среднее время ожидания пассажиров.")

    if not valid_routes:
        st.warning("Не найдено маршрутов для расчета.")
    else:
        col_left, col_right = st.columns([1, 1.2])

        with col_left:
            selected_route = st.selectbox("1. Выберите маршрут:", sorted(valid_routes.keys()), key="sim_route")

            bus_option = st.selectbox("2. Класс подвижного состава:", [
                ("Малый класс (35 пасс.)", 35),
                ("Средний класс (65 пасс.)", 65),
                ("Большой класс (105 пасс.)", 105),
                ("Особо большой / Гармошка (150 пасс.)", 150)
            ], format_func=lambda x: x[0], index=2)
            bus_capacity = bus_option[1]

            current_interval = st.slider("3. Текущий интервал движения (мин):", min_value=2, max_value=40, value=15)

        # Расчет показателей для выбранного маршрута
        r_df = pd.DataFrame(valid_routes[selected_route])
        route_demand = int(r_df['peak_passengers_per_hour'].sum() * 0.18)

        buses_per_hour = 60 / current_interval
        total_capacity = buses_per_hour * bus_capacity
        load_factor = route_demand / total_capacity if total_capacity > 0 else 0
        avg_wait_time = current_interval / 2.0

        # Расчет оптимального интервала (целевая загрузка 75%)
        target_load = 0.75
        needed_buses = route_demand / (bus_capacity * target_load)
        optimal_interval = max(2, min(40, round(60 / needed_buses))) if needed_buses > 0 else 30
        opt_wait_time = optimal_interval / 2.0

        with col_right:
            m1, m2 = st.columns(2)
            m1.metric("Спрос на маршрут (час пик)", f"{route_demand} пасс/ч")
            m2.metric("Рейсов в час", f"{buses_per_hour:.1f} рейс/ч")

            m3, m4 = st.columns(2)
            m3.metric(
                "Коэффициент наполняемости",
                f"{load_factor:.2f}",
                delta="Перегрузка (>0.85)" if load_factor > 0.85 else (
                    "Недогруз (<0.35)" if load_factor < 0.35 else "Комфортная норма"),
                delta_color="inverse"
            )
            m4.metric(
                "Среднее время ожидания",
                f"{avg_wait_time:.1f} мин",
                delta=f"{opt_wait_time - avg_wait_time:+.1f} мин при оптим." if optimal_interval != current_interval else "Оптимально",
                delta_color="inverse"
            )

            # Визуальная шкала загрузки салона
            st.write("**Уровень заполнения салона (норматив — 75%):**")
            st.progress(min(1.0, load_factor))

            if optimal_interval < current_interval:
                st.error(
                    f"**Рекомендация:** Увеличить выпуск! Сократите интервал с **{current_interval}** до **{optimal_interval} мин.**")
            elif optimal_interval > current_interval:
                st.info(
                    f"**Рекомендация:** Возможна оптимизация расходов. Увеличьте интервал с **{current_interval}** до **{optimal_interval} мин.**")
            else:
                st.success("**Рекомендация:** Текущее расписание сбалансировано.")

        # Сводный отчет по всем маршрутам убираем в аккуратный спойлер
        with st.expander("Показать сводный отчет по всем маршрутам округа и скачать CSV"):
            opt_rows = []
            for r, stops in valid_routes.items():
                df_r = pd.DataFrame(stops)
                dem = int(df_r['peak_passengers_per_hour'].sum() * 0.18)
                nb = dem / (105 * 0.75)
                opt_int = max(3, min(30, round(60 / nb))) if nb > 0 else 30
                opt_rows.append({
                    "Маршрут": r,
                    "Остановок": len(df_r),
                    "Спрос (пасс/ч)": dem,
                    "Оптим. интервал (при БК 105 чел.)": opt_int
                })
            opt_df = pd.DataFrame(opt_rows).sort_values(by="Спрос (пасс/ч)", ascending=False)
            st.dataframe(opt_df, use_container_width=True)
            st.download_button(
                "Скачать отчет (CSV)",
                opt_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
                "optimization_report.csv",
                "text/csv"
            )
