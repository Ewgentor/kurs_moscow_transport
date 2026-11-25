import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
from itertools import combinations

st.set_page_config(layout="wide", page_title="Оптимизация маршрутной сети")
st.title("Анализ и оптимизация маршрутной сети общественного транспорта (СЗАО)")


@st.cache_data
def load_data():
    df = pd.read_csv('utils/transport_data.csv')
    df['RouteNumbers'] = df['RouteNumbers'].fillna('')
    return df


df = load_data()

# Боковая панель фильтров
st.sidebar.header("Параметры фильтрации")
districts = ["Все районы"] + sorted(df['District'].dropna().unique().tolist())
selected_district = st.sidebar.selectbox("Район:", districts)

if selected_district != "Все районы":
    filtered_df = df[df['District'] == selected_district].copy()
else:
    filtered_df = df.copy()

tab1, tab2, tab3 = st.tabs([
    "1. Карта пассажиропотока и ТПУ",
    "2. Анализ дублирования маршрутов",
    "3. Симулятор оптимизации интервалов"
])

# --- ВКЛАДКА 1: КАРТА И АНАЛИТИКА ---
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Всего остановок", len(filtered_df))
    col2.metric("Суммарный поток (чел/час пик)", f"{filtered_df['peak_passengers_per_hour'].sum():,}".replace(",", " "))
    col3.metric("Макс. нагрузка на узел", f"{filtered_df['peak_passengers_per_hour'].max()} пасс/ч")

    map_col, chart_col = st.columns([1.3, 1])

    with map_col:
        st.subheader("Тепловая карта спроса и перегруженные узлы")
        m = folium.Map(location=[filtered_df['lat'].mean(), filtered_df['lon'].mean()], zoom_start=12)

        css = folium.Element("<style>.leaflet-attribution-flag { display: none !important; }</style>")
        m.get_root().html.add_child(css)

        # Слой тепловой карты
        heat_data = [[row['lat'], row['lon'], row['peak_passengers_per_hour']] for _, row in filtered_df.iterrows()]
        HeatMap(heat_data, radius=18, blur=15).add_to(m)

        # Маркеры для топ-15 самых загруженных остановок
        top_stops = filtered_df.nlargest(15, 'peak_passengers_per_hour')
        for _, row in top_stops.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=8,
                color='red',
                fill=True,
                popup=f"<b>{row['Name']}</b><br>Поток: {row['peak_passengers_per_hour']} пасс/ч<br>Маршруты: {row['RouteNumbers']}"
            ).add_to(m)

        st_folium(m, width=750, height=500)

    with chart_col:
        st.subheader("Топ-10 загруженных остановок")
        fig = px.bar(
            top_stops.head(10),
            x='peak_passengers_per_hour',
            y='Name',
            orientation='h',
            labels={'peak_passengers_per_hour': 'Пассажиров в час', 'Name': 'Остановка'},
            color='peak_passengers_per_hour',
            color_continuous_scale='Reds'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig, use_container_width=True)

# --- ВКЛАДКА 2: ПОИСК ДУБЛИРОВАНИЯ (КОЭФФИЦИЕНТ ЖАККАРА) ---
with tab2:
    st.subheader("Выявление дублирующих маршрутов")
    st.markdown(
        "Алгоритм вычисляет **коэффициент Жаккара** (отношение общих остановок к общему числу уникальных остановок двух маршрутов).")

    threshold = st.slider("Порог совпадения маршрутов (%)", 30, 90, 50)

    # Собираем словарь: маршрут -> множество ID остановок
    route_stops = {}
    for _, row in df.iterrows():
        for r in str(row['RouteNumbers']).split(';'):
            r = r.strip()
            if r:
                route_stops.setdefault(r, set()).add(row['ID'])

    # Фильтруем маршруты, у которых больше 5 остановок в округе
    valid_routes = {r: s for r, s in route_stops.items() if len(s) >= 5}

    duplicates = []
    for r1, r2 in combinations(valid_routes.keys(), 2):
        s1, s2 = valid_routes[r1], valid_routes[r2]
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        similarity = (intersection / union) * 100
        if similarity >= threshold:
            duplicates.append({
                "Маршрут А": r1,
                "Маршрут Б": r2,
                "Общих остановок": intersection,
                "Совпадение (%)": round(similarity, 1),
                "Рекомендация": "Объединить или развести трассы" if similarity > 70 else "Синхронизировать расписание"
            })

    dup_df = pd.DataFrame(duplicates).sort_values(by="Совпадение (%)", ascending=False)
    st.dataframe(dup_df, use_container_width=True)

# --- ВКЛАДКА 3: СИМУЛЯТОР ОПТИМИЗАЦИИ ---
with tab3:
    st.subheader("Расчет оптимального интервала движения (What-If анализ)")

    all_routes = sorted(list(valid_routes.keys()))
    selected_route = st.selectbox("Выберите маршрут для оптимизации:", all_routes)

    # Считаем суммарный поток на остановках выбранного маршрута
    route_mask = df['RouteNumbers'].apply(lambda x: selected_route in [r.strip() for r in str(x).split(';')])
    route_df = df[route_mask]

    # Принимаем, что на конкретный маршрут садится ~20% от общего потока остановок
    route_demand = int(route_df['peak_passengers_per_hour'].sum() * 0.20)

    col_a, col_b = st.columns(2)
    with col_a:
        bus_option = st.selectbox("Класс вместимости автобуса:", [
            ("Малый класс (35 чел.)", 35),
            ("Средний класс (65 чел.)", 65),
            ("Большой класс (105 чел.)", 105),
            ("Особо большой / Гармошка (150 чел.)", 150)
        ], format_func=lambda x: x[0], index=2)
        bus_capacity = bus_option[1]

        current_interval = st.slider("Текущий интервал движения (минут):", 3, 40, 15)

    # Расчеты
    buses_per_hour = 60 / current_interval
    total_capacity = buses_per_hour * bus_capacity
    load_factor = route_demand / total_capacity if total_capacity > 0 else 0

    # Оптимальный интервал для целевой загрузки 75% (норматив комфорта)
    target_load = 0.75
    needed_buses = route_demand / (bus_capacity * target_load)
    optimal_interval = max(2, min(40, round(60 / needed_buses))) if needed_buses > 0 else 30

    with col_b:
        st.metric("Расчетный спрос на маршрут", f"{route_demand} пасс/час")
        st.metric(
            "Коэффициент наполняемости салона",
            f"{load_factor:.2f}",
            delta="Перегрузка!" if load_factor > 0.85 else ("Недогруз" if load_factor < 0.35 else "Норма"),
            delta_color="inverse"
        )
        st.success(
            f"**Рекомендуемый интервал движения:** каждые **{optimal_interval} мин.** (для комфортной загрузки 75%)")
