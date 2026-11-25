import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


def load_mos_json(filepath):
    for enc in ['utf-8', 'cp1251', 'utf-8-sig']:
        try:
            return pd.read_json(filepath, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Не удалось определить кодировку файла {filepath}")


# 1. Загрузка пассажиропотока метро (набор № 62743)
metro_flow = load_mos_json('../datasets/data-62743-23-10-2026.json')
latest_year = metro_flow['Year'].max()
metro_flow = metro_flow[metro_flow['Year'] == latest_year]
latest_quarter = metro_flow['Quarter'].iloc[-1]
metro_flow = metro_flow[metro_flow['Quarter'] == latest_quarter].drop_duplicates('NameOfStation')

# Среднесуточный поток станции (вход + выход за квартал / 90 дней)
metro_flow['daily_metro_flow'] = (metro_flow['IncomingPassengers'] + metro_flow['OutgoingPassengers']) / 90

# 2. Загрузка координат вестибюлей метро (набор № 624)
vestibules = load_mos_json('../datasets/data-624-23-10-2026.json')
vestibules = vestibules[vestibules['ObjectStatus'] == 'действует'].copy()
vestibules['lon'] = pd.to_numeric(vestibules['Longitude_WGS84'], errors='coerce')
vestibules['lat'] = pd.to_numeric(vestibules['Latitude_WGS84'], errors='coerce')

# Усредняем координаты выходов для каждой станции
metro_coords = vestibules.groupby('NameOfStation', as_index=False)[['lat', 'lon']].mean().dropna()

# Объединяем поток и координаты метро
metro = pd.merge(metro_flow[['NameOfStation', 'Line', 'daily_metro_flow']], metro_coords, on='NameOfStation')
print(f"Станций метро с координатами и потоком: {len(metro)}")

# 3. Загрузка остановок наземного транспорта (набор № 752)
stops = load_mos_json('../datasets/data-752-11-11-2026.json')
stops = stops[stops['EntryState'] == 'активна'].copy()
stops['lon'] = pd.to_numeric(stops['Longitude_WGS84'], errors='coerce')
stops['lat'] = pd.to_numeric(stops['Latitude_WGS84'], errors='coerce')
stops = stops.dropna(subset=['lat', 'lon'])


stops_filtered = stops[stops['AdmArea'].notna()].copy()

# 4. Поиск ближайшей станции метро для каждой остановки
tree = cKDTree(np.radians(metro[['lat', 'lon']].values))
distances, indices = tree.query(np.radians(stops_filtered[['lat', 'lon']].values), k=1)

stops_filtered['dist_to_metro_m'] = np.round(distances * 6371000, 1)
stops_filtered['nearest_metro'] = metro.iloc[indices]['NameOfStation'].values
stops_filtered['nearest_metro_daily'] = metro.iloc[indices]['daily_metro_flow'].values

# 5. Расчет часового пассажиропотока остановки в утренний час пик (чел/час)
np.random.seed(42)


def simulate_peak_flow(row):
    routes = [r.strip() for r in str(row['RouteNumbers']).split(';') if r.strip() and r.strip() != 'None']
    num_routes = max(1, len(routes))

    # Базовый пассажиропоток жилого района
    base_flow = num_routes * np.random.randint(20, 45)

    # Если остановка ближе 400 м от метро (зона ТПУ), добавляем долю от пикового потока метро
    if row['dist_to_metro_m'] <= 400:
        metro_peak_hour = row['nearest_metro_daily'] * 0.12
        transfer_share = (metro_peak_hour * 0.25) / num_routes
        return int(base_flow + transfer_share)
    return int(base_flow)


stops_filtered['peak_passengers_per_hour'] = stops_filtered.apply(simulate_peak_flow, axis=1)

# Используем чистое название StationName, а если оно пустое — берем Name
stops_filtered['CleanName'] = stops_filtered['StationName'].fillna(stops_filtered['Name'])

# Сохраняем итоговый датасет для веб-приложения
cols = ['ID', 'CleanName', 'Direction', 'AdmArea', 'District', 'RouteNumbers', 'lat', 'lon', 'nearest_metro', 'dist_to_metro_m', 'peak_passengers_per_hour']
stops_filtered[cols].rename(columns={'CleanName': 'Name'}).to_csv('transport_data.csv', index=False, encoding='utf-8-sig')
print(f"Готово! Сохранено {len(stops_filtered)} остановок по всем округам Москвы.")
