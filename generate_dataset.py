import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)


def generate_phones_data(n=8000, catalog_path=None):
    brands = {
        'Apple': (45000, 150000, 0.78),
        'Samsung': (15000, 150000, 0.68),
        'Xiaomi': (15000, 80000, 0.60),
        'Redmi': (8000, 35000, 0.55),
        'POCO': (8000, 35000, 0.56),
        'OnePlus': (20000, 75000, 0.66),
        'OPPO': (15000, 95000, 0.60),
        'vivo': (12000, 95000, 0.60),
        'iQOO': (15000, 60000, 0.63),
        'realme': (10000, 60000, 0.57),
        'Motorola': (10000, 95000, 0.58),
        'Google': (40000, 125000, 0.68),
        'Nothing': (20000, 60000, 0.63),
        'CMF': (14000, 22000, 0.56),
        'HONOR': (20000, 90000, 0.59),
        'Huawei': (45000, 180000, 0.66),
        'Sony': (45000, 140000, 0.65),
        'ASUS': (60000, 120000, 0.64),
        'TECNO': (10000, 85000, 0.54),
        'Infinix': (12000, 30000, 0.54),
        'Lava': (8000, 22000, 0.53),
        'Nokia': (8000, 50000, 0.52),
        'Lenovo': (12000, 50000, 0.55),
        'TCL': (14000, 30000, 0.53),
        'ZTE': (35000, 70000, 0.58),
    }

    brand_names = list(brands.keys())
    rows = []
    for _ in range(n):
        brand = np.random.choice(brand_names)
        min_p, max_p, retention_base = brands[brand]

        ram = np.random.choice([4, 6, 8, 12, 16], p=[0.15, 0.25, 0.35, 0.20, 0.05])
        storage = np.random.choice([64, 128, 256, 512], p=[0.15, 0.45, 0.30, 0.10])
        has_5g = np.random.choice([1, 0], p=[0.60, 0.40])

        base_price = np.random.uniform(min_p, max_p)
        original_price = int(round((base_price + (ram * 800) + (storage * 35)) / 100) * 100)

        months_used = np.random.randint(1, 48)
        screen_condition = np.random.choice([5, 4, 3, 2, 1], p=[0.25, 0.35, 0.20, 0.12, 0.08])
        body_condition = np.random.choice([5, 4, 3, 2, 1], p=[0.25, 0.35, 0.25, 0.10, 0.05])

        battery_health = max(65, int(100 - (months_used * np.random.uniform(0.4, 0.75))))
        under_warranty = np.random.choice([1, 0], p=[0.40, 0.60])
        has_box_bill = np.random.choice([1, 0], p=[0.50, 0.50])

        age_decay = np.exp(-0.028 * months_used)
        cond_factor = (screen_condition * 0.12) + (body_condition * 0.05) + 0.15
        battery_factor = 1.0 if battery_health >= 85 else (0.88 if battery_health >= 78 else 0.74)
        five_g_factor = 1.10 if has_5g else 0.90
        warranty_bonus = 1.12 if under_warranty else 0.96
        box_bonus = 1.10 if has_box_bill else 0.90

        resale_price = original_price * retention_base * age_decay * cond_factor * battery_factor * five_g_factor * warranty_bonus * box_bonus
        noise = np.random.normal(1.0, 0.015)
        resale_price = max(1500, int(round((resale_price * noise) / 50) * 50))

        rows.append({
            'brand': brand,
            'ram_gb': ram,
            'storage_gb': storage,
            'has_5g': has_5g,
            'original_price_inr': original_price,
            'months_used': months_used,
            'screen_condition': screen_condition,
            'body_condition': body_condition,
            'battery_health_pct': battery_health,
            'under_warranty': under_warranty,
            'has_box_bill': has_box_bill,
            'resale_price_inr': resale_price,
        })
    return pd.DataFrame(rows)


def generate_accessories_data(n=1500):
    accessory_types = {
        'TWS_Earbuds': (999, 18999, 0.55),
        'Smartwatch': (1299, 29999, 0.52),
        'Power_Bank': (899, 4999, 0.60),
        'Fast_Charger': (599, 3999, 0.64),
        'Neckband': (799, 3499, 0.50),
    }
    brands = ['Apple', 'Samsung', 'OnePlus', 'boAt', 'Noise', 'Anker', 'Xiaomi', 'Realme']

    rows = []
    for _ in range(n):
        acc_type = np.random.choice(list(accessory_types.keys()))
        brand = np.random.choice(brands)
        min_p, max_p, retention_base = accessory_types[acc_type]

        if brand in ['Apple', 'Samsung'] and acc_type in ['TWS_Earbuds', 'Smartwatch']:
            original_price = int(round(np.random.uniform(7000, max_p) / 50) * 50)
            retention_base += 0.12
        else:
            original_price = int(round(np.random.uniform(min_p, max_p * 0.5) / 50) * 50)

        months_used = np.random.randint(1, 30)
        physical_condition = np.random.choice([5, 4, 3, 2, 1], p=[0.30, 0.35, 0.20, 0.10, 0.05])
        battery_backup_score = 5 if acc_type == 'Fast_Charger' else np.random.choice([5, 4, 3, 2], p=[0.30, 0.40, 0.20, 0.10])
        under_warranty = 1 if months_used < 12 and np.random.rand() > 0.2 else 0
        has_box_accessories = np.random.choice([1, 0], p=[0.65, 0.35])

        age_decay = np.exp(-0.045 * months_used)
        cond_factor = (physical_condition * 0.13) + (battery_backup_score * 0.05) + 0.10
        warranty_bonus = 1.12 if under_warranty else 1.0
        box_bonus = 1.05 if has_box_accessories else 0.92

        resale_price = original_price * retention_base * age_decay * cond_factor * warranty_bonus * box_bonus
        noise = np.random.normal(1.0, 0.035)
        resale_price = max(150, int(round((resale_price * noise) / 10) * 10))

        rows.append({
            'accessory_type': acc_type,
            'brand': brand,
            'original_price_inr': original_price,
            'months_used': months_used,
            'physical_condition': physical_condition,
            'battery_backup_score': battery_backup_score,
            'under_warranty': under_warranty,
            'has_box_accessories': has_box_accessories,
            'resale_price_inr': resale_price,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic used mobile and accessory resale datasets.')
    parser.add_argument('--phones', type=int, default=8000, help='Number of phone rows to generate.')
    parser.add_argument('--accessories', type=int, default=1500, help='Number of accessory rows to generate.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path.cwd(),
        help='Directory where CSV files should be written. Defaults to the current working directory.',
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    phones_df = generate_phones_data(args.phones)
    accessories_df = generate_accessories_data(args.accessories)

    phones_path = output_dir / 'used_phones.csv'
    accessories_path = output_dir / 'used_accessories.csv'
    phones_df.to_csv(phones_path, index=False)
    accessories_df.to_csv(accessories_path, index=False)

    print(f'Created {phones_path.name} ({len(phones_df)} rows) and {accessories_path.name} ({len(accessories_df)} rows) successfully!')


if __name__ == '__main__':
    main()
