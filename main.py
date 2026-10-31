"""
Анализ заболеваемости ВИЧ по регионам России.
Модель: линейная регрессия с One-Hot Encoding регионов.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(filepath=r'C:\Users\1\Desktop\Lessons\Lessons online\Block 7\Exam Block 7\data\analytics_result.csv'):
    """Загружает данные, очищает и применяет One-Hot Encoding."""
    df = pd.read_csv(filepath)
    
    print("Первые 5 строк:")
    print(df.head())
    print("\nКолонки:")
    print(df.columns.tolist())
    
    assert "hiv_firstdiagnosis_100k" in df.columns, "Целевая переменная не найдена!"
    
    # Очистка: убираем РФ и федеральные округа
    df = df[df['Регион'] != 'российская_федерация']
    df = df[~df['Регион'].str.contains('федеральный_округ', case=False, na=False)]
    print(f"\nСтрок после очистки: {len(df)}")
    
    # One-Hot Encoding для регионов
    df_encoded = pd.get_dummies(df, columns=['Регион'], drop_first=True, dtype=int)
    region_cols = [col for col in df_encoded.columns if col.startswith('Регион_')]
    print(f"Колонок регионов: {len(region_cols)}")
    
    # Признаки и цель
    features_base = [
        'hiv_therapy_regdispensaries',
        'hiv_regdispensaries_numdetect',
        'hiv_motherchild_chemoprophylaxis',
        'число_впервые_выявленных',
        'тестировано_россиян',
        'тестировано_иностранцев',
        'num_alldoctors_inmedorgservices',
        'num_infectiousdoctors_inmedorgservices',
        'per_infectious_inalldoctors',
        'year'
    ]
    features = features_base + region_cols
    target = 'hiv_firstdiagnosis_100k'
    
    X = df_encoded[features]
    y = df_encoded[target]
    
    print(f"\nПризнаков всего: {len(features)}")
    print(f"  Базовых: {len(features_base)}")
    print(f"  Регионов: {len(region_cols)}")
    
    # Убираем пропуски
    data = pd.concat([X, y], axis=1).dropna()
    X = data[features]
    y = data[target]
    
    print(f"\nРазмер X: {X.shape}")
    print(f"Размер y: {y.shape}")
    
    return X, y, features


def train_model(X, y):
    """Делит данные, масштабирует и обучает линейную регрессию."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nВсего строк: {len(X)}")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Масштабирование (только признаки, не цель)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    assert model is not None
    
    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test


def evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test):
    """Считает метрики качества модели."""
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    
    print(f"\nR² train: {r2_train:.4f}")
    print(f"R² test:  {r2_test:.4f}")
    
    return y_pred_test, mae, mse, rmse, r2_test


def show_coefficients(model, features):
    """Выводит топ-20 важных признаков по модулю коэффициента."""
    coef_df = pd.DataFrame({
        'Признак': features,
        'Коэффициент': model.coef_
    })
    coef_df['abs_coef'] = coef_df['Коэффициент'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=False).drop(columns='abs_coef')
    
    print("\nТоп-20 важных признаков:")
    print(coef_df.head(20).to_string(index=False))
    return coef_df


def plot_predictions(y_test, y_pred, filename=None):
    """Строит график реальных vs предсказанных значений."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color='coral', edgecolor='black', linewidth=0.5)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2, label='Идеальное предсказание')

    plt.xlabel("Факт")
    plt.ylabel("Прогноз")
    plt.title("Линейная регрессия")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Создаём папку Pictures рядом с main.py
    if filename is None:
        pictures_dir = Path(__file__).parent / 'Pictures'
        pictures_dir.mkdir(exist_ok=True)
        filename = pictures_dir / 'predictions.png'

    plt.savefig(filename)
    plt.show()
    print(f"График сохранён в {filename}")


def main():
    """Главная функция."""
    # 1. Загрузка и подготовка
    data_path = Path(__file__).parent / 'data' / 'analytics_result.csv'
    X, y, features = load_and_prepare_data(data_path)
    
    # 2. Обучение
    model, scaler, X_train_scaled, X_test_scaled, y_train, y_test = train_model(X, y)
    
    # 3. Оценка
    y_pred_test, mae, mse, rmse, r2_test = evaluate_model(
        model, X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # 4. Сравнение предсказаний
    comparison = pd.DataFrame({
        'Реальные': y_test.values[:15],
        'Предсказанные': y_pred_test[:15].round(2),
        'Ошибка': (y_test.values[:15] - y_pred_test[:15]).round(2)
    })
    print("\nПервые 15 предсказаний:")
    print(comparison)
    
    # 5. Метрики
    print(f"\nMAE:  {mae:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2_test:.2f}")
    
    # 6. Коэффициенты
    show_coefficients(model, features)
    
    # 7. График
    plot_predictions(y_test, y_pred_test)
    
    # 8. Итоговый отчёт
    report = {
        "rows": len(X),
        "features": len(X.columns),
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "R2": round(float(r2_test), 3)
    }

    print("ИТОГОВЫЙ ОТЧЁТ")

    print(report)



if __name__ == "__main__":
    main()