# entrenar.py — Sistema de estimación de pedidos de kits de robótica — caso práctico intermedio
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1) Leer datos (utf-8-sig quita el BOM que trae el CSV)
datos = pd.read_csv("datos.csv", encoding="utf-8-sig")

# 2) Variables de entrada X y respuesta y
columnas = ['consultas_web', 'inversion_publicidad_soles', 'campana_escolar']
objetivo = "pedidos_kits"
X = datos[columnas]
y = datos[objetivo]

# 3) 80 % entrenamiento / 20 % prueba (random_state fijo para repetir resultados)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4) Entrenar la regresión lineal múltiple
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 5) Evaluar SOLO con datos de prueba
pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

# 6) Guardar coeficientes en modelo.json (lo usa la app Flask)
info = {
    "columnas": columnas,
    "intercepto": float(modelo.intercept_),
    "coeficientes": [float(v) for v in modelo.coef_],
    "metricas": {"mae": round(float(mae), 2), "r2": round(float(r2), 4)},
}
with open("modelo.json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print("Intercepto:", round(modelo.intercept_, 4))
for n, v in zip(columnas, modelo.coef_):
    print(f"  {n}: {v:.4f}")
print(f"MAE = {mae:.2f} | R2 = {r2:.4f}")
print("modelo.json guardado")
