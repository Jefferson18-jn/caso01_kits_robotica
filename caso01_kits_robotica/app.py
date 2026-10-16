# app.py — Aplicación Flask: Estimador de pedidos semanales de kits de robótica
# Solo Python + Flask + HTML + CSS (sin JavaScript).
import json
from flask import Flask, render_template, request

app = Flask(__name__)

# 1) Cargamos los coeficientes guardados por entrenar.py
with open("modelo.json", encoding="utf-8") as f:
    MODELO = json.load(f)

# 2) Campos del formulario (nombre, etiqueta, unidad y rango permitido)
CAMPOS = [
    {'n': 'consultas_web', 'et': 'Consultas web de la semana', 'u': 'consultas', 'mn': 5, 'mx': 70, 'step': 1, 'tipo': 'num'},
    {'n': 'inversion_publicidad_soles', 'et': 'Inversión en publicidad', 'u': 'S/ (soles)', 'mn': 0, 'mx': 1100, 'step': 10, 'tipo': 'num'},
    {'n': 'campana_escolar', 'et': '¿Hay campaña escolar?', 'u': '', 'mn': 0, 'mx': 1, 'step': 1, 'tipo': 'sel', 'op': ['Sin campaña', 'Con campaña']},
]


def predecir(valores):
    """Ecuación de la regresión: y = intercepto + suma(coef * x)."""
    y = MODELO["intercepto"]
    for nombre, coef in zip(MODELO["columnas"], MODELO["coeficientes"]):
        y += coef * valores[nombre]
    return max(y, 0)  # no tiene sentido un resultado negativo


@app.route("/", methods=["GET", "POST"])
def index():
    resultado, error, valores = None, None, {}
    if request.method == "POST":
        for c in CAMPOS:
            texto = request.form.get(c["n"], "").strip()
            valores[c["n"]] = texto
            try:
                v = float(texto)
            except ValueError:
                error = f"El campo «{c['et']}» debe ser un número."
                break
            # 3) Validación de rangos
            if not (c["mn"] <= v <= c["mx"]):
                error = f"«{c['et']}» debe estar entre {c['mn']} y {c['mx']} {c['u']}."
                break
            if c["tipo"] == "sel" and v not in (0, 1):
                error = f"«{c['et']}» solo admite 0 o 1."
                break
            valores[c["n"]] = v
        if error is None:
            y = predecir(valores)
            resultado = round(y)
    return render_template("index.html", campos=CAMPOS, valores=valores,
                           resultado=resultado, error=error,
                           titulo='Estimador de pedidos semanales de kits de robótica', unidad='pedidos de kits por semana', pie='Predicción con datos SIMULADOS. Es una estimación, no una garantía de ventas.',
                           metricas=MODELO["metricas"])


if __name__ == "__main__":
    app.run(debug=True)
