import os
import logging
import datetime
import jwt
from functools import wraps

from flask import Flask, request, jsonify, Response
import joblib
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from flasgger import Swagger
import yaml

JWT_SECRET = "MEUSEGREDOAQUI"
JWT_ALGORITHM = "HS256"
JWT_EXP_DELTA_SECONDS = 3600

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_modelo_flask")

DB_URL = "sqlite:///predictions.db"
engine = create_engine(DB_URL, echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    sepal_length = Column(Float, nullable=False)
    sepal_width = Column(Float, nullable=False)
    petal_length = Column(Float, nullable=False)
    petal_width = Column(Float, nullable=False)
    predicted_class = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(engine)

model = joblib.load("modelo_iris_LR.pkl")
logger.info("Modelo carregado com sucesso.")

app = Flask(__name__)

# Lê o YAML
with open('swagger.yaml', 'r') as f:
    swagger_template = yaml.safe_load(f)
swagger = Swagger(app, template=swagger_template)

predictions_cache = {}

TEST_USERNAME = "admin"
TEST_PASSWORD = "secret"

def create_token(username):
    payload = {
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXP_DELTA_SECONDS)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        """
        Middleware para verificar token JWT.
        ---
        security:
          - JWT: []
        responses:
          401:
            description: Token inválido ou ausente.
        """
        auth_header = request.headers.get("Authorization", "")
        parts = auth_header.split()

        # Checa se veio exatamente "JWT <token>"
        if len(parts) != 2 or parts[0] != "JWT":
            return jsonify({
                "error": "Token inválido ou ausente.",
                "message": "Use Authorization: 'JWT <token>'"
            }), 401
        
        token = parts[1]
        try:
            jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expirado."}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Token inválido."}), 401

        return f(*args, **kwargs)
    return decorated

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    username = data.get("username")
    password = data.get("password")
    if username == TEST_USERNAME and password == TEST_PASSWORD:
        token = create_token(username)
        return jsonify({"token": token})
    else:
        return jsonify({"error": "Credenciais inválidas"}), 401

@app.route("/predict", methods=["POST"])
@token_required
def predict():
    data = request.get_json(force=True)
    try:
        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])
    except (ValueError, KeyError) as e:
        logger.error("Dados de entrada inválidos: %s", e)
        return jsonify({"error": "Dados inválidos, verifique parâmetros"}), 400

    # Verificar se já está no cache
    features = (sepal_length, sepal_width, petal_length, petal_width)
    if features in predictions_cache:
        logger.info("Cache hit para %s", features)
        predicted_class = predictions_cache[features]
    else:
        # Rodar o modelo
        input_data = np.array([features])
        prediction = model.predict(input_data)
        predicted_class = int(prediction[0])
        # Armazenar no cache
        predictions_cache[features] = predicted_class
        logger.info("Cache updated para %s", features)

    # Armazenar em DB
    db = SessionLocal()
    new_pred = Prediction(
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_length=petal_length,
        petal_width=petal_width,
        predicted_class=predicted_class
    )
    db.add(new_pred)
    db.commit()
    db.close()

    return jsonify({"predicted_class": predicted_class})

@app.route("/predictions", methods=["GET"])
@token_required
def list_predictions():
    limit = int(request.args.get("limit", 10))
    offset = int(request.args.get("offset", 0))
    db = SessionLocal()
    preds = db.query(Prediction).order_by(Prediction.id.desc()).limit(limit).offset(offset).all()
    db.close()
    results = []
    for p in preds:
        results.append({
            "id": p.id,
            "sepal_length": p.sepal_length,
            "sepal_width": p.sepal_width,
            "petal_length": p.petal_length,
            "petal_width": p.petal_width,
            "predicted_class": p.predicted_class,
            "created_at": p.created_at.isoformat()
        })
    return jsonify(results)

@app.route("/", methods=["GET"])
def root():
    """
    Página inicial da API.
    ---
    tags:
      - Raiz
    responses:
      200:
        description: Página HTML de boas-vindas
        schema:
          type: string
          example: "<!DOCTYPE html>…"
    """
    # protocolo (http/https) e host
    proto = "https" if request.is_secure else "http"
    host = request.host

    # monta a URL completa para o Swagger UI (Flasgger default: /apidocs/)
    docs_url = f"{proto}://{host}/apidocs/#/"

    # ano atual
    year = datetime.datetime.utcnow().year

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>API Iris Prediction</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f7f9fc; color: #333; }}
    h1 {{ color: #2c3e50; }}
    ul {{ list-style-type: square; padding-left: 20px; }}
    a {{ color: #3498db; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .footer {{ margin-top: 40px; font-size: 0.9em; color: #777; }}
  </style>
</head>
<body>
  <h1>🌼 Bem-vindo à API de Previsão Iris com Flask!</h1>
  <p>Esta é uma API REST simples para prever espécies de Iris com base em medidas de pétalas e sépalas.</p>

  <h2>🔗 Endpoints Disponíveis:</h2>
  <ul>
    <li><code>GET  /</code>            – Gera esta página HTML</li>
    <li><code>POST /login</code>       – Autenticação e geração de token JWT</li>
    <li><code>POST /predict</code>     – Realizar predição (protegido por token JWT)</li>
    <li><code>GET  /predictions</code> – Listar predições (protegido por token JWT)</li>
    <li><code>GET  /health</code>      – Verificar status da API e do Banco</li>
  </ul>

  <h2>📄 Documentação Interativa:</h2>
  <p><a href="{docs_url}" target="_blank">Acesse o Swagger UI</a></p>

  <div class="footer">
    &copy; {year} API Iris – Desenvolvido com Python + Flask + Flasgger
  </div>
</body>
</html>"""

    return Response(html, mimetype="text/html")

@app.route("/health", methods=["GET"])
def health():
    # verifica conexão com o banco
    try:
        conn = engine.connect()
        conn.close()
        db_status = "up"
    except Exception as e:
        logger.error("Falha no health check do DB: %s", e)
        db_status = "down"

    return jsonify({
        "status": "ok" if db_status == "up" else "fail",
        "db": db_status
    }), 200
