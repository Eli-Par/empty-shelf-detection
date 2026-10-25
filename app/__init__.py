from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)

    CORS(app, origins=["http://localhost:4200"])

    from .prediction_routes import prediction
    app.register_blueprint(prediction)

    return app