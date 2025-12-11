"""
MovieApp ML Backend - Ana Flask Uygulaması
Modern hibrit film öneri sistemi - Web UI + REST API

Bu dosya Flask web uygulamasının ana giriş noktasıdır.
TMDb API entegrasyonu, makine öğrenmesi modelleri,
Web arayüzü ve REST API burada yönetilir.
"""

import os
import json
import requests
import pickle
import pandas as pd
import numpy as np
import re
from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask_cors import CORS
import logging
from dotenv import load_dotenv
import sys
import datetime
import jwt
from datetime import timedelta

# Mevcut dizini ve ml_recommendation_engine'i ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'ml_recommendation_engine'))

# ML Recommendation Engine modüllerini içe aktar
try:
    from ml_recommendation_engine.models.collaborative import CollaborativeFiltering
    from ml_recommendation_engine.models.content_based import ContentBasedFiltering
    from ml_recommendation_engine.models.hybrid import HybridRecommender
    from ml_recommendation_engine.data.loader import TMDBDataLoader
    from ml_recommendation_engine.data.preprocessor import DataPreprocessor
except ImportError as e:
    print(f"Import hatası: {e}")
    print("ml_recommendation_engine modülleri yüklenemedi!")

# Ortam değişkenlerini yükle
load_dotenv()

# Flask uygulamasını başlat
app = Flask(__name__,
            template_folder='ml_recommendation_engine/app/templates',
            static_folder='ml_recommendation_engine/app/static')

CORS(app)  # CORS desteği ekle

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Yapılandırma
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'gizli-anahtar-degistirin')
TMDB_API_KEY = os.getenv('TMDB_API_KEY', '')
MODEL_DIR = os.getenv('MODEL_DIR', 'ml_recommendation_engine/models')
DATA_DIR = os.getenv('DATA_DIR', 'ml_recommendation_engine/data')

# Model ve veri yolları
CF_MODEL_PATH = os.path.join(MODEL_DIR, 'collaborative_model.pkl')
CB_MODEL_PATH = os.path.join(MODEL_DIR, 'content_based_model.pkl')
HYBRID_MODEL_PATH = os.path.join(MODEL_DIR, 'hybrid_model.pkl')
ITEMS_DATA_PATH = os.path.join(DATA_DIR, 'items.csv')
RATINGS_DATA_PATH = os.path.join(DATA_DIR, 'ratings.csv')

# Dizinleri oluştur
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Global değişkenler
cf_model = None
cb_model = None
hybrid_model = None
items_df = None
ratings_df = None
user_watchlist = {}

# --- Yardımcı Fonksiyonlar ---

def convert_numpy_types(obj):
    """NumPy değerlerini JSON serileştirilebilir tiplere dönüştürür"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def clean_text(text):
    """Metni JSON serileştirme için güvenli hale getirir"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\n\r\t\\"]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_models():
    """Modelleri yükle"""
    global cf_model, cb_model, hybrid_model

    if os.path.exists(CF_MODEL_PATH):
        cf_model = CollaborativeFiltering.load(CF_MODEL_PATH)
        logger.info("İşbirlikçi filtreleme modeli yüklendi")

    if os.path.exists(CB_MODEL_PATH):
        cb_model = ContentBasedFiltering.load(CB_MODEL_PATH)
        logger.info("İçerik tabanlı filtreleme modeli yüklendi")

    if os.path.exists(HYBRID_MODEL_PATH):
        hybrid_model = HybridRecommender.load(HYBRID_MODEL_PATH)
        logger.info("Hibrit model yüklendi")

def load_data():
    """Verileri yükle"""
    global items_df, ratings_df

    if os.path.exists(ITEMS_DATA_PATH):
        items_df = pd.read_csv(ITEMS_DATA_PATH)
        logger.info(f"İçerik verileri yüklendi: {len(items_df)} öğe")

    if os.path.exists(RATINGS_DATA_PATH):
        ratings_df = pd.read_csv(RATINGS_DATA_PATH)
        logger.info(f"Değerlendirme verileri yüklendi: {len(ratings_df)} değerlendirme")

def generate_token(user_id):
    """JWT token oluştur"""
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + timedelta(days=1)
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def verify_token(token):
    """JWT token doğrula"""
    try:
        return jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

def auth_required(f):
    """Kimlik doğrulama dekoratörü"""
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Yetkilendirme başlığı eksik'}), 401
        try:
            token = auth_header.split(' ')[1]
        except IndexError:
            return jsonify({'error': 'Geçersiz yetkilendirme başlığı formatı'}), 401
        if not verify_token(token):
            return jsonify({'error': 'Geçersiz veya süresi dolmuş token'}), 401
        return f(*args, **kwargs)
    decorated.__name__ = f.__name__
    return decorated

def search_movies(query, page=1):
    """TMDb API'dan film arama"""
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'query': query,
            'page': page,
            'language': 'tr-TR'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get('results', [])
    except Exception as e:
        logger.error(f"Film arama hatası: {str(e)}")
        return []

def search_tv_series(query, page=1):
    """TMDb API'dan dizi arama"""
    try:
        url = "https://api.themoviedb.org/3/search/tv"
        params = {
            'api_key': TMDB_API_KEY,
            'query': query,
            'page': page,
            'language': 'tr-TR'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get('results', [])
    except Exception as e:
        logger.error(f"Dizi arama hatası: {str(e)}")
        return []

def get_movie_details(movie_id):
    """TMDb API'dan film detayları"""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {
            'api_key': TMDB_API_KEY,
            'language': 'tr-TR',
            'append_to_response': 'credits,videos,similar'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Film detay hatası: {str(e)}")
        return None

def get_tv_details(tv_id):
    """TMDb API'dan dizi detayları"""
    try:
        url = f"https://api.themoviedb.org/3/tv/{tv_id}"
        params = {
            'api_key': TMDB_API_KEY,
            'language': 'tr-TR',
            'append_to_response': 'credits,videos,similar'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Dizi detay hatası: {str(e)}")
        return None

def get_popular_movies(page=1):
    """Popüler filmler"""
    try:
        url = "https://api.themoviedb.org/3/movie/popular"
        params = {'api_key': TMDB_API_KEY, 'language': 'tr-TR', 'page': page}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get('results', [])
    except Exception as e:
        logger.error(f"Popüler film hatası: {str(e)}")
        return []

def get_popular_tv(page=1):
    """Popüler diziler"""
    try:
        url = "https://api.themoviedb.org/3/tv/popular"
        params = {'api_key': TMDB_API_KEY, 'language': 'tr-TR', 'page': page}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get('results', [])
    except Exception as e:
        logger.error(f"Popüler dizi hatası: {str(e)}")
        return []

def get_ml_recommendations(user_id, n=10):
    """Makine öğrenmesi tabanlı öneriler"""
    try:
        if hybrid_model:
            return hybrid_model.recommend(user_id, n, ratings_df=ratings_df)
        if cb_model:
            return cb_model.recommend_for_user(user_id, n, ratings_df=ratings_df)
        return []
    except Exception as e:
        logger.error(f"ML öneri hatası: {str(e)}")
        return []

# --- Context Processors ---

@app.context_processor
def inject_current_year():
    return {'current_year': datetime.datetime.now().year}

@app.context_processor
def inject_user_watchlist():
    global user_watchlist
    return {'user_watchlist': user_watchlist}

# --- Web UI Routes ---

@app.route('/')
def index():
    """Ana sayfa"""
    popular_movies = get_popular_movies()[:8]
    popular_tv = get_popular_tv()[:8]
    return render_template('index.html',
                         popular_movies=popular_movies,
                         popular_tv=popular_tv)

@app.route('/search')
def search():
    """Arama sayfası"""
    query = request.args.get('q', '')
    if not query:
        return render_template('search.html')

    movies = search_movies(query)
    tv_series = search_tv_series(query)

    return render_template('search.html',
                         query=query,
                         movies=movies,
                         tv_series=tv_series)

@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    """Film detay sayfası"""
    movie = get_movie_details(movie_id)
    if not movie:
        return redirect(url_for('index'))
    return render_template('detail.html', item=movie, content_type='movie')

@app.route('/tv/<int:tv_id>')
def tv_detail(tv_id):
    """Dizi detay sayfası"""
    tv = get_tv_details(tv_id)
    if not tv:
        return redirect(url_for('index'))
    return render_template('detail.html', item=tv, content_type='tv')

@app.route('/recommendations')
def recommendations_page():
    """Öneriler sayfası"""
    user_id = request.args.get('user_id', 1, type=int)

    # ML önerileri
    ml_recommendations = []
    if hybrid_model or cb_model:
        raw_recommendations = get_ml_recommendations(user_id, n=20)
        for item_id, score in raw_recommendations:
            item_id = convert_numpy_types(item_id)
            if items_df is not None and item_id in items_df['item_id'].values:
                item_info = items_df[items_df['item_id'] == item_id].iloc[0].to_dict()
                item_info['score'] = convert_numpy_types(score)
                ml_recommendations.append(item_info)

    # Popüler içerikler
    popular_movies = get_popular_movies()[:10]
    popular_tv = get_popular_tv()[:10]

    return render_template('recommendations.html',
                         ml_recommendations=ml_recommendations,
                         popular_movies=popular_movies,
                         popular_tv=popular_tv,
                         user_id=user_id)

@app.route('/watchlist')
def watchlist_page():
    """İzleme listesi sayfası"""
    return render_template('watchlist.html', watchlist=user_watchlist)

# --- REST API Routes ---

@app.route('/api/health')
def health_check():
    """Sağlık kontrolü"""
    return jsonify({
        'status': 'ok',
        'version': '1.0.0',
        'models': {
            'collaborative': cf_model is not None,
            'content_based': cb_model is not None,
            'hybrid': hybrid_model is not None
        },
        'data': {
            'items': len(items_df) if items_df is not None else 0,
            'ratings': len(ratings_df) if ratings_df is not None else 0
        }
    })

@app.route('/api/recommendations/<int:user_id>')
def api_recommendations(user_id):
    """Kullanıcı için öneriler API endpoint'i"""
    limit = request.args.get('limit', default=10, type=int)
    strategy = request.args.get('strategy', default='hybrid')

    try:
        if strategy == 'collaborative' and cf_model:
            recommendations = cf_model.recommend(user_id, n=limit)
        elif strategy == 'content_based' and cb_model:
            recommendations = cb_model.recommend_for_user(user_id, n=limit, ratings_df=ratings_df)
        elif hybrid_model:
            recommendations = hybrid_model.recommend(user_id, n=limit, ratings_df=ratings_df)
        else:
            return jsonify({'error': 'Model yüklenmedi'}), 500

        results = []
        for item_id, score in recommendations:
            item_id = convert_numpy_types(item_id)
            score = convert_numpy_types(score)

            item_info = {}
            if items_df is not None and item_id in items_df['item_id'].values:
                item_info = items_df[items_df['item_id'] == item_id].iloc[0].to_dict()
                item_info = {k: convert_numpy_types(v) for k, v in item_info.items() if not pd.isna(v)}

            results.append({
                'item_id': item_id,
                'score': score,
                'title': clean_text(item_info.get('title', '')),
                'content_type': item_info.get('content_type', ''),
                'poster_path': item_info.get('poster_path', ''),
                'overview': clean_text(item_info.get('overview', ''))
            })

        return jsonify({
            'user_id': user_id,
            'strategy': strategy,
            'recommendations': results,
            'count': len(results)
        })

    except Exception as e:
        logger.error(f"Öneri API hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/similar/<int:item_id>')
def api_similar_items(item_id):
    """Benzer öğeler API endpoint'i"""
    limit = request.args.get('limit', default=10, type=int)

    if cb_model is None:
        return jsonify({'error': 'İçerik tabanlı model yüklenmedi'}), 500

    try:
        similar_items = cb_model.get_similar_items(item_id, n=limit)

        results = []
        for similar_id, score in similar_items:
            similar_id = convert_numpy_types(similar_id)
            score = convert_numpy_types(score)

            item_info = {}
            if items_df is not None and similar_id in items_df['item_id'].values:
                item_info = items_df[items_df['item_id'] == similar_id].iloc[0].to_dict()
                item_info = {k: convert_numpy_types(v) for k, v in item_info.items() if not pd.isna(v)}

            results.append({
                'item_id': similar_id,
                'score': score,
                'title': clean_text(item_info.get('title', '')),
                'content_type': item_info.get('content_type', ''),
                'poster_path': item_info.get('poster_path', ''),
                'overview': clean_text(item_info.get('overview', ''))
            })

        return jsonify({
            'item_id': item_id,
            'similar_items': results
        })

    except Exception as e:
        logger.error(f"Benzer öğe hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search')
def api_search():
    """Arama API endpoint'i"""
    query = request.args.get('q', '')
    content_type = request.args.get('type', 'all')
    page = request.args.get('page', 1, type=int)

    if not query:
        return jsonify({'error': 'Arama sorgusu gerekli'}), 400

    results = {'movies': [], 'tv_series': []}

    if content_type in ['all', 'movie']:
        results['movies'] = search_movies(query, page)

    if content_type in ['all', 'tv']:
        results['tv_series'] = search_tv_series(query, page)

    return jsonify(results)

@app.route('/api/ratings', methods=['POST'])
def api_add_rating():
    """Değerlendirme ekle"""
    global ratings_df

    data = request.json
    user_id = data.get('user_id')
    item_id = data.get('item_id')
    rating = data.get('rating')

    if not all([user_id, item_id, rating]):
        return jsonify({'error': 'Eksik parametreler'}), 400

    try:
        if ratings_df is None:
            ratings_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])

        mask = (ratings_df['user_id'] == user_id) & (ratings_df['item_id'] == item_id)
        if mask.any():
            ratings_df.loc[mask, 'rating'] = rating
            ratings_df.loc[mask, 'timestamp'] = pd.Timestamp.now().timestamp()
        else:
            new_rating = pd.DataFrame({
                'user_id': [user_id],
                'item_id': [item_id],
                'rating': [rating],
                'timestamp': [pd.Timestamp.now().timestamp()]
            })
            ratings_df = pd.concat([ratings_df, new_rating], ignore_index=True)

        ratings_df.to_csv(RATINGS_DATA_PATH, index=False)

        return jsonify({
            'success': True,
            'message': 'Değerlendirme kaydedildi'
        })

    except Exception as e:
        logger.error(f"Değerlendirme hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth', methods=['POST'])
def api_authenticate():
    """Kimlik doğrulama"""
    data = request.json
    username = data.get('username')
    password = data.get('password')

    # Basit kimlik doğrulama (production'da değiştirilmeli)
    if username == 'admin' and password == 'password':
        token = generate_token(user_id=1)
        return jsonify({'token': token, 'user_id': 1})

    return jsonify({'error': 'Geçersiz kimlik bilgileri'}), 401

@app.route('/api/watchlist', methods=['GET', 'POST', 'DELETE'])
def api_watchlist():
    """İzleme listesi API"""
    global user_watchlist

    if request.method == 'GET':
        return jsonify({'watchlist': list(user_watchlist.values())})

    elif request.method == 'POST':
        data = request.json
        item_id = str(data.get('item_id'))
        user_watchlist[item_id] = data
        return jsonify({'success': True, 'message': 'Listeye eklendi'})

    elif request.method == 'DELETE':
        data = request.json
        item_id = str(data.get('item_id'))
        if item_id in user_watchlist:
            del user_watchlist[item_id]
        return jsonify({'success': True, 'message': 'Listeden çıkarıldı'})

# --- Uygulama Başlatma ---

if __name__ == '__main__':
    logger.info("MovieApp ML Backend başlatılıyor...")

    # Verileri ve modelleri yükle
    load_data()
    load_models()

    # Uygulama host ve port bilgileri
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 5000))
    debug = os.getenv('API_DEBUG', 'True').lower() in ('true', '1', 't')

    logger.info(f"Uygulama başlatılıyor: http://{host}:{port}")
    logger.info(f"Debug modu: {debug}")

    # Flask uygulamasını başlat
    app.run(host=host, port=port, debug=debug)
