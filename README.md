# MovieApp ML - Hibrit Film Oneri Sistemi

Modern hibrit film oneri sistemi - Web arayuzu + REST API + Makine ogrenmesi algoritmalari.

## Ozellikler

### Web Arayuzu
- Ana sayfa (populer filmler/diziler)
- Film/dizi arama
- Detay sayfalari
- Kisisellestirilmis oneriler
- Izleme listesi

### Makine Ogrenmesi
- **Collaborative Filtering**: Matrix Factorization, SVD
- **Content-Based Filtering**: TF-IDF, Cosine Similarity
- **Hybrid System**: Agirlikli kombinasyon

### REST API
- JWT Authentication
- CORS destegi
- Tum islemler icin endpoint'ler

## Kurulum

### 1. Repository'yi klonlayin
```bash
git clone https://github.com/fthsrlk/MovieApp-ML.git
cd MovieApp-ML
```

### 2. Virtual environment olusturun
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Bagimliliklari yukleyin
```bash
pip install -r requirements.txt
```

### 4. Ortam degiskenlerini ayarlayin
```bash
cp .env.example .env
```

`.env` dosyasini duzenleyin:
```
TMDB_API_KEY=your_tmdb_api_key_here
SECRET_KEY=your_secret_key_here
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=True
```

### 5. Uygulamayi calistirin
```bash
python app.py
```

Tarayicinizda acin: http://localhost:5000

## Proje Yapisi

```
MovieApp-ML/
├── app.py                      # Ana uygulama (Web + API)
├── wsgi.py                     # Production server
├── requirements.txt
├── .env.example
├── README.md
└── ml_recommendation_engine/
    ├── api/
    │   └── app.py              # Sadece REST API
    ├── models/
    │   ├── collaborative.py    # Collaborative Filtering
    │   ├── content_based.py    # Content-Based Filtering
    │   └── hybrid.py           # Hybrid Recommender
    ├── data/
    │   ├── loader.py           # TMDb veri yukleme
    │   ├── preprocessor.py     # Veri isleme
    │   ├── items.csv           # 919 film/dizi
    │   └── ratings.csv         # 1673 kullanici degerlendirmesi
    └── app/
        ├── templates/          # HTML sablonlari
        │   ├── base.html
        │   ├── index.html
        │   ├── search.html
        │   ├── detail.html
        │   ├── recommendations.html
        │   └── watchlist.html
        └── static/
            ├── css/style.css
            └── js/main.js
```

## Web Sayfalari

| Sayfa | URL | Aciklama |
|-------|-----|----------|
| Ana Sayfa | `/` | Populer filmler ve diziler |
| Arama | `/search?q=batman` | Film/dizi arama |
| Film Detay | `/movie/27205` | Film detaylari |
| Dizi Detay | `/tv/1399` | Dizi detaylari |
| Oneriler | `/recommendations` | ML tabanli oneriler |
| Izleme Listesi | `/watchlist` | Kullanici listesi |

## API Endpoints

### Sistem
```http
GET /api/health              # Sistem durumu
```

### Oneriler
```http
GET /api/recommendations/1   # Kullanici onerileri
GET /api/similar/27205       # Benzer icerikler
```

### Arama
```http
GET /api/search?q=inception  # Arama
```

### Degerlendirme
```http
POST /api/ratings            # Puan ekleme
```

### Kimlik Dogrulama
```http
POST /api/auth               # Giris
```

### Izleme Listesi
```http
GET    /api/watchlist        # Liste getir
POST   /api/watchlist        # Ekle
DELETE /api/watchlist        # Cikar
```

## Ornek API Kullanimi

```python
import requests

# Sistem durumu
response = requests.get('http://localhost:5000/api/health')
print(response.json())

# Kullanici onerileri
response = requests.get('http://localhost:5000/api/recommendations/1')
recommendations = response.json()

# Benzer filmler
response = requests.get('http://localhost:5000/api/similar/27205')
similar = response.json()
```

## Production Deployment

### Gunicorn
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 wsgi:create_app
```

### Waitress (Windows)
```bash
waitress-serve --port=5000 --call wsgi:create_app
```

### Docker
```bash
docker build -t movieapp-ml .
docker run -p 5000:5000 -e TMDB_API_KEY=your_key movieapp-ml
```

## Teknoloji Stack

- **Backend**: Flask 2.0+
- **ML**: scikit-learn, pandas, numpy
- **API**: TMDb API v3
- **Auth**: JWT (PyJWT)
- **Frontend**: HTML, CSS, JavaScript

## Lisans

MIT License

## Iletisim

**Fatih Sarlak**
- GitHub: [@fthsrlk](https://github.com/fthsrlk)
