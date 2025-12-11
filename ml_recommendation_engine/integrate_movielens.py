"""
MovieLens Dataset Integration Script
=====================================
Bu script, MovieLens veri setini mevcut TMDb tabanlı sistemle entegre eder.

Adımlar:
1. MovieLens movies.csv ve links.csv dosyalarını yükle
2. TMDb ID'lerini kullanarak eşleştirme yap
3. Mevcut items.csv'yi MovieLens filmleriyle genişlet
4. MovieLens ratings.csv'yi sistemimize uygun formata dönüştür
"""

import pandas as pd
import os
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MOVIELENS_DIR = os.path.join(DATA_DIR, 'ml-latest-small')

# TMDb API
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
TMDB_BASE_URL = "https://api.themoviedb.org/3"

def load_movielens_data():
    """MovieLens veri setini yükle"""
    print("MovieLens veri seti yükleniyor...")

    movies_df = pd.read_csv(os.path.join(MOVIELENS_DIR, 'movies.csv'))
    ratings_df = pd.read_csv(os.path.join(MOVIELENS_DIR, 'ratings.csv'))
    links_df = pd.read_csv(os.path.join(MOVIELENS_DIR, 'links.csv'))

    print(f"  - {len(movies_df)} film")
    print(f"  - {len(ratings_df)} değerlendirme")
    print(f"  - {len(links_df)} TMDb eşleşmesi")

    return movies_df, ratings_df, links_df

def load_existing_data():
    """Mevcut items.csv ve ratings.csv dosyalarını yükle"""
    print("\nMevcut veri dosyaları yükleniyor...")

    items_path = os.path.join(DATA_DIR, 'items.csv')
    ratings_path = os.path.join(DATA_DIR, 'ratings.csv')

    if os.path.exists(items_path):
        items_df = pd.read_csv(items_path)
        print(f"  - Mevcut items: {len(items_df)}")
    else:
        items_df = pd.DataFrame()
        print("  - items.csv bulunamadı, yeni oluşturulacak")

    if os.path.exists(ratings_path):
        existing_ratings = pd.read_csv(ratings_path)
        print(f"  - Mevcut ratings: {len(existing_ratings)}")
    else:
        existing_ratings = pd.DataFrame()
        print("  - ratings.csv bulunamadı, yeni oluşturulacak")

    return items_df, existing_ratings

def get_tmdb_movie_details(tmdb_id):
    """TMDb API'den film detaylarını al"""
    if not TMDB_API_KEY:
        return None

    try:
        url = f"{TMDB_BASE_URL}/movie/{int(tmdb_id)}"
        params = {
            'api_key': TMDB_API_KEY,
            'language': 'tr-TR',
            'append_to_response': 'credits'
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"    TMDb API hatası (ID: {tmdb_id}): {e}")
    return None

def create_item_from_movielens(movie_row, links_df, existing_tmdb_ids):
    """MovieLens filminden item oluştur"""
    movie_id = movie_row['movieId']

    # TMDb ID'yi bul
    link_row = links_df[links_df['movieId'] == movie_id]
    if link_row.empty or pd.isna(link_row['tmdbId'].values[0]):
        return None

    tmdb_id = int(link_row['tmdbId'].values[0])

    # Zaten varsa atla
    if tmdb_id in existing_tmdb_ids:
        return None

    # Temel bilgiler
    title = movie_row['title']
    genres = movie_row['genres'].replace('|', ', ') if pd.notna(movie_row['genres']) else ''

    # Yılı başlıktan çıkar (örn: "Toy Story (1995)" -> 1995)
    release_year = None
    if '(' in title and ')' in title:
        try:
            year_str = title[title.rfind('(')+1:title.rfind(')')]
            if year_str.isdigit():
                release_year = int(year_str)
        except:
            pass

    item = {
        'item_id': tmdb_id,
        'title': title,
        'overview': '',
        'poster_path': '',
        'release_date': f"{release_year}-01-01" if release_year else '',
        'vote_average': 0,
        'vote_count': 0,
        'popularity': 0,
        'original_language': 'en',
        'content_type': 'movie',
        'tmdb_details': '{}',
        'genres': genres,
        'release_year': release_year if release_year else 0,
        'popularity_norm': 0,
        'vote_average_norm': 0,
        'vote_count_norm': 0,
        'features_text': f"{title} {genres}"
    }

    return item

def enrich_with_tmdb(item, tmdb_id):
    """TMDb API'den ek bilgiler al"""
    details = get_tmdb_movie_details(tmdb_id)
    if details:
        item['overview'] = details.get('overview', '')
        item['poster_path'] = details.get('poster_path', '')
        item['release_date'] = details.get('release_date', item['release_date'])
        item['vote_average'] = details.get('vote_average', 0)
        item['vote_count'] = details.get('vote_count', 0)
        item['popularity'] = details.get('popularity', 0)
        item['original_language'] = details.get('original_language', 'en')

        # Genres from TMDb
        if 'genres' in details:
            item['genres'] = ', '.join([g['name'] for g in details['genres']])

        # Release year
        if details.get('release_date'):
            try:
                item['release_year'] = int(details['release_date'][:4])
            except:
                pass

        # Features text
        item['features_text'] = f"{item['title']} {item['genres']} {item['overview']}"

    return item

def convert_movielens_ratings(ml_ratings_df, links_df):
    """MovieLens ratings'i sistemimize uygun formata dönüştür"""
    print("\nMovieLens ratings dönüştürülüyor...")

    # MovieLens movieId -> TMDb ID eşleştirmesi
    ml_ratings_df = ml_ratings_df.merge(links_df[['movieId', 'tmdbId']], on='movieId', how='left')

    # TMDb ID'si olmayan kayıtları çıkar
    ml_ratings_df = ml_ratings_df.dropna(subset=['tmdbId'])
    ml_ratings_df['tmdbId'] = ml_ratings_df['tmdbId'].astype(int)

    # Sistemimize uygun format
    converted_ratings = pd.DataFrame({
        'user_id': ml_ratings_df['userId'] + 1000,  # Mevcut kullanıcılarla çakışmaması için offset
        'item_id': ml_ratings_df['tmdbId'],
        'rating': ml_ratings_df['rating'],
        'timestamp': ml_ratings_df['timestamp']
    })

    print(f"  - {len(converted_ratings)} rating dönüştürüldü")

    return converted_ratings

def integrate_movielens(enrich_from_tmdb=False, max_movies=None, sample_ratings=None):
    """
    MovieLens veri setini entegre et

    Args:
        enrich_from_tmdb: TMDb API'den ek bilgiler al (yavaş ama zengin)
        max_movies: İşlenecek maksimum film sayısı (None = hepsi)
        sample_ratings: Örnek rating sayısı (None = hepsi)
    """
    print("=" * 60)
    print("MovieLens Veri Seti Entegrasyonu")
    print("=" * 60)

    # Veri yükle
    movies_df, ml_ratings_df, links_df = load_movielens_data()
    existing_items, existing_ratings = load_existing_data()

    # Mevcut TMDb ID'leri
    existing_tmdb_ids = set()
    if not existing_items.empty and 'item_id' in existing_items.columns:
        existing_tmdb_ids = set(existing_items['item_id'].values)

    print(f"\nMevcut TMDb ID sayısı: {len(existing_tmdb_ids)}")

    # Yeni filmler oluştur
    print("\nYeni filmler oluşturuluyor...")
    new_items = []
    processed = 0
    added = 0

    movies_to_process = movies_df if max_movies is None else movies_df.head(max_movies)

    for idx, movie_row in movies_to_process.iterrows():
        item = create_item_from_movielens(movie_row, links_df, existing_tmdb_ids)

        if item:
            if enrich_from_tmdb:
                item = enrich_with_tmdb(item, item['item_id'])
                time.sleep(0.25)  # Rate limiting

            new_items.append(item)
            existing_tmdb_ids.add(item['item_id'])
            added += 1

            if added % 100 == 0:
                print(f"  - {added} yeni film eklendi...")

        processed += 1
        if processed % 500 == 0:
            print(f"  - {processed}/{len(movies_to_process)} film işlendi")

    print(f"\nToplam {added} yeni film eklendi")

    # Items birleştir ve kaydet
    if new_items:
        new_items_df = pd.DataFrame(new_items)

        if not existing_items.empty:
            combined_items = pd.concat([existing_items, new_items_df], ignore_index=True)
        else:
            combined_items = new_items_df

        # Normalizasyon
        if 'popularity' in combined_items.columns:
            max_pop = combined_items['popularity'].max()
            if max_pop > 0:
                combined_items['popularity_norm'] = combined_items['popularity'] / max_pop

        if 'vote_average' in combined_items.columns:
            combined_items['vote_average_norm'] = combined_items['vote_average'] / 10.0

        if 'vote_count' in combined_items.columns:
            max_votes = combined_items['vote_count'].max()
            if max_votes > 0:
                combined_items['vote_count_norm'] = combined_items['vote_count'] / max_votes

        # Kaydet
        items_path = os.path.join(DATA_DIR, 'items.csv')
        combined_items.to_csv(items_path, index=False)
        print(f"\nItems kaydedildi: {items_path}")
        print(f"  - Toplam: {len(combined_items)} film")

    # Ratings dönüştür ve kaydet
    print("\n" + "-" * 40)

    if sample_ratings:
        ml_ratings_sample = ml_ratings_df.sample(n=min(sample_ratings, len(ml_ratings_df)), random_state=42)
    else:
        ml_ratings_sample = ml_ratings_df

    converted_ratings = convert_movielens_ratings(ml_ratings_sample, links_df)

    if not existing_ratings.empty:
        combined_ratings = pd.concat([existing_ratings, converted_ratings], ignore_index=True)
    else:
        combined_ratings = converted_ratings

    # Duplikatları kaldır (aynı user-item çifti)
    combined_ratings = combined_ratings.drop_duplicates(subset=['user_id', 'item_id'], keep='last')

    ratings_path = os.path.join(DATA_DIR, 'ratings.csv')
    combined_ratings.to_csv(ratings_path, index=False)
    print(f"\nRatings kaydedildi: {ratings_path}")
    print(f"  - Toplam: {len(combined_ratings)} rating")

    print("\n" + "=" * 60)
    print("Entegrasyon tamamlandı!")
    print("=" * 60)

    return combined_items if new_items else existing_items, combined_ratings

def quick_integrate():
    """
    Hızlı entegrasyon (TMDb API çağrısı olmadan)
    MovieLens'teki tüm filmleri ve ratings'leri ekler
    """
    return integrate_movielens(enrich_from_tmdb=False, max_movies=None, sample_ratings=None)

def full_integrate(max_movies=500):
    """
    Tam entegrasyon (TMDb API ile zenginleştirilmiş)
    Daha yavaş ama daha detaylı
    """
    return integrate_movielens(enrich_from_tmdb=True, max_movies=max_movies, sample_ratings=50000)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        print("Tam entegrasyon başlatılıyor (TMDb API ile)...")
        full_integrate()
    else:
        print("Hızlı entegrasyon başlatılıyor...")
        quick_integrate()
