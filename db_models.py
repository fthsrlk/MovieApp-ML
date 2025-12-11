"""
MovieApp Database Models
SQLAlchemy ORM models for User, Rating, and Watchlist
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from flask_bcrypt import Bcrypt
from datetime import datetime

db = SQLAlchemy()
bcrypt = Bcrypt()


class User(db.Model, UserMixin):
    """Kullanıcı modeli"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    # İlişkiler
    ratings = db.relationship('Rating', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    watchlist_items = db.relationship('WatchlistItem', backref='user', lazy='dynamic', cascade='all, delete-orphan')

    def set_password(self, password):
        """Şifreyi hashle ve kaydet"""
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        """Şifreyi doğrula"""
        return bcrypt.check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self):
        return f'<User {self.username}>'


class Rating(db.Model):
    """Kullanıcı değerlendirmesi modeli"""
    __tablename__ = 'ratings'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    item_id = db.Column(db.Integer, nullable=False)  # TMDb ID
    media_type = db.Column(db.String(10), default='movie')  # 'movie' veya 'tv'
    rating = db.Column(db.Float, nullable=False)  # 1.0 - 10.0
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Benzersiz kısıt: bir kullanıcı bir içeriği bir kez değerlendirebilir
    __table_args__ = (
        db.UniqueConstraint('user_id', 'item_id', 'media_type', name='unique_user_item_rating'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'item_id': self.item_id,
            'media_type': self.media_type,
            'rating': self.rating,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self):
        return f'<Rating user={self.user_id} item={self.item_id} rating={self.rating}>'


class WatchlistItem(db.Model):
    """İzleme listesi öğesi modeli"""
    __tablename__ = 'watchlist_items'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    item_id = db.Column(db.Integer, nullable=False)  # TMDb ID
    media_type = db.Column(db.String(10), default='movie')  # 'movie' veya 'tv'
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Benzersiz kısıt: bir kullanıcı bir içeriği bir kez ekleyebilir
    __table_args__ = (
        db.UniqueConstraint('user_id', 'item_id', 'media_type', name='unique_user_watchlist_item'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'item_id': self.item_id,
            'media_type': self.media_type,
            'added_at': self.added_at.isoformat() if self.added_at else None
        }

    def __repr__(self):
        return f'<WatchlistItem user={self.user_id} item={self.item_id}>'


class Item(db.Model):
    """Film/Dizi içerik modeli (cache için)"""
    __tablename__ = 'items'

    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, unique=True, nullable=False)  # TMDb ID
    title = db.Column(db.String(255), nullable=False)
    overview = db.Column(db.Text)
    poster_path = db.Column(db.String(255))
    backdrop_path = db.Column(db.String(255))
    release_date = db.Column(db.String(20))
    vote_average = db.Column(db.Float)
    vote_count = db.Column(db.Integer)
    popularity = db.Column(db.Float)
    original_language = db.Column(db.String(10))
    content_type = db.Column(db.String(10), default='movie')  # 'movie' veya 'tv'
    genres = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'item_id': self.item_id,
            'title': self.title,
            'overview': self.overview,
            'poster_path': self.poster_path,
            'backdrop_path': self.backdrop_path,
            'release_date': self.release_date,
            'vote_average': self.vote_average,
            'vote_count': self.vote_count,
            'popularity': self.popularity,
            'original_language': self.original_language,
            'content_type': self.content_type,
            'genres': self.genres
        }

    def __repr__(self):
        return f'<Item {self.title}>'
