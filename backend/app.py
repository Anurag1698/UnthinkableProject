import os
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import openai
from openai import OpenAI
from typing import List, Dict, Any

# Get the directory containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Look for .env in parent directory
env_path = os.path.join(os.path.dirname(BASE_DIR), ".env")
# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///ecommerce_recommender.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "your-secret-key-here")

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)

# OpenAI configuration
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENROUTER_API_KEY")


# Database Models
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    category = db.Column(db.String(100))
    price = db.Column(db.Float)
    brand = db.Column(db.String(100))
    rating = db.Column(db.Float, default=0.0)
    image_url = db.Column(db.String(500))
    features = db.Column(db.Text)  # JSON string of product features
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "price": self.price,
            "brand": self.brand,
            "rating": self.rating,
            "image_url": self.image_url,
            "features": json.loads(self.features) if self.features else {},
            "created_at": self.created_at.isoformat(),
        }


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
        }


class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey("product.id"), nullable=False)
    interaction_type = db.Column(
        db.String(50), nullable=False
    )  # 'view', 'click', 'purchase', 'rating'
    rating = db.Column(db.Float)  # Only for rating interactions
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "product_id": self.product_id,
            "interaction_type": self.interaction_type,
            "rating": self.rating,
            "timestamp": self.timestamp.isoformat(),
        }


class Cart(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey("product.id"), nullable=False)
    quantity = db.Column(db.Integer, default=1)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        product = Product.query.get(self.product_id)
        return {
            "id": self.id,
            "user_id": self.user_id,
            "product_id": self.product_id,
            "quantity": self.quantity,
            "added_at": self.added_at.isoformat(),
            "product": product.to_dict() if product else None,
        }


class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    total_amount = db.Column(db.Float, nullable=False)
    status = db.Column(
        db.String(50), default="pending"
    )  # 'pending', 'completed', 'cancelled'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "total_amount": self.total_amount,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }


class OrderItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey("order.id"), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey("product.id"), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)

    def to_dict(self):
        product = Product.query.get(self.product_id)
        return {
            "id": self.id,
            "order_id": self.order_id,
            "product_id": self.product_id,
            "quantity": self.quantity,
            "price": self.price,
            "product": product.to_dict() if product else None,
        }


# Recommendation Engine
class RecommendationEngine:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        self.svd = TruncatedSVD(n_components=50)
        self.scaler = StandardScaler()
        # Configurable recommendation limits
        self.DEFAULT_RECOMMENDATIONS = 5
        self.LOW_ACTIVITY_RECOMMENDATIONS = 4

    def calculate_optimal_recommendations(self, user_id: int) -> int:
        """
        Calculate optimal number of recommendations based on user activity.
        Simple rule: 4 recommendations if <=3 items, otherwise 5 recommendations
        """
        # Get cart items count
        cart_count = Cart.query.filter_by(user_id=user_id).count()

        # Get view history count (last 30 days to avoid over-counting)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        view_count = UserInteraction.query.filter(
            UserInteraction.user_id == user_id,
            UserInteraction.interaction_type == "view",
            UserInteraction.timestamp >= thirty_days_ago,
        ).count()

        # Calculate total activity (cart + views)
        total_activity = cart_count + view_count

        # Simple logic: 4 if <=3 items, otherwise 5
        if total_activity <= 3:
            return self.LOW_ACTIVITY_RECOMMENDATIONS  # 4 recommendations
        else:
            return self.DEFAULT_RECOMMENDATIONS  # 5 recommendations

    def get_user_interactions(self, user_id: int) -> pd.DataFrame:
        """Get all user interactions as a DataFrame"""
        interactions = UserInteraction.query.filter_by(user_id=user_id).all()
        data = []
        for interaction in interactions:
            data.append(
                {
                    "product_id": interaction.product_id,
                    "interaction_type": interaction.interaction_type,
                    "rating": interaction.rating,
                    "timestamp": interaction.timestamp,
                }
            )
        return pd.DataFrame(data)

    def get_product_features(self) -> pd.DataFrame:
        """Get all products with their features"""
        products = Product.query.all()
        data = []
        for product in products:
            data.append(
                {
                    "id": product.id,
                    "name": product.name,
                    "description": product.description,
                    "category": product.category,
                    "brand": product.brand,
                    "price": product.price,
                    "rating": product.rating,
                    "features": product.features,
                }
            )
        return pd.DataFrame(data)

    def calculate_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """Calculate user preferences based on interactions"""
        interactions = self.get_user_interactions(user_id)
        if interactions.empty:
            return {}

        # Weight different interaction types
        weights = {"view": 1, "click": 2, "purchase": 5, "rating": 3}

        preferences = {}
        for _, interaction in interactions.iterrows():
            product_id = interaction["product_id"]
            interaction_type = interaction["interaction_type"]
            rating = interaction.get("rating", 0)

            if product_id not in preferences:
                preferences[product_id] = {"score": 0, "interactions": 0}

            # Calculate weighted score
            weight = weights.get(interaction_type, 1)
            if interaction_type == "rating" and rating > 0:
                weight *= rating / 5.0  # Normalize rating to 0-1

            preferences[product_id]["score"] += weight
            preferences[product_id]["interactions"] += 1

        return preferences

    def get_content_based_recommendations(
        self, user_id: int, n_recommendations: int = None
    ) -> List[Dict]:
        """Get content-based recommendations with dynamic count"""
        if n_recommendations is None:
            n_recommendations = self.calculate_optimal_recommendations(user_id)

        user_preferences = self.calculate_user_preferences(user_id)
        if not user_preferences:
            # If no user history, return popular products
            products = (
                Product.query.order_by(Product.rating.desc())
                .limit(n_recommendations)
                .all()
            )
            return [product.to_dict() for product in products]

        # Get products the user has interacted with
        liked_products = list(user_preferences.keys())

        # Get all products
        all_products = self.get_product_features()

        # Create feature vectors for products
        product_texts = []
        for _, product in all_products.iterrows():
            text = f"{product['name']} {product['description']} {product['category']} {product['brand']}"
            product_texts.append(text)

        # Fit TF-IDF and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(product_texts)

        # Get user profile (average of liked products)
        user_profile = np.zeros(tfidf_matrix.shape[1])
        for product_id in liked_products:
            product_idx = all_products[all_products["id"] == product_id].index[0]
            user_profile += (
                tfidf_matrix[product_idx].toarray().flatten()
                * user_preferences[product_id]["score"]
            )

        user_profile = user_profile / len(liked_products)

        # Calculate similarities
        similarities = cosine_similarity(
            user_profile.reshape(1, -1), tfidf_matrix
        ).flatten()

        # Get recommendations (exclude already interacted products)
        product_indices = np.argsort(similarities)[::-1]
        recommendations = []

        for idx in product_indices:
            product_id = all_products.iloc[idx]["id"]
            if product_id not in liked_products:
                product = Product.query.get(product_id)
                if product:
                    recommendations.append(product.to_dict())
                    if len(recommendations) >= n_recommendations:
                        break

        return recommendations

    def get_collaborative_recommendations(
        self, user_id: int, n_recommendations: int = None
    ) -> List[Dict]:
        """Get collaborative filtering recommendations with dynamic count"""
        if n_recommendations is None:
            n_recommendations = self.calculate_optimal_recommendations(user_id)

        # Get all user interactions
        all_interactions = UserInteraction.query.all()

        if len(all_interactions) < 10:  # Not enough data for collaborative filtering
            return self.get_content_based_recommendations(user_id, n_recommendations)

        # Create user-item matrix
        interactions_data = []
        for interaction in all_interactions:
            interactions_data.append(
                {
                    "user_id": interaction.user_id,
                    "product_id": interaction.product_id,
                    "rating": interaction.rating
                    or 1.0,  # Default rating for non-rating interactions
                    "interaction_type": interaction.interaction_type,
                }
            )

        df = pd.DataFrame(interactions_data)

        # Create pivot table
        user_item_matrix = df.pivot_table(
            index="user_id", columns="product_id", values="rating", fill_value=0
        )

        # Calculate user similarity
        user_similarities = cosine_similarity(user_item_matrix)

        # Get similar users
        user_idx = (
            user_item_matrix.index.get_loc(user_id)
            if user_id in user_item_matrix.index
            else None
        )
        if user_idx is None:
            return self.get_content_based_recommendations(user_id, n_recommendations)

        similar_users = np.argsort(user_similarities[user_idx])[::-1][
            1:6
        ]  # Top 5 similar users

        # Get recommendations from similar users
        recommendations = []
        for similar_user_idx in similar_users:
            similar_user_id = user_item_matrix.index[similar_user_idx]
            similar_user_items = user_item_matrix.loc[similar_user_id]

            # Get items the similar user liked but current user hasn't interacted with
            current_user_items = user_item_matrix.loc[user_id]
            for product_id, rating in similar_user_items.items():
                if rating > 0 and current_user_items[product_id] == 0:
                    product = Product.query.get(product_id)
                    if product and product not in [r for r in recommendations]:
                        recommendations.append(product.to_dict())
                        if len(recommendations) >= n_recommendations:
                            break

            if len(recommendations) >= n_recommendations:
                break

        return recommendations[:n_recommendations]

    def get_cart_and_views_based_recommendations(
        self, user_id: int, n_recommendations: int = None
    ) -> List[Dict]:
        """
        Get recommendations based on current cart contents and view history
        with dynamic recommendation count
        """
        if n_recommendations is None:
            n_recommendations = self.calculate_optimal_recommendations(user_id)

        # Get current cart items
        cart_items = Cart.query.filter_by(user_id=user_id).all()

        # Get user's recent view history (limit to last 20 to avoid over-processing)
        view_interactions = (
            UserInteraction.query.filter_by(user_id=user_id, interaction_type="view")
            .order_by(UserInteraction.timestamp.desc())
            .limit(20)  # Limit views to improve performance
            .all()
        )

        # Combine cart and views data
        cart_categories = set()
        cart_brands = set()
        cart_product_ids = set()
        viewed_categories = set()
        viewed_brands = set()
        viewed_product_ids = set()

        # Process cart items
        for cart_item in cart_items:
            product = Product.query.get(cart_item.product_id)
            if product:
                cart_categories.add(product.category)
                cart_brands.add(product.brand)
                cart_product_ids.add(product.id)

        # Process view history
        for view_interaction in view_interactions:
            product = Product.query.get(view_interaction.product_id)
            if product:
                viewed_categories.add(product.category)
                viewed_brands.add(product.brand)
                viewed_product_ids.add(product.id)

        # Combine all categories and brands for better recommendations
        all_categories = cart_categories.union(viewed_categories)
        all_brands = cart_brands.union(viewed_brands)
        all_product_ids = cart_product_ids.union(viewed_product_ids)

        if not all_product_ids:
            # If no cart or views, return popular products
            products = (
                Product.query.order_by(Product.rating.desc())
                .limit(n_recommendations)
                .all()
            )
            return [product.to_dict() for product in products]

        recommendations = []
        seen_product_ids = set()  # Track to avoid duplicates

        # 1. Products from same categories as cart/views but different brands
        for category in all_categories:
            category_products = (
                Product.query.filter(
                    Product.category == category, ~Product.id.in_(all_product_ids)
                )
                .order_by(Product.rating.desc())
                .limit(n_recommendations * 2)
                .all()
            )

            for product in category_products:
                if (
                    product.brand not in all_brands
                    and product.id not in seen_product_ids
                ):
                    recommendations.append(product.to_dict())
                    seen_product_ids.add(product.id)
                    if len(recommendations) >= n_recommendations:
                        break

            if len(recommendations) >= n_recommendations:
                break

        # 2. Products from brands user has shown interest in but different categories
        if len(recommendations) < n_recommendations:
            for brand in all_brands:
                brand_products = (
                    Product.query.filter(
                        Product.brand == brand, ~Product.id.in_(all_product_ids)
                    )
                    .order_by(Product.rating.desc())
                    .limit(n_recommendations)
                    .all()
                )

                for product in brand_products:
                    if (
                        product.category not in all_categories
                        and product.id not in seen_product_ids
                    ):
                        recommendations.append(product.to_dict())
                        seen_product_ids.add(product.id)
                        if len(recommendations) >= n_recommendations:
                            break

                if len(recommendations) >= n_recommendations:
                    break

        # 3. If we need more recommendations, add popular products from other categories
        if len(recommendations) < n_recommendations:
            remaining_needed = n_recommendations - len(recommendations)
            other_products = (
                Product.query.filter(
                    ~Product.category.in_(all_categories),
                    ~Product.id.in_(all_product_ids),
                )
                .order_by(Product.rating.desc())
                .limit(remaining_needed)
                .all()
            )

            for product in other_products:
                if product.id not in seen_product_ids:
                    recommendations.append(product.to_dict())
                    seen_product_ids.add(product.id)

        return recommendations[:n_recommendations]


# Initialize recommendation engine
recommendation_engine = RecommendationEngine()


# LLM Explanation Generator with Fallback Chain
class LLMExplanationGenerator:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.primary_model = os.getenv("LLM_MODEL", "meta-llama/llama-4-maverick:free")
        print(f"Loading .env from: {env_path}")
        print(api_key)
        print(self.primary_model)
        # Fallback models in case primary hits rate limit (ordered by preference)
        self.fallback_models = [
            "openai/gpt-oss-20b:free",
            "meta-llama/llama-4-scout:free",
            "deepseek/deepseek-chat-v3-0324:free",
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "qwen/qwen-2.5-7b-instruct:free",
            "deepseek/deepseek-chat-v3.1:free",
        ]
        print("=" * 60)
        print(f"Loading .env from: {env_path}")
        print("LLM Configuration:")
        print(f"  Primary Model: {self.primary_model}")
        print(f"  Fallback Models: {len(self.fallback_models)} configured")
        print(f"  API Key Status: {'✅ Valid' if api_key else '❌ Missing'}")
        print("=" * 60)

        if api_key and api_key != "your-openai-api-key-here":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=api_key
            )
            self.api_available = True
        else:
            self.client = None
            self.api_available = False
            print(
                "⚠️ WARNING: No valid API key found. Using fallback explanations only."
            )

    def _call_llm_with_retry(self, prompt: str) -> str:
        """
        Try to call LLM with automatic fallback to other models if rate limited
        """
        models_to_try = [self.primary_model] + self.fallback_models

        for idx, model in enumerate(models_to_try):
            try:
                print(f"🔄 Attempting LLM call with: {model}")

                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful Indian e-commerce recommendation assistant who provides unique, personalized explanations.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=150,
                    temperature=0.8,
                    extra_headers={
                        "HTTP-Referer": "https://ecommerce-recommender.com",
                        "X-Title": "E-commerce Recommendation System",
                    },
                )

                explanation = completion.choices[0].message.content.strip()
                print(f"✅ Success with {model}")
                return explanation

            except Exception as model_error:
                error_str = str(model_error)

                # Check if it's a rate limit error (429)
                if "429" in error_str or "rate limit" in error_str.lower():
                    print(f"⚠️ Rate limit hit for {model}")

                    # If this was the last model, give up
                    if idx == len(models_to_try) - 1:
                        print("❌ All models exhausted. Using fallback explanation.")
                        raise Exception("All LLM models rate limited")
                    else:
                        print(f"   Trying next fallback model...")
                        continue  # Try next model
                else:
                    # Other error (authentication, network, etc.)
                    print(f"❌ Error with {model}: {error_str[:100]}")
                    raise model_error

        # Should never reach here, but just in case
        raise Exception("No models available")

    def generate_explanation(
        self, user_id: int, product: Dict, recommendation_type: str
    ) -> str:
        """Generate explanation for why a product is recommended"""
        if not self.api_available:
            print("⚠️ API not available, using fallback explanation")
            return self._generate_fallback_explanation(
                user_id, product, recommendation_type
            )

        try:
            # Get user interaction history
            user_interactions = recommendation_engine.get_user_interactions(user_id)

            # Get user preferences
            user_preferences = recommendation_engine.calculate_user_preferences(user_id)

            # Get cart contents for cart-based recommendations
            cart_items = Cart.query.filter_by(user_id=user_id).all()
            cart_products = []
            for cart_item in cart_items:
                product_obj = Product.query.get(cart_item.product_id)
                if product_obj:
                    cart_products.append(
                        {
                            "name": product_obj.name,
                            "category": product_obj.category,
                            "brand": product_obj.brand,
                        }
                    )

            # Get view history for enhanced recommendations
            view_interactions = (
                UserInteraction.query.filter_by(
                    user_id=user_id, interaction_type="view"
                )
                .order_by(UserInteraction.timestamp.desc())
                .limit(10)
                .all()
            )
            viewed_products = []
            for view_interaction in view_interactions:
                product_obj = Product.query.get(view_interaction.product_id)
                if product_obj:
                    viewed_products.append(
                        {
                            "name": product_obj.name,
                            "category": product_obj.category,
                            "brand": product_obj.brand,
                        }
                    )

            # Get specific products user has interacted with
            interacted_products = []
            if not user_interactions.empty:
                for _, interaction in user_interactions.iterrows():
                    product_obj = Product.query.get(interaction["product_id"])
                    if product_obj:
                        interacted_products.append(
                            {
                                "name": product_obj.name,
                                "category": product_obj.category,
                                "brand": product_obj.brand,
                                "interaction_type": interaction["interaction_type"],
                            }
                        )

            # Create detailed context for the LLM
            context = f"""
User ID: {user_id}
Recommended Product: {product['name']}
Product Category: {product['category']}
Product Brand: {product['brand']}
Product Price: ₹{product['price']:,.2f}
Product Rating: {product['rating']}/5
Recommendation Type: {recommendation_type}

Current Cart Contents:
"""

            if cart_products:
                context += f"User has {len(cart_products)} items in cart:\n"
                for cart_prod in cart_products:
                    context += f"- {cart_prod['name']} ({cart_prod['category']}, {cart_prod['brand']})\n"
            else:
                context += "Cart is empty\n"

            context += "\nUser's Recent View History:\n"
            if viewed_products:
                context += (
                    f"User has viewed {len(viewed_products)} products recently:\n"
                )
                for viewed_prod in viewed_products[:5]:  # Limit to 5 most recent
                    context += f"- {viewed_prod['name']} ({viewed_prod['category']}, {viewed_prod['brand']})\n"
            else:
                context += "No recent view history\n"

            context += "\nUser's Previous Interactions:\n"
            if interacted_products:
                context += (
                    f"User has interacted with {len(interacted_products)} products:\n"
                )
                for prod in interacted_products[:5]:  # Limit to 5 most recent
                    context += f"- {prod['name']} ({prod['category']}, {prod['brand']}) - {prod['interaction_type']}\n"
            else:
                context += "No previous interactions (new user)\n"

            if user_preferences:
                context += f"\nUser preferences calculated from {len(user_preferences)} product interactions\n"

            prompt = f"""
You are an AI assistant that explains product recommendations for an Indian e-commerce platform.

Context: {context}

Generate a unique, personalized explanation for why THIS specific product is recommended to THIS user.
The explanation must be:
1. Completely unique and specific to this product and user combination
2. Reference the user's cart contents when available (for cart-views-based recommendations)
3. Reference the user's view history and browsing patterns when available
4. Mention specific features of this product that match the user's interests
5. Explain the recommendation type (cart-views-based: considers both cart and view history, content-based: similar to liked products)
6. Use Indian context and pricing (₹)
7. Conversational and friendly tone
8. Under 120 words
9. Never repeat generic explanations

Make it feel like a personal shopping assistant who knows this user's preferences and cart.
"""

            # Try to call LLM with automatic fallback
            return self._call_llm_with_retry(prompt)

        except Exception as e:
            print(f"❌ All LLM attempts failed: {str(e)[:100]}")
            return self._generate_fallback_explanation(
                user_id, product, recommendation_type
            )

    def _generate_fallback_explanation(
        self, user_id: int, product: Dict, recommendation_type: str
    ) -> str:
        """Generate a fallback explanation when LLM is not available"""
        print(f"📝 Using fallback explanation for product: {product['name']}")

        # Get cart contents for cart-based recommendations
        cart_items = Cart.query.filter_by(user_id=user_id).all()
        cart_categories = set()
        cart_brands = set()

        for cart_item in cart_items:
            product_obj = Product.query.get(cart_item.product_id)
            if product_obj:
                cart_categories.add(product_obj.category)
                cart_brands.add(product_obj.brand)

        # Get user's interaction history for more personalized fallback
        user_interactions = recommendation_engine.get_user_interactions(user_id)

        if recommendation_type == "cart-views-based":
            if cart_categories:
                if (
                    product["category"] in cart_categories
                    and product["brand"] not in cart_brands
                ):
                    return f"Since you have {product['category'].lower()} items in your cart, we recommend this {product['brand']} {product['name']} as a great alternative. It has a {product['rating']}-star rating and complements your current selection perfectly."
                elif product["category"] in cart_categories:
                    return f"This {product['name']} from {product['brand']} would be a perfect addition to your {product['category'].lower()} collection. With excellent reviews and great value at ₹{product['price']:,.2f}, it's a smart choice."
                else:
                    return f"Based on your cart contents, we think you'll love this {product['name']}. This {product['category'].lower()} from {product['brand']} has a {product['rating']}-star rating and great value at ₹{product['price']:,.2f}."
            else:
                return f"We recommend {product['name']} as it's one of our top-rated {product['category'].lower()} products. With a {product['rating']}-star rating and great value at ₹{product['price']:,.2f}, it's perfect for new customers."

        elif not user_interactions.empty:
            categories = (
                user_interactions["product_id"]
                .apply(
                    lambda x: (
                        Product.query.get(x).category if Product.query.get(x) else None
                    )
                )
                .dropna()
                .unique()
            )
            brands = (
                user_interactions["product_id"]
                .apply(
                    lambda x: (
                        Product.query.get(x).brand if Product.query.get(x) else None
                    )
                )
                .dropna()
                .unique()
            )

            category_match = product["category"] in categories
            brand_match = product["brand"] in brands

            if category_match and brand_match:
                return f"We recommend {product['name']} because you've shown interest in {product['category'].lower()} products from {product['brand']}. This item has a {product['rating']}-star rating and fits your established preferences perfectly."
            elif category_match:
                return f"Based on your interest in {product['category'].lower()} products, we think you'll love {product['name']}. This {product['brand']} product has excellent reviews and matches your browsing patterns."
            else:
                return f"We recommend {product['name']} because it's similar to products you've interacted with. This {product['category'].lower()} from {product['brand']} has a {product['rating']}-star rating and great value at ₹{product['price']:,.2f}."
        else:
            # New user fallback
            return f"We recommend {product['name']} as it's one of our top-rated {product['category'].lower()} products. With a {product['rating']}-star rating and great value at ₹{product['price']:,.2f}, it's perfect for new customers."


# Initialize LLM explanation generator
explanation_generator = LLMExplanationGenerator()


# API Routes
@app.route("/")
def index():
    """Serve the main dashboard"""
    return render_template("dashboard.html")


@app.route("/api/recommend/<int:user_id>", methods=["GET"])
def get_recommendations(user_id):
    """Get product recommendations for a user based on cart contents"""
    try:
        # Check if user exists
        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get cart and views-based recommendations (uses dynamic calculation)
        cart_views_recommendations = (
            recommendation_engine.get_cart_and_views_based_recommendations(user_id)
        )

        # No need for content-based backup since we're using dynamic count
        # Just use the cart-views recommendations

        # Generate explanations for each recommendation
        recommendations_with_explanations = []
        for rec in cart_views_recommendations:
            explanation = explanation_generator.generate_explanation(
                user_id, rec, "cart-views-based"
            )

            recommendations_with_explanations.append(
                {
                    "product": rec,
                    "explanation": explanation,
                    "recommendation_type": "cart-views-based",
                }
            )

        return jsonify(
            {
                "user_id": user_id,
                "recommendations": recommendations_with_explanations,
                "total_recommendations": len(recommendations_with_explanations),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/interaction", methods=["POST"])
def log_interaction():
    """Log user interaction with a product"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ["user_id", "product_id", "interaction_type"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Check if user and product exist
        user = User.query.get(data["user_id"])
        product = Product.query.get(data["product_id"])

        if not user:
            return jsonify({"error": "User not found"}), 404
        if not product:
            return jsonify({"error": "Product not found"}), 404

        # Create interaction
        interaction = UserInteraction(
            user_id=data["user_id"],
            product_id=data["product_id"],
            interaction_type=data["interaction_type"],
            rating=data.get("rating"),
        )

        db.session.add(interaction)
        db.session.commit()

        return jsonify(
            {
                "message": "Interaction logged successfully",
                "interaction_id": interaction.id,
            }
        )

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route("/api/products", methods=["GET"])
def get_products():
    """Get all products"""
    try:
        products = Product.query.all()
        return jsonify([product.to_dict() for product in products])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/users", methods=["GET"])
def get_users():
    """Get all users"""
    try:
        users = User.query.all()
        return jsonify([user.to_dict() for user in users])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/users", methods=["POST"])
def create_user():
    """Create a new user"""
    try:
        data = request.get_json()

        # Validate required fields
        if "username" not in data or "email" not in data:
            return jsonify({"error": "Missing required fields: username, email"}), 400

        # Check if user already exists
        existing_user = User.query.filter(
            (User.username == data["username"]) | (User.email == data["email"])
        ).first()

        if existing_user:
            return jsonify({"error": "User already exists"}), 400

        # Create user
        user = User(username=data["username"], email=data["email"])

        db.session.add(user)
        db.session.commit()

        return jsonify({"message": "User created successfully", "user": user.to_dict()})

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


# Cart API endpoints
@app.route("/api/cart/<int:user_id>", methods=["GET"])
def get_cart(user_id):
    """Get user's cart"""
    try:
        cart_items = Cart.query.filter_by(user_id=user_id).all()
        total_amount = 0

        for item in cart_items:
            product = Product.query.get(item.product_id)
            if product:
                total_amount += item.quantity * product.price

        return jsonify(
            {
                "user_id": user_id,
                "items": [item.to_dict() for item in cart_items],
                "total_amount": total_amount,
                "total_items": sum(item.quantity for item in cart_items),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cart/add", methods=["POST"])
def add_to_cart():
    """Add item to cart"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ["user_id", "product_id", "quantity"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Check if user and product exist
        user = User.query.get(data["user_id"])
        product = Product.query.get(data["product_id"])

        if not user:
            return jsonify({"error": "User not found"}), 404
        if not product:
            return jsonify({"error": "Product not found"}), 404

        # Check if item already exists in cart
        existing_item = Cart.query.filter_by(
            user_id=data["user_id"], product_id=data["product_id"]
        ).first()

        if existing_item:
            existing_item.quantity += data["quantity"]
        else:
            cart_item = Cart(
                user_id=data["user_id"],
                product_id=data["product_id"],
                quantity=data["quantity"],
            )
            db.session.add(cart_item)

        db.session.commit()

        return jsonify({"message": "Item added to cart successfully"})

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route("/api/cart/update", methods=["PUT"])
def update_cart_item():
    """Update cart item quantity"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ["user_id", "product_id", "quantity"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        cart_item = Cart.query.filter_by(
            user_id=data["user_id"], product_id=data["product_id"]
        ).first()

        if not cart_item:
            return jsonify({"error": "Cart item not found"}), 404

        if data["quantity"] <= 0:
            db.session.delete(cart_item)
        else:
            cart_item.quantity = data["quantity"]

        db.session.commit()

        return jsonify({"message": "Cart updated successfully"})

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route("/api/cart/remove", methods=["DELETE"])
def remove_from_cart():
    """Remove item from cart"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ["user_id", "product_id"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        cart_item = Cart.query.filter_by(
            user_id=data["user_id"], product_id=data["product_id"]
        ).first()

        if not cart_item:
            return jsonify({"error": "Cart item not found"}), 404

        db.session.delete(cart_item)
        db.session.commit()

        return jsonify({"message": "Item removed from cart successfully"})

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


# Orders API endpoints
@app.route("/api/orders/<int:user_id>", methods=["GET"])
def get_orders(user_id):
    """Get user's orders"""
    try:
        orders = (
            Order.query.filter_by(user_id=user_id)
            .order_by(Order.created_at.desc())
            .all()
        )

        orders_with_items = []
        for order in orders:
            order_items = OrderItem.query.filter_by(order_id=order.id).all()
            orders_with_items.append(
                {
                    "order": order.to_dict(),
                    "items": [item.to_dict() for item in order_items],
                }
            )

        return jsonify({"user_id": user_id, "orders": orders_with_items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/orders/create", methods=["POST"])
def create_order():
    """Create order from cart"""
    try:
        data = request.get_json()

        # Validate required fields
        if "user_id" not in data:
            return jsonify({"error": "Missing required field: user_id"}), 400

        user_id = data["user_id"]

        # Get cart items
        cart_items = Cart.query.filter_by(user_id=user_id).all()

        if not cart_items:
            return jsonify({"error": "Cart is empty"}), 400

        # Calculate total amount
        total_amount = 0
        for item in cart_items:
            product = Product.query.get(item.product_id)
            if product:
                total_amount += item.quantity * product.price

        # Create order
        order = Order(user_id=user_id, total_amount=total_amount, status="completed")
        db.session.add(order)
        db.session.flush()  # Get the order ID

        # Create order items
        for cart_item in cart_items:
            product = Product.query.get(cart_item.product_id)
            if product:
                order_item = OrderItem(
                    order_id=order.id,
                    product_id=cart_item.product_id,
                    quantity=cart_item.quantity,
                    price=product.price,
                )
                db.session.add(order_item)

        # Clear cart
        Cart.query.filter_by(user_id=user_id).delete()

        db.session.commit()

        return jsonify(
            {
                "message": "Order created successfully",
                "order_id": order.id,
                "total_amount": total_amount,
            }
        )

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


# Views API endpoints
@app.route("/api/views/<int:user_id>", methods=["GET"])
def get_views(user_id):
    """Get user's view history"""
    try:
        # Get all view interactions for the user
        view_interactions = (
            UserInteraction.query.filter_by(user_id=user_id, interaction_type="view")
            .order_by(UserInteraction.timestamp.desc())
            .all()
        )

        views_with_products = []
        for interaction in view_interactions:
            product = Product.query.get(interaction.product_id)
            if product:
                views_with_products.append(
                    {
                        "id": interaction.id,
                        "user_id": interaction.user_id,
                        "product_id": interaction.product_id,
                        "timestamp": interaction.timestamp.isoformat(),
                        "product": product.to_dict(),
                    }
                )

        return jsonify({"user_id": user_id, "views": views_with_products})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/views/remove", methods=["DELETE"])
def remove_from_views():
    """Remove item from view history"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ["user_id", "product_id"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Remove all view interactions for this product and user
        UserInteraction.query.filter_by(
            user_id=data["user_id"],
            product_id=data["product_id"],
            interaction_type="view",
        ).delete()

        db.session.commit()

        return jsonify({"message": "Item removed from view history successfully"})

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


# Database initialization function
def init_database():
    """Initialize database with sample data"""
    with app.app_context():
        db.create_all()

        # Add sample data if database is empty
        if Product.query.count() == 0:
            sample_products = [
                {
                    "name": "Wireless Bluetooth Headphones",
                    "description": "High-quality wireless headphones with active noise cancellation and 20-hour battery life",
                    "category": "Electronics",
                    "price": 2999.00,
                    "brand": "TechSound",
                    "rating": 4.5,
                    "image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400&h=300&fit=crop",
                    "features": '{"wireless": true, "noise_cancellation": true, "battery_life": "20 hours", "bluetooth_version": "5.0"}',
                },
                {
                    "name": "Smart Fitness Watch",
                    "description": "Advanced smartwatch with heart rate monitoring, GPS tracking, and water resistance",
                    "category": "Electronics",
                    "price": 8999.00,
                    "brand": "FitTech",
                    "rating": 4.3,
                    "image_url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400&h=300&fit=crop",
                    "features": '{"heart_rate": true, "gps": true, "water_resistant": true, "battery_life": "7 days"}',
                },
                {
                    "name": "Organic Coffee Beans",
                    "description": "Premium organic coffee beans from Colombia with medium roast",
                    "category": "Food & Beverage",
                    "price": 899.00,
                    "brand": "CoffeeCo",
                    "rating": 4.7,
                    "image_url": "https://images.unsplash.com/photo-1447933601403-0c6688de566e?w=400&h=300&fit=crop",
                    "features": '{"organic": true, "origin": "Colombia", "roast": "medium", "weight": "1 lb"}',
                },
                {
                    "name": "Yoga Mat Premium",
                    "description": "Non-slip yoga mat with 6mm thickness, perfect for all types of yoga practice",
                    "category": "Sports & Fitness",
                    "price": 1999.00,
                    "brand": "YogaLife",
                    "rating": 4.6,
                    "image_url": "https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?w=400&h=300&fit=crop",
                    "features": '{"non_slip": true, "thickness": "6mm", "material": "PVC", "size": "72x24 inches"}',
                },
                {
                    "name": "Leather Wallet",
                    "description": "Genuine leather wallet with RFID protection and 8 card slots",
                    "category": "Accessories",
                    "price": 2499.00,
                    "brand": "LeatherCraft",
                    "rating": 4.4,
                    "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400&h=300&fit=crop",
                    "features": '{"material": "leather", "rfid_protection": true, "card_slots": 8, "coin_pocket": true}',
                },
                {
                    "name": "Wireless Charging Pad",
                    "description": "Fast wireless charging pad compatible with all Qi-enabled devices",
                    "category": "Electronics",
                    "price": 1499.00,
                    "brand": "PowerTech",
                    "rating": 4.2,
                    "image_url": "https://images.unsplash.com/photo-1583394838336-acd977736f90?w=400&h=300&fit=crop",
                    "features": '{"wireless_charging": true, "qi_compatible": true, "fast_charging": true, "led_indicator": true}',
                },
                {
                    "name": "Protein Powder",
                    "description": "Whey protein powder with 25g protein per serving, vanilla flavor",
                    "category": "Sports & Fitness",
                    "price": 1299.00,
                    "brand": "FitLife",
                    "rating": 4.5,
                    "image_url": "https://images.unsplash.com/photo-1593095948071-474c5cc2989d?w=400&h=300&fit=crop",
                    "features": '{"protein_per_serving": "25g", "flavor": "vanilla", "servings": 30, "whey": true}',
                },
                {
                    "name": "Bluetooth Speaker",
                    "description": "Portable Bluetooth speaker with 360-degree sound and waterproof design",
                    "category": "Electronics",
                    "price": 3499.00,
                    "brand": "SoundWave",
                    "rating": 4.4,
                    "image_url": "https://images.unsplash.com/photo-1608043152269-423dbba4e7e1?w=400&h=300&fit=crop",
                    "features": '{"bluetooth": true, "waterproof": true, "360_sound": true, "battery_life": "12 hours"}',
                },
                {
                    "name": "Running Shoes",
                    "description": "Lightweight running shoes with responsive cushioning and breathable mesh",
                    "category": "Sports & Fitness",
                    "price": 4999.00,
                    "brand": "RunFast",
                    "rating": 4.6,
                    "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&h=300&fit=crop",
                    "features": '{"lightweight": true, "cushioning": "responsive", "material": "mesh", "arch_support": true}',
                },
                {
                    "name": "Ceramic Tea Set",
                    "description": "Handcrafted ceramic tea set with 6 cups, perfect for afternoon tea",
                    "category": "Home & Kitchen",
                    "price": 2299.00,
                    "brand": "TeaTime",
                    "rating": 4.3,
                    "image_url": "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=400&h=300&fit=crop",
                    "features": '{"material": "ceramic", "cups": 6, "dishwasher_safe": true, "handmade": true}',
                },
                {
                    "name": "Laptop Stand",
                    "description": "Adjustable aluminum laptop stand for better ergonomics and cooling",
                    "category": "Electronics",
                    "price": 1799.00,
                    "brand": "ErgoTech",
                    "rating": 4.4,
                    "image_url": "https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=400&h=300&fit=crop",
                    "features": '{"adjustable": true, "material": "aluminum", "portable": true, "cooling": true}',
                },
                {
                    "name": "Essential Oil Diffuser",
                    "description": "Ultrasonic essential oil diffuser with LED lights and timer",
                    "category": "Home & Kitchen",
                    "price": 1599.00,
                    "brand": "AromaLife",
                    "rating": 4.5,
                    "image_url": "https://images.unsplash.com/photo-1607853202273-797f1c22a7e2?w=400&h=300&fit=crop",
                    "features": '{"ultrasonic": true, "led_lights": true, "timer": true, "capacity": "300ml"}',
                },
                {
                    "name": "Resistance Bands Set",
                    "description": "Set of 5 resistance bands for full-body workouts at home",
                    "category": "Sports & Fitness",
                    "price": 799.00,
                    "brand": "FitBand",
                    "rating": 4.3,
                    "image_url": "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400&h=300&fit=crop",
                    "features": '{"set_of_5": true, "different_resistance": true, "portable": true, "door_anchor": true}',
                },
                {
                    "name": "Mechanical Keyboard",
                    "description": "RGB mechanical keyboard with blue switches for gaming and typing",
                    "category": "Electronics",
                    "price": 3999.00,
                    "brand": "GameTech",
                    "rating": 4.6,
                    "image_url": "https://images.unsplash.com/photo-1541140532154-b024d705b90a?w=400&h=300&fit=crop",
                    "features": '{"rgb_lighting": true, "blue_switches": true, "gaming": true, "backlit": true}',
                },
                {
                    "name": "Cast Iron Skillet",
                    "description": "Pre-seasoned cast iron skillet perfect for searing and baking",
                    "category": "Home & Kitchen",
                    "price": 1899.00,
                    "brand": "CookMaster",
                    "rating": 4.7,
                    "image_url": "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=400&h=300&fit=crop",
                    "features": '{"pre_seasoned": true, "cast_iron": true, "oven_safe": true, "size": "10 inch"}',
                },
                {
                    "name": "Wireless Mouse",
                    "description": "Ergonomic wireless mouse with precision tracking and long battery life",
                    "category": "Electronics",
                    "price": 1299.00,
                    "brand": "ClickTech",
                    "rating": 4.4,
                    "image_url": "https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=400&h=300&fit=crop",
                    "features": '{"wireless": true, "ergonomic": true, "precision": true, "battery_life": "12 months"}',
                },
                {
                    "name": "Herbal Tea Collection",
                    "description": "Assorted herbal tea collection with 20 different flavors",
                    "category": "Food & Beverage",
                    "price": 1199.00,
                    "brand": "TeaGarden",
                    "rating": 4.5,
                    "image_url": "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=400&h=300&fit=crop",
                    "features": '{"20_flavors": true, "herbal": true, "organic": true, "caffeine_free": true}',
                },
                {
                    "name": "Dumbbell Set",
                    "description": "Adjustable dumbbell set with multiple weight plates",
                    "category": "Sports & Fitness",
                    "price": 5999.00,
                    "brand": "PowerGym",
                    "rating": 4.6,
                    "image_url": "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400&h=300&fit=crop",
                    "features": '{"adjustable": true, "multiple_weights": true, "space_saving": true, "home_gym": true}',
                },
                {
                    "name": "Smartphone Case",
                    "description": "Protective smartphone case with wireless charging compatibility",
                    "category": "Accessories",
                    "price": 899.00,
                    "brand": "PhoneGuard",
                    "rating": 4.3,
                    "image_url": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=400&h=300&fit=crop",
                    "features": '{"protective": true, "wireless_charging": true, "drop_protection": true, "clear": true}',
                },
                {
                    "name": "Gaming Headset",
                    "description": "Professional gaming headset with 7.1 surround sound and noise cancellation",
                    "category": "Electronics",
                    "price": 4499.00,
                    "brand": "GameAudio",
                    "rating": 4.7,
                    "image_url": "https://images.unsplash.com/photo-1484704849700-f032a568e944?w=400&h=300&fit=crop",
                    "features": '{"surround_sound": "7.1", "noise_cancellation": true, "gaming": true, "microphone": true}',
                },
                {
                    "name": "Stainless Steel Water Bottle",
                    "description": "Insulated stainless steel water bottle that keeps drinks cold for 24 hours",
                    "category": "Accessories",
                    "price": 1299.00,
                    "brand": "HydroLife",
                    "rating": 4.6,
                    "image_url": "https://images.unsplash.com/photo-1602143407151-7111542de6e8?w=400&h=300&fit=crop",
                    "features": '{"insulated": true, "stainless_steel": true, "cold_retention": "24 hours", "leak_proof": true}',
                },
                {
                    "name": "Digital Camera",
                    "description": "Professional digital camera with 24MP sensor and 4K video recording",
                    "category": "Electronics",
                    "price": 24999.00,
                    "brand": "PhotoPro",
                    "rating": 4.8,
                    "image_url": "https://images.unsplash.com/photo-1502920917128-1aa500764cbd?w=400&h=300&fit=crop",
                    "features": '{"megapixels": 24, "4k_video": true, "professional": true, "interchangeable_lens": true}',
                },
                {
                    "name": "Memory Foam Pillow",
                    "description": "Contour memory foam pillow for better sleep and neck support",
                    "category": "Home & Kitchen",
                    "price": 1899.00,
                    "brand": "SleepWell",
                    "rating": 4.4,
                    "image_url": "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=400&h=300&fit=crop",
                    "features": '{"memory_foam": true, "contour": true, "neck_support": true, "hypoallergenic": true}',
                },
                {
                    "name": "Portable Power Bank",
                    "description": "High-capacity portable power bank with fast charging and multiple ports",
                    "category": "Electronics",
                    "price": 1999.00,
                    "brand": "PowerMax",
                    "rating": 4.5,
                    "image_url": "https://images.unsplash.com/photo-1609592807900-4a8b3b5a3a3a?w=400&h=300&fit=crop",
                    "features": '{"capacity": "20000mAh", "fast_charging": true, "multiple_ports": true, "portable": true}',
                },
                {
                    "name": "Organic Green Tea",
                    "description": "Premium organic green tea with antioxidants and natural flavor",
                    "category": "Food & Beverage",
                    "price": 699.00,
                    "brand": "GreenLeaf",
                    "rating": 4.6,
                    "image_url": "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=400&h=300&fit=crop",
                    "features": '{"organic": true, "antioxidants": true, "natural_flavor": true, "caffeine": "low"}',
                },
            ]

            for product_data in sample_products:
                product = Product(**product_data)
                db.session.add(product)

            # Add sample users
            sample_users = [
                {"username": "john_doe", "email": "john@example.com"},
                {"username": "jane_smith", "email": "jane@example.com"},
                {"username": "mike_wilson", "email": "mike@example.com"},
            ]

            for user_data in sample_users:
                user = User(**user_data)
                db.session.add(user)

            db.session.commit()


if __name__ == "__main__":
    init_database()
    app.run(debug=True, host="0.0.0.0", port=5000)
