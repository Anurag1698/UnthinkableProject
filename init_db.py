#!/usr/bin/env python3
"""
Database initialization script for E-commerce Product Recommender
Run this script to set up the database with sample data
"""

import os
import sys
from datetime import datetime
import json

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app import app, db, Product, User, UserInteraction, Cart, Order, OrderItem

def init_database():
    """Initialize the database with sample data"""
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Check if data already exists
        if Product.query.count() > 0:
            print("Clearing existing data and reinitializing...")
            # Clear existing data
            UserInteraction.query.delete()
            Cart.query.delete()
            OrderItem.query.delete()
            Order.query.delete()
            Product.query.delete()
            User.query.delete()
            db.session.commit()
        
        print("Initializing database with sample data...")
        
        # Sample products - 25 unique products with correct stock images
        sample_products = [
            {
                'name': 'Wireless Bluetooth Headphones',
                'description': 'High-quality wireless headphones with active noise cancellation and 20-hour battery life',
                'category': 'Electronics',
                'price': 2999.00,
                'brand': 'TechSound',
                'rating': 4.5,
                'image_url': 'https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400&h=300&fit=crop',
                'features': '{"wireless": true, "noise_cancellation": true, "battery_life": "20 hours", "bluetooth_version": "5.0"}'
            },
            {
                'name': 'Smart Fitness Watch',
                'description': 'Advanced smartwatch with heart rate monitoring, GPS tracking, and water resistance',
                'category': 'Electronics',
                'price': 8999.00,
                'brand': 'FitTech',
                'rating': 4.3,
                'image_url': 'https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400&h=300&fit=crop',
                'features': '{"heart_rate": true, "gps": true, "water_resistant": true, "battery_life": "7 days"}'
            },
            {
                'name': 'Organic Coffee Beans',
                'description': 'Premium organic coffee beans from Colombia with medium roast',
                'category': 'Food & Beverage',
                'price': 899.00,
                'brand': 'CoffeeCo',
                'rating': 4.7,
                'image_url': 'https://images.unsplash.com/photo-1447933601403-0c6688de566e?w=400&h=300&fit=crop',
                'features': '{"organic": true, "origin": "Colombia", "roast": "medium", "weight": "1 lb"}'
            },
            {
                'name': 'Yoga Mat Premium',
                'description': 'Non-slip yoga mat with 6mm thickness, perfect for all types of yoga practice',
                'category': 'Sports & Fitness',
                'price': 1999.00,
                'brand': 'YogaLife',
                'rating': 4.6,
                'image_url': 'https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?w=400&h=300&fit=crop',
                'features': '{"non_slip": true, "thickness": "6mm", "material": "PVC", "size": "72x24 inches"}'
            },
            {
                'name': 'Leather Wallet',
                'description': 'Genuine leather wallet with RFID protection and 8 card slots',
                'category': 'Accessories',
                'price': 2499.00,
                'brand': 'LeatherCraft',
                'rating': 4.4,
                'image_url': 'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400&h=300&fit=crop',
                'features': '{"material": "leather", "rfid_protection": true, "card_slots": 8, "coin_pocket": true}'
            },
            {
                'name': 'Wireless Charging Pad',
                'description': 'Fast wireless charging pad compatible with all Qi-enabled devices',
                'category': 'Electronics',
                'price': 1499.00,
                'brand': 'PowerTech',
                'rating': 4.2,
                'image_url': 'https://images.unsplash.com/photo-1583394838336-acd977736f90?w=400&h=300&fit=crop',
                'features': '{"wireless_charging": true, "qi_compatible": true, "fast_charging": true, "led_indicator": true}'
            },
            {
                'name': 'Protein Powder',
                'description': 'Whey protein powder with 25g protein per serving, vanilla flavor',
                'category': 'Sports & Fitness',
                'price': 1299.00,
                'brand': 'FitLife',
                'rating': 4.5,
                'image_url': 'https://images.unsplash.com/photo-1593095948071-474c5cc2989d?w=400&h=300&fit=crop',
                'features': '{"protein_per_serving": "25g", "flavor": "vanilla", "servings": 30, "whey": true}'
            },
            {
                'name': 'Bluetooth Speaker',
                'description': 'Portable Bluetooth speaker with 360-degree sound and waterproof design',
                'category': 'Electronics',
                'price': 3499.00,
                'brand': 'SoundWave',
                'rating': 4.4,
                'image_url': 'https://images.unsplash.com/photo-1608043152269-423dbba4e7e1?w=400&h=300&fit=crop',
                'features': '{"bluetooth": true, "waterproof": true, "360_sound": true, "battery_life": "12 hours"}'
            },
            {
                'name': 'Running Shoes',
                'description': 'Lightweight running shoes with responsive cushioning and breathable mesh',
                'category': 'Sports & Fitness',
                'price': 4999.00,
                'brand': 'RunFast',
                'rating': 4.6,
                'image_url': 'https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&h=300&fit=crop',
                'features': '{"lightweight": true, "cushioning": "responsive", "material": "mesh", "arch_support": true}'
            },
            {
                'name': 'Ceramic Tea Set',
                'description': 'Handcrafted ceramic tea set with 6 cups, perfect for afternoon tea',
                'category': 'Home & Kitchen',
                'price': 2299.00,
                'brand': 'TeaTime',
                'rating': 4.3,
                'image_url': 'https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=400&h=300&fit=crop',
                'features': '{"material": "ceramic", "cups": 6, "dishwasher_safe": true, "handmade": true}'
            },
            {
                'name': 'Laptop Stand',
                'description': 'Adjustable aluminum laptop stand for better ergonomics and cooling',
                'category': 'Electronics',
                'price': 1799.00,
                'brand': 'ErgoTech',
                'rating': 4.4,
                'image_url': 'https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=400&h=300&fit=crop',
                'features': '{"adjustable": true, "material": "aluminum", "portable": true, "cooling": true}'
            },
            {
                'name': 'Essential Oil Diffuser',
                'description': 'Ultrasonic essential oil diffuser with LED lights and timer',
                'category': 'Home & Kitchen',
                'price': 1599.00,
                'brand': 'AromaLife',
                'rating': 4.5,
                'image_url': 'https://images.unsplash.com/photo-1607853202273-797f1c22a7e2?w=400&h=300&fit=crop',
                'features': '{"ultrasonic": true, "led_lights": true, "timer": true, "capacity": "300ml"}'
            },
            {
                'name': 'Resistance Bands Set',
                'description': 'Set of 5 resistance bands for full-body workouts at home',
                'category': 'Sports & Fitness',
                'price': 799.00,
                'brand': 'FitBand',
                'rating': 4.3,
                'image_url': 'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400&h=300&fit=crop',
                'features': '{"set_of_5": true, "different_resistance": true, "portable": true, "door_anchor": true}'
            },
            {
                'name': 'Mechanical Keyboard',
                'description': 'RGB mechanical keyboard with blue switches for gaming and typing',
                'category': 'Electronics',
                'price': 3999.00,
                'brand': 'GameTech',
                'rating': 4.6,
                'image_url': 'https://images.unsplash.com/photo-1541140532154-b024d705b90a?w=400&h=300&fit=crop',
                'features': '{"rgb_lighting": true, "blue_switches": true, "gaming": true, "backlit": true}'
            },
            {
                'name': 'Cast Iron Skillet',
                'description': 'Pre-seasoned cast iron skillet perfect for searing and baking',
                'category': 'Home & Kitchen',
                'price': 1899.00,
                'brand': 'CookMaster',
                'rating': 4.7,
                'image_url': 'https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=400&h=300&fit=crop',
                'features': '{"pre_seasoned": true, "cast_iron": true, "oven_safe": true, "size": "10 inch"}'
            },
            {
                'name': 'Wireless Mouse',
                'description': 'Ergonomic wireless mouse with precision tracking and long battery life',
                'category': 'Electronics',
                'price': 1299.00,
                'brand': 'ClickTech',
                'rating': 4.4,
                'image_url': 'https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=400&h=300&fit=crop',
                'features': '{"wireless": true, "ergonomic": true, "precision": true, "battery_life": "12 months"}'
            },
            {
                'name': 'Herbal Tea Collection',
                'description': 'Assorted herbal tea collection with 20 different flavors',
                'category': 'Food & Beverage',
                'price': 1199.00,
                'brand': 'TeaGarden',
                'rating': 4.5,
                'image_url': 'https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=400&h=300&fit=crop',
                'features': '{"20_flavors": true, "herbal": true, "organic": true, "caffeine_free": true}'
            },
            {
                'name': 'Dumbbell Set',
                'description': 'Adjustable dumbbell set with multiple weight plates',
                'category': 'Sports & Fitness',
                'price': 5999.00,
                'brand': 'PowerGym',
                'rating': 4.6,
                'image_url': 'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400&h=300&fit=crop',
                'features': '{"adjustable": true, "multiple_weights": true, "space_saving": true, "home_gym": true}'
            },
            {
                'name': 'Smartphone Case',
                'description': 'Protective smartphone case with wireless charging compatibility',
                'category': 'Accessories',
                'price': 899.00,
                'brand': 'PhoneGuard',
                'rating': 4.3,
                'image_url': 'https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=400&h=300&fit=crop',
                'features': '{"protective": true, "wireless_charging": true, "drop_protection": true, "clear": true}'
            },
            {
                'name': 'Gaming Headset',
                'description': 'Professional gaming headset with 7.1 surround sound and noise cancellation',
                'category': 'Electronics',
                'price': 4499.00,
                'brand': 'GameAudio',
                'rating': 4.7,
                'image_url': 'https://images.unsplash.com/photo-1484704849700-f032a568e944?w=400&h=300&fit=crop',
                'features': '{"surround_sound": "7.1", "noise_cancellation": true, "gaming": true, "microphone": true}'
            },
            {
                'name': 'Stainless Steel Water Bottle',
                'description': 'Insulated stainless steel water bottle that keeps drinks cold for 24 hours',
                'category': 'Accessories',
                'price': 1299.00,
                'brand': 'HydroLife',
                'rating': 4.6,
                'image_url': 'https://images.unsplash.com/photo-1602143407151-7111542de6e8?w=400&h=300&fit=crop',
                'features': '{"insulated": true, "stainless_steel": true, "cold_retention": "24 hours", "leak_proof": true}'
            },
            {
                'name': 'Digital Camera',
                'description': 'Professional digital camera with 24MP sensor and 4K video recording',
                'category': 'Electronics',
                'price': 24999.00,
                'brand': 'PhotoPro',
                'rating': 4.8,
                'image_url': 'https://images.unsplash.com/photo-1502920917128-1aa500764cbd?w=400&h=300&fit=crop',
                'features': '{"megapixels": 24, "4k_video": true, "professional": true, "interchangeable_lens": true}'
            },
            {
                'name': 'Memory Foam Pillow',
                'description': 'Contour memory foam pillow for better sleep and neck support',
                'category': 'Home & Kitchen',
                'price': 1899.00,
                'brand': 'SleepWell',
                'rating': 4.4,
                'image_url': 'https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=400&h=300&fit=crop',
                'features': '{"memory_foam": true, "contour": true, "neck_support": true, "hypoallergenic": true}'
            },
            {
                'name': 'Portable Power Bank',
                'description': 'High-capacity portable power bank with fast charging and multiple ports',
                'category': 'Electronics',
                'price': 1999.00,
                'brand': 'PowerMax',
                'rating': 4.5,
                'image_url': 'https://images.unsplash.com/photo-1609592807900-4a8b3b5a3a3a?w=400&h=300&fit=crop',
                'features': '{"capacity": "20000mAh", "fast_charging": true, "multiple_ports": true, "portable": true}'
            },
            {
                'name': 'Organic Green Tea',
                'description': 'Premium organic green tea with antioxidants and natural flavor',
                'category': 'Food & Beverage',
                'price': 699.00,
                'brand': 'GreenLeaf',
                'rating': 4.6,
                'image_url': 'https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=400&h=300&fit=crop',
                'features': '{"organic": true, "antioxidants": true, "natural_flavor": true, "caffeine": "low"}'
            }
        ]
        
        # Add products to database
        for product_data in sample_products:
            product = Product(**product_data)
            db.session.add(product)
        
        # Sample users
        sample_users = [
            {'username': 'john_doe', 'email': 'john@example.com'},
            {'username': 'jane_smith', 'email': 'jane@example.com'},
            {'username': 'mike_wilson', 'email': 'mike@example.com'},
            {'username': 'sarah_jones', 'email': 'sarah@example.com'},
            {'username': 'alex_brown', 'email': 'alex@example.com'}
        ]
        
        # Add users to database
        for user_data in sample_users:
            user = User(**user_data)
            db.session.add(user)
        
        # Commit users and products first
        db.session.commit()
        
        # Sample user interactions
        sample_interactions = [
            # John's interactions (electronics enthusiast)
            {'user_id': 1, 'product_id': 1, 'interaction_type': 'view'},
            {'user_id': 1, 'product_id': 1, 'interaction_type': 'click'},
            {'user_id': 1, 'product_id': 1, 'interaction_type': 'purchase'},
            {'user_id': 1, 'product_id': 2, 'interaction_type': 'view'},
            {'user_id': 1, 'product_id': 2, 'interaction_type': 'click'},
            {'user_id': 1, 'product_id': 6, 'interaction_type': 'view'},
            {'user_id': 1, 'product_id': 6, 'interaction_type': 'purchase'},
            {'user_id': 1, 'product_id': 8, 'interaction_type': 'view'},
            
            # Jane's interactions (fitness enthusiast)
            {'user_id': 2, 'product_id': 2, 'interaction_type': 'view'},
            {'user_id': 2, 'product_id': 2, 'interaction_type': 'click'},
            {'user_id': 2, 'product_id': 2, 'interaction_type': 'purchase'},
            {'user_id': 2, 'product_id': 4, 'interaction_type': 'view'},
            {'user_id': 2, 'product_id': 4, 'interaction_type': 'purchase'},
            {'user_id': 2, 'product_id': 7, 'interaction_type': 'view'},
            {'user_id': 2, 'product_id': 7, 'interaction_type': 'click'},
            {'user_id': 2, 'product_id': 9, 'interaction_type': 'view'},
            {'user_id': 2, 'product_id': 9, 'interaction_type': 'purchase'},
            
            # Mike's interactions (coffee lover)
            {'user_id': 3, 'product_id': 3, 'interaction_type': 'view'},
            {'user_id': 3, 'product_id': 3, 'interaction_type': 'click'},
            {'user_id': 3, 'product_id': 3, 'interaction_type': 'purchase'},
            {'user_id': 3, 'product_id': 10, 'interaction_type': 'view'},
            {'user_id': 3, 'product_id': 10, 'interaction_type': 'click'},
            {'user_id': 3, 'product_id': 5, 'interaction_type': 'view'},
            
            # Sarah's interactions (mixed interests)
            {'user_id': 4, 'product_id': 1, 'interaction_type': 'view'},
            {'user_id': 4, 'product_id': 4, 'interaction_type': 'view'},
            {'user_id': 4, 'product_id': 4, 'interaction_type': 'click'},
            {'user_id': 4, 'product_id': 5, 'interaction_type': 'view'},
            {'user_id': 4, 'product_id': 5, 'interaction_type': 'purchase'},
            {'user_id': 4, 'product_id': 8, 'interaction_type': 'view'},
            
            # Alex's interactions (minimal)
            {'user_id': 5, 'product_id': 2, 'interaction_type': 'view'},
            {'user_id': 5, 'product_id': 6, 'interaction_type': 'view'},
        ]
        
        # Add interactions to database
        for interaction_data in sample_interactions:
            interaction = UserInteraction(**interaction_data)
            db.session.add(interaction)
        
        # Commit all changes
        db.session.commit()
        
        print(f"[SUCCESS] Added {len(sample_products)} products")
        print(f"[SUCCESS] Added {len(sample_users)} users")
        print(f"[SUCCESS] Added {len(sample_interactions)} interactions")
        print("Database initialization completed successfully!")

if __name__ == '__main__':
    init_database()
