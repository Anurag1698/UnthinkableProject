# E-commerce Product Recommender

An AI-powered e-commerce product recommendation platform that combines recommendation algorithms with LLM-generated explanations to help users understand why specific products are recommended.

## Features

- **Hybrid Recommendation System**: Combines content-based and collaborative filtering
- **LLM-Powered Explanations**: Natural language explanations for each recommendation
- **User Behavior Tracking**: Logs views, clicks, purchases, and ratings
- **Interactive Dashboard**: Clean, modern web interface
- **Real-time Recommendations**: Get personalized recommendations instantly

## Project Structure

```
Unthinkable Task/
├── .env                      # Environment variables (SECRET_KEY, OPENAI_API_KEY, etc.)
├── README.md                 # Project README
├── PROJECT_SUMMARY.md        # Project summary
├── requirements.txt          # Python dependencies
├── init_db.py                # DB initialization script (imports backend.app)
├── start_app.py              # Script to start app (if present)
├── test.py                   # Small test script
├── backend/                  # Flask backend application
│   ├── app.py                # Main Flask app, models, API routes, recommendation engine
│   └── ...                   # other backend modules
├── frontend/                 # Frontend assets (if any)
├── templates/                # Jinja2 templates (dashboard.html)
│   └── dashboard.html
├── static/                   # Static assets (style.css, images, JS)
│   └── style.css
├── instance/                 # Flask instance folder (runtime data/config)
└── myenv/                    # Virtual environment (local)
    ├── pyvenv.cfg
    ├── Include/
    ├── Lib/
    │   └── site-packages/    # installed packages (flask, sklearn, openai, etc.)
    └── Scripts/              # activation scripts
```

## Setup Instructions

### 1. Prerequisites

- Python 3.10 or higher
- OPENROUTER API key (for LLM explanations)

### 2. Environment Setup

1. **Clone or download the project**
2. **Create and activate virtual environment**:

   ```bash
   py -3.10 -m venv myenv
   .\myenv\Scripts\Activate.ps1  # On Windows
   # or
   source myenv/bin/activate    # On Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Configuration

1. **Copy environment template**:

   ```bash
   copy env_template.txt .env
   ```

2. **Edit `.env` file** and add your OPENROUTER API key:
   ```
   OPENROUTER_API_KEY=your-openai-api-key-here
   SECRET_KEY=your-secret-key-change-this-in-production
   ```

### 4. Run the Application

1. **Start the Flask server**:

   ```bash
   python init_db.py
   python start_app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Usage Guide

### Getting Started

1. **Select a User**: Choose from the dropdown menu (sample users are pre-loaded)
2. **Get Recommendations**: Click "Get Recommendations" to see personalized product suggestions
3. **View Explanations**: Each recommendation includes an AI-generated explanation
4. **Interact with Products**: Use the action buttons to log interactions (view, click, purchase)

### Understanding Recommendations

The system uses two recommendation methods:

- **Content-Based**: Recommends products similar to what you've interacted with
- **Collaborative**: Recommends products liked by users with similar preferences

Each recommendation includes:

- Product details (name, description, price, rating)
- AI-generated explanation of why it's recommended
- Recommendation type (content-based or collaborative)

### User Interactions

The system tracks various user interactions:

- **View**: When you view a product
- **Click**: When you click on a product
- **Purchase**: When you purchase a product
- **Rating**: When you rate a product (1-5 stars)

These interactions help improve future recommendations.

## API Endpoints

### Recommendations

- `GET /api/recommend/<user_id>` - Get recommendations for a user

### Products

- `GET /api/products` - Get all products

### Users

- `GET /api/users` - Get all users
- `POST /api/users` - Create a new user

### Interactions

- `POST /api/interaction` - Log user interaction

## Technical Details

### Recommendation Engine

The system implements a hybrid recommendation approach:

1. **Content-Based Filtering**:

   - Uses TF-IDF vectorization on product descriptions
   - Calculates user preferences based on interaction history
   - Recommends products similar to user's preferences

2. **Collaborative Filtering**:
   - Creates user-item interaction matrix
   - Finds similar users using cosine similarity
   - Recommends products liked by similar users

### LLM Integration

- Uses OpenAI's GPT-3.5-turbo for generating explanations
- Provides context about user behavior and product features
- Generates natural, conversational explanations

### Database Schema

- **Products**: Product catalog with features and metadata
- **Users**: User information and preferences
- **UserInteractions**: Log of all user-product interactions

## Customization

### Adding New Products

You can add products programmatically or through the database:

```python
product = Product(
    name="New Product",
    description="Product description",
    category="Electronics",
    price=99.99,
    brand="Brand Name",
    rating=4.5,
    features='{"feature1": true, "feature2": "value"}'
)
db.session.add(product)
db.session.commit()
```

### Modifying Recommendation Logic

The recommendation engine can be customized by:

- Adjusting interaction weights in `calculate_user_preferences()`
- Modifying TF-IDF parameters
- Changing similarity thresholds
- Adding new recommendation algorithms

### Styling

The frontend uses modern CSS with:

- Responsive design
- Gradient backgrounds
- Card-based layouts
- Smooth animations
- Font Awesome icons

## Troubleshooting

### Common Issues

1. **OpenAI API Error**: Ensure your API key is correctly set in `.env`
2. **Database Errors**: Delete `ecommerce_recommender.db` to reset the database
3. **Port Already in Use**: Change the port in `app.py` or kill the existing process

### Debug Mode

The application runs in debug mode by default. To disable:

```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

## Future Enhancements

- User authentication and profiles
- Advanced recommendation algorithms (matrix factorization, deep learning)
- Real-time recommendation updates
- A/B testing for recommendation strategies
- Analytics dashboard for recommendation performance
- Mobile-responsive improvements
- Integration with external product catalogs

## License

This project is for educational and demonstration purposes. Feel free to modify and extend it for your needs.

## Support

For questions or issues, please check the troubleshooting section or review the code comments for implementation details.
