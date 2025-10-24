# Project structure

Below is the project structure (root: `c:\Users\kumar\Downloads\Unthinkable`).

```
Unthinkable/
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

Notes

- The Flask app is implemented in `backend/app.py`. This file loads configuration and initializes `SQLAlchemy` and other extensions.
- Place the `.env` file at the project root (`Unthinkable/.env`) so `backend/app.py` can load it; the code already looks for `.env` in the parent of `backend/`.
- The database used is SQLite by default and the DB file `ecommerce_recommender.db` will be created in the working directory when the app runs.
- `init_db.py` adds `backend/` to the Python path and imports models from `backend.app` to create and seed the database.

If you want, I can also:

- Add a link to this file from `README.md` (small edit).
- Generate a graphical diagram (SVG/PNG) if you prefer an image instead of ASCII.

```
# How to view
# - Open `PROJECT_STRUCTURE.md` in VS Code or any Markdown viewer.
```
