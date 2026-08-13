from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset
data = pd.read_csv("data/cleaned_fashion_data.csv")

# Load saved model components
encoder = joblib.load("models/encoder.pkl")
encoded_features = joblib.load("models/encoded_features.pkl")

@app.get("/")
def home():
    from fastapi.responses import FileResponse
    return FileResponse("frontend/index.html")


@app.get("/data-info")
def data_info():
    return {
        "rows": len(data),
        "columns": len(data.columns)
    }
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity


class UserPreferences(BaseModel):
    gender: str | None = None
    occasion: str | None = None
    season: str | None = None
    color: str | None = None
    fit_type: str | None = None
    max_price: float | None = None
    fabric: str | None = None
    n: int = 5

def match_color(user_color):
    if user_color is None:
        return None

    user_color = user_color.lower()

    matching_colors = [
        color for color in data["color"].dropna().unique()
        if user_color in color.lower()
    ]

    if matching_colors:
        return matching_colors[0]

    return user_color
@app.post("/recommend")
def recommend(preferences: UserPreferences):
    matched_color = match_color(preferences.color)
    user_preferences = pd.DataFrame([{
        "color": matched_color,
        "occasion": preferences.occasion,
        "season": preferences.season,
        "Gender": preferences.gender,
        "fit_type": preferences.fit_type,
        "fabric": preferences.fabric,
    }])

    user_vector = encoder.transform(user_preferences)

    # Base ML similarity
    scores = cosine_similarity(
        user_vector,
        encoded_features
    )[0]

    # --------------------------------
    # Weighted preference scoring
    # --------------------------------

    for i, row in data.iterrows():

        score = 0
        total_weight = 0

        # Color - 40%
        if preferences.color:
            total_weight += 40

            if preferences.color.lower() in str(row["color"]).lower():
                score += 40

        # Occasion - 20%
        if preferences.occasion:
            total_weight += 20

            if str(row["occasion"]).lower() == preferences.occasion.lower():
                score += 20

        # Season - 15%
        if preferences.season:
            total_weight += 15

            if str(row["season"]).lower() == preferences.season.lower():
                score += 15

        # Gender - 10%
        if preferences.gender:
            total_weight += 10

            if str(row["Gender"]).lower() == preferences.gender.lower():
                score += 10

        # Fit - 7.5%
        if preferences.fit_type:
            total_weight += 7.5

            if str(row["fit_type"]).lower() == preferences.fit_type.lower():
                score += 7.5

        # Fabric - 7.5%
        if preferences.fabric:
            total_weight += 7.5

            if str(row["fabric"]).lower() == preferences.fabric.lower():
                score += 7.5

        # If user selected preferences,
        # calculate percentage based on them.
        if total_weight > 0:
            scores[i] = score / total_weight

    # Keep score between 0 and 1
    scores = scores.clip(0, 1)
    candidates = data.copy()
    candidates["similarity_score"] = scores
    # Filter by color
    if preferences.color:
        candidates = candidates[
            candidates["color"]
            .str.lower()
            .str.contains(preferences.color.lower(), na=False)
        ]
    # Filter by gender
    if preferences.gender:
        candidates = candidates[
            candidates["Gender"].str.lower() == preferences.gender.lower()
            ]

    # Filter by maximum price
    if preferences.max_price is not None and preferences.max_price > 0:
        candidates = candidates[
            candidates["price_pkr"] <= preferences.max_price
            ]
    candidates = candidates.sort_values(
        "similarity_score",
        ascending=False
    )

    results = candidates.head(preferences.n)

    return results[
        [
            "product_name",
            "main_category",
            "color",
            "occasion",
            "season",
            "Gender",
            "price_pkr",
            "similarity_score",
            "Image URL-src",
            "Products-href"
        ]
    ].to_dict(orient="records")
app.mount("/static", StaticFiles(directory=".", html=True), name="static")