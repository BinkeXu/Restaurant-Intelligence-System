import pandas as pd
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import os

class ReviewModel(BaseModel):
    """
    Pydantic model for validating and sanitizing restaurant review data.
    Ensures data integrity during the ingestion process.
    """
    restaurant: str = Field(alias="Restaurant")
    reviewer: str = Field(alias="Reviewer")
    review_text: str = Field(alias="Review")
    rating: float = Field(alias="Rating")
    metadata: str = Field(alias="Metadata")
    time: str = Field(alias="Time")
    pictures: int = Field(alias="Pictures")

    @validator("rating", pre=True)
    def parse_rating(cls, v):
        """
        Parses the rating value to a float. 
        Handles non-numeric values by defaulting to 0.0.
        """
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return 0.0
        return v or 0.0

    @validator("review_text", pre=True)
    def sanitize_review(cls, v):
        """
        Sanitizes the review text by stripping whitespace.
        Returns an empty string if the input is not a string.
        """
        if not isinstance(v, str):
            return ""
        return v.strip()

class ReviewDataLoader:
    """
    Handles loading of review data from CSV files and conversion to validated models.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_reviews(self) -> List[ReviewModel]:
        """
        Reads the CSV file using pandas and iterates through rows to create ReviewModel instances.
        Uses 'latin-1' encoding to handle special characters common in reviews.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        # Using latin-1 encoding as reviews often contain special characters (emojis, symbols)
        df = pd.read_csv(self.file_path, encoding='latin-1')
        
        reviews = []
        for _, row in df.iterrows():
            try:
                # Convert row to dict, handling NaNs
                row_dict = row.to_dict()
                # Pydantic will use the aliases to map CSV headers to model fields
                review = ReviewModel(**row_dict)
                reviews.append(review)
            except Exception:
                # Silently skip invalid rows to ensure the rest of the file is processed
                # In a production environment, this should be logged properly.
                pass
        
        return reviews

if __name__ == "__main__":
    # Test the loader
    loader = ReviewDataLoader("data/raw/Restaurant reviews.csv")
    reviews = loader.load_reviews()
    print(f"Loaded {len(reviews)} reviews.")
    if reviews:
        print(f"First review: {reviews[0].review_text[:100]}...")
