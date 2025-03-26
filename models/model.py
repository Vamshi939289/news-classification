import os
from transformers import pipeline

# Define the label-to-category mapping based on the model's output labels
label_to_category = {
    "ARTS": "arts",
    "ARTS & CULTURE": "arts & culture",
    "BLACK VOICES": "black voices",
    "BUSINESS": "business",
    "COLLEGE": "college",
    "COMEDY": "comedy",
    "CRIME": "crime",
    "CULTURE & ARTS": "culture & arts",
    "DIVORCE": "divorce",
    "EDUCATION": "education",
    "ENTERTAINMENT": "entertainment",
    "ENVIRONMENT": "environment",
    "FIFTY": "fifty",
    "FOOD & DRINK": "food & drink",
    "GOOD NEWS": "good news",
    "GREEN": "green",
    "HEALTHY LIVING": "healthy living",
    "HOME & LIVING": "home & living",
    "IMPACT": "impact",
    "LATINO VOICES": "latino voices",
    "MEDIA": "media",
    "MONEY": "money",
    "PARENTING": "parenting",
    "PARENTS": "parents",
    "POLITICS": "politics",
    "QUEER VOICES": "queer voices",
    "RELIGION": "religion",
    "SCIENCE": "science",
    "SPORTS": "sports",
    "STYLE": "style",
    "STYLE & BEAUTY": "style & beauty",
    "TASTE": "taste",
    "TECH": "technology",
    "THE WORLDPOST": "the worldpost",
    "TRAVEL": "travel",
    "U.S. NEWS": "u.s. news",
    "WEDDINGS": "weddings",
    "WEIRD NEWS": "weird news",
    "WELLNESS": "wellness",
    "WOMEN": "women",
    "WORLD NEWS": "world news",
    "WORLDPOST": "worldpost"
}

# Load the pre-trained model for text classification
classifier = pipeline("text-classification", model="dima806/news-category-classifier-distilbert")

def classify_news(article):
    """
    Classifies a news article into a predefined category using a Hugging Face Transformer model.

    Args:
        article (str): The news article content.

    Returns:
        str: The predicted category of the news article.
    """
    if not article or not isinstance(article, str):
        raise ValueError("Invalid input: Article must be a non-empty string.")

    max_length = 512
    truncated_article = article[:max_length]
    
    result = classifier(truncated_article)
    label = result[0]['label']

    # Map label to category or return the original label if mapping is not found
    category = label_to_category.get(label, label)

    return category


# Example usage:
if __name__ == "__main__":
    sample_article = "The latest Hollywood superhero film has shattered box office records, earning over $500 million in its opening weekend. Critics praise its stunning visual effects and compelling storyline, making it a must-watch for movie lovers."
    category = classify_news(sample_article)
    print(f"The news belongs to the '{category}' category.")
