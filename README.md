# IBM Watson Studio Recommendation Systems

A comprehensive recommendation engine implementation using real data from the IBM Watson Studio platform. This project implements and compares five different recommendation techniques to provide personalized article recommendations to users.

## Project Overview

This project builds multiple recommendation systems to suggest relevant articles to users on the IBM Watson Studio platform. It explores various recommendation techniques, from simple popularity-based methods to advanced matrix factorization, and compares their effectiveness for different user scenarios.

## Features

The project implements **five distinct recommendation approaches**:

1. **Exploratory Data Analysis (Part I)**
   - Data exploration and statistical analysis
   - Handling missing values
   - Distribution analysis of user-article interactions

2. **Rank-Based Recommendations (Part II)**
   - Popularity-based recommendations
   - Best for new users (cold start problem)
   - Simple and fast implementation

3. **User-User Collaborative Filtering (Part III)**
   - Similarity-based recommendations using cosine similarity
   - Personalized recommendations based on similar users
   - Optimized ranking by user engagement

4. **Content-Based Recommendations (Part IV)**
   - TF-IDF vectorization of article titles
   - LSA (Latent Semantic Analysis) for dimensionality reduction
   - KMeans clustering for article similarity

5. **Matrix Factorization with SVD (Part V)**
   - Singular Value Decomposition for latent features
   - Scalable and accurate recommendations
   - Configurable number of latent features

## Getting Started

### Prerequisites

- Python 3.12+ (recommended)
- Jupyter Notebook
- Virtual environment (recommended)

### Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
jupyter>=1.0.0
notebook>=6.4.0
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/dsnd-recommendation-systems-project.git
cd dsnd-recommendation-systems-project
```

2. **Create and activate virtual environment:**
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook:**
```bash
cd Recommendations_IBM
jupyter notebook Recommendations_with_IBM.ipynb
```

## Dataset

The project uses two main datasets:
- **user-item-interactions.csv**: Contains 45,993 interactions between 5,149 users and 714 articles
- Article metadata including titles and IDs

## Implementation Details

### Part I: Exploratory Data Analysis
- Analyzed 45,993 user-article interactions
- Identified median user engagement: 3 articles
- Handled missing email values
- Created visualizations of user and article distributions

### Part II: Rank-Based Recommendations
**Functions implemented:**
- `get_top_articles(n, df)`: Returns top n article titles by popularity
- `get_top_article_ids(n, df)`: Returns top n article IDs

### Part III: User-User Collaborative Filtering
**Functions implemented:**
- `create_user_item_matrix(df)`: Creates binary user-item interaction matrix (5149 x 714)
- `find_similar_users(user_id)`: Finds similar users using cosine similarity
- `user_user_recs(user_id, m)`: Basic collaborative filtering recommendations
- `user_user_recs_part2(user_id, m)`: Improved version with popularity ranking

### Part IV: Content-Based Recommendations
**Approach:**
- TF-IDF vectorization (200 features, max_df=0.75, min_df=5)
- LSA with TruncatedSVD (50 components, ~X% variance explained)
- KMeans clustering (50 clusters based on elbow method)

**Functions implemented:**
- `get_similar_articles(article_id)`: Articles in same cluster
- `make_content_recs(article_id, n)`: Top n similar articles ranked by popularity

### Part V: Matrix Factorization (SVD)
**Configuration:**
- 200 latent features (optimal balance between performance and complexity)
- TruncatedSVD decomposition of user-item matrix

**Functions implemented:**
- `get_svd_similar_article_ids(article_id, vt)`: Recommendations using SVD factorization

## Results & Insights

### Method Comparison

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| Rank-Based | New users | Simple, fast | Not personalized |
| Collaborative Filtering | Active users | Personalized | Cold start problem |
| Content-Based | All users | Explainable | Limited by content |
| SVD | Production | Scalable, accurate | Hard to interpret |

### Recommended Hybrid Approach

For production deployment, use an **adaptive hybrid system**:
- New users (0-5 interactions): 70% Rank-Based + 30% Content
- Recent users (5-20 interactions): 50% Collaborative + 30% Content + 20% Rank
- Active users (20+ interactions): 60% SVD + 30% Collaborative + 10% Content

## Testing

All implementations include automated tests:

```bash
# Tests are integrated in the notebook
# Run all cells to execute tests automatically
# Expected output: "Nice job!" or "Great job!" for each part
```

**Test Coverage:**
- Part I: Statistical validation (t.sol_1_test)
- Part II: Top articles validation (t.sol_2_test)
- Part III: Matrix shape and function assertions
- Part IV: Recommendation validation
- Part V: SVD similarity assertions (t.sol_5_test)

##  Project Structure

```
dsnd-recommendation-systems-project/
├── README.md
├── requirements.txt
├── LICENSE.txt
├── CODEOWNERS
└── Recommendations_IBM/
    ├── Recommendations_with_IBM.ipynb    # Main notebook
    ├── project_tests.py                   # Test suite
    ├── top_5.p, top_10.p, top_20.p       # Test data
    └── data/
        └── user-item-interactions.csv     # Dataset
```

## Built With

* [Python 3.12](https://www.python.org/) - Programming language
* [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
* [NumPy](https://numpy.org/) - Numerical computing
* [Scikit-learn](https://scikit-learn.org/) - Machine learning algorithms
* [Matplotlib](https://matplotlib.org/) - Data visualization
* [Jupyter](https://jupyter.org/) - Interactive development environment

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgments

* IBM Watson Studio for providing the dataset
* Udacity Data Scientist Nanodegree program
* Project template and structure by Udacity

## Author

**Beatriz Barrios Perez**

---

*This project was completed as part of the Udacity Data Scientist Nanodegree program.*

