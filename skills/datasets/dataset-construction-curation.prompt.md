# Dataset Construction and Curation: Building High-Quality Training Datasets

**Author:** Shuvam Banerji Seal  
**Last Updated:** April 2026  
**Category:** Dataset Engineering & Management

## 1. Overview and Importance

Dataset construction and curation is the foundational process of collecting, filtering, and organizing data for machine learning projects. The quality of a dataset directly determines the upper bound of model performance, making this arguably the most critical phase of ML development.

### The "Data is the New Oil" Principle

Quality datasets are the most valuable asset in ML:
- Models are only as good as their training data
- Garbage in = Garbage out
- Data quality often matters more than algorithm choice
- Top tech companies invest heavily in data annotation

### Dataset Curation Challenges

- **Data Collection:** Sourcing relevant, representative data
- **Labeling:** Expensive, time-consuming, subjective
- **Quality Control:** Ensuring consistency and correctness
- **Privacy & Ethics:** Handling sensitive information
- **Scalability:** Managing large-scale datasets efficiently

## 2. Dataset Collection Strategies

### 2.1 Data Sources

```python
class DataCollectionFramework:
    """Framework for collecting data from various sources."""
    
    @staticmethod
    def collect_from_public_repositories():
        """List of major public data sources."""
        sources = {
            'Kaggle': 'https://www.kaggle.com/datasets',
            'UCI ML Repository': 'https://archive.ics.uci.edu/ml/',
            'Google Dataset Search': 'https://datasetsearch.research.google.com/',
            'Hugging Face Datasets': 'https://huggingface.co/datasets',
            'OpenML': 'https://www.openml.org/',
            'AWS Open Data Registry': 'https://registry.opendata.aws/',
            'Paper with Code': 'https://paperswithcode.com/datasets',
            'GitHub': 'https://github.com (search for datasets)',
            'Government Data Portals': 'data.gov, etc.'
        }
        return sources
    
    @staticmethod
    def collect_from_apis(api_type='twitter', credentials=None):
        """
        Collect data from public APIs.
        Examples: Twitter API, Reddit API, YouTube API, etc.
        """
        
        if api_type == 'twitter':
            # Requires tweepy and API credentials
            try:
                import tweepy
                client = tweepy.Client(
                    bearer_token=credentials.get('bearer_token'),
                    consumer_key=credentials.get('consumer_key'),
                    consumer_secret=credentials.get('consumer_secret'),
                    access_token=credentials.get('access_token'),
                    access_token_secret=credentials.get('access_token_secret')
                )
                
                # Search tweets
                tweets = client.search_recent_tweets(
                    query="machine learning",
                    max_results=100
                )
                return tweets.data
            except ImportError:
                print("Install tweepy: pip install tweepy")
                return []
        
        return None
    
    @staticmethod
    def collect_from_web_scraping(url, parser='html.parser'):
        """
        Web scraping for data collection.
        IMPORTANT: Respect robots.txt and terms of service
        """
        try:
            from bs4 import BeautifulSoup
            import requests
            
            response = requests.get(url)
            soup = BeautifulSoup(response.content, parser)
            
            # Extract data (customize based on page structure)
            data = []
            for item in soup.find_all('div', class_='item'):
                data.append({
                    'title': item.find('h2').text,
                    'url': item.find('a')['href'],
                    'description': item.find('p').text
                })
            
            return data
        except ImportError:
            print("Install required packages: pip install beautifulsoup4 requests")
            return []

# Example usage
framework = DataCollectionFramework()
sources = framework.collect_from_public_repositories()
print("Available data sources:", sources)
```

### 2.2 Synthetic Data Generation

```python
class SyntheticDataGeneration:
    """Generate synthetic data for various purposes."""
    
    @staticmethod
    def generate_synthetic_tabular(n_samples=1000, n_features=10):
        """Generate synthetic tabular data."""
        from sklearn.datasets import make_classification, make_regression
        
        # Classification data
        X_class, y_class = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.7),
            n_redundant=int(n_features * 0.2),
            random_state=42
        )
        
        # Regression data
        X_reg, y_reg = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.8),
            random_state=42
        )
        
        return {
            'classification': (X_class, y_class),
            'regression': (X_reg, y_reg)
        }
    
    @staticmethod
    def generate_synthetic_time_series(n_samples=1000, n_features=3):
        """Generate synthetic time series data."""
        import pandas as pd
        
        # Time index
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')
        
        # Generate time series with trend and seasonality
        data = {}
        
        for i in range(n_features):
            trend = np.linspace(0, 10, n_samples)
            seasonality = 5 * np.sin(np.arange(n_samples) * 2 * np.pi / 365)
            noise = np.random.normal(0, 1, n_samples)
            
            data[f'feature_{i}'] = trend + seasonality + noise
        
        return pd.DataFrame(data, index=dates)
    
    @staticmethod
    def generate_synthetic_images(n_samples=100, image_size=28, num_classes=10):
        """
        Generate synthetic image data (e.g., random noise, geometric patterns).
        For realistic synthetic images, use GANs or diffusion models.
        """
        X = np.random.rand(n_samples, image_size, image_size, 3) * 255
        y = np.random.randint(0, num_classes, n_samples)
        
        return X.astype(np.uint8), y
    
    @staticmethod
    def generate_synthetic_text(n_samples=100, vocab_size=1000):
        """Generate synthetic text data."""
        np.random.seed(42)
        
        texts = []
        labels = []
        
        for i in range(n_samples):
            # Generate random word sequence
            word_ids = np.random.randint(0, vocab_size, size=np.random.randint(5, 20))
            text = ' '.join([f'word_{wid}' for wid in word_ids])
            texts.append(text)
            labels.append(np.random.randint(0, 2))  # Binary classification
        
        return texts, labels

# Example
synthetic_data = SyntheticDataGeneration.generate_synthetic_tabular(
    n_samples=1000,
    n_features=20
)

X_class, y_class = synthetic_data['classification']
print(f"Synthetic classification data shape: {X_class.shape}, {y_class.shape}")
```

## 3. Dataset Cleaning and Filtering

### 3.1 Duplicate and Near-Duplicate Detection

```python
import hashlib
from difflib import SequenceMatcher

class DatasetCleaning:
    """Dataset cleaning and filtering utilities."""
    
    @staticmethod
    def remove_exact_duplicates(df, subset=None):
        """Remove exact duplicate rows."""
        initial_len = len(df)
        
        if subset is None:
            df_clean = df.drop_duplicates()
        else:
            df_clean = df.drop_duplicates(subset=subset)
        
        removed = initial_len - len(df_clean)
        print(f"Removed {removed} exact duplicates")
        
        return df_clean
    
    @staticmethod
    def detect_near_duplicates(texts, similarity_threshold=0.95):
        """
        Detect near-duplicate texts using sequence similarity.
        Mathematical: SequenceMatcher uses Ratcliff/Obershelp algorithm
        """
        near_duplicates = []
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = SequenceMatcher(
                    None, 
                    texts[i], 
                    texts[j]
                ).ratio()
                
                if similarity >= similarity_threshold:
                    near_duplicates.append({
                        'idx1': i,
                        'idx2': j,
                        'similarity': similarity,
                        'text1': texts[i][:50],
                        'text2': texts[j][:50]
                    })
        
        return near_duplicates
    
    @staticmethod
    def detect_near_duplicates_hashing(texts, num_permutations=128):
        """
        MinHash for efficient near-duplicate detection.
        Better for large datasets.
        """
        from datasketch import MinHash
        
        hashes = []
        
        for text in texts:
            m = MinHash(num_perm=num_permutations)
            for word in text.split():
                m.update(word.encode('utf8'))
            hashes.append(m)
        
        # Find near-duplicates
        threshold = 0.5
        near_duplicates = []
        
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                similarity = hashes[i].jaccard(hashes[j])
                
                if similarity >= threshold:
                    near_duplicates.append({
                        'idx1': i,
                        'idx2': j,
                        'jaccard_similarity': similarity
                    })
        
        return near_duplicates
    
    @staticmethod
    def remove_low_quality_samples(df, quality_metrics_dict):
        """
        Remove samples based on quality metrics.
        
        Example quality_metrics_dict:
        {
            'column_name': {'type': 'min', 'value': 0.5},  # min value
            'text_length': {'type': 'min', 'value': 10}    # min length
        }
        """
        initial_len = len(df)
        
        for col, metrics in quality_metrics_dict.items():
            if metrics['type'] == 'min':
                df = df[df[col] >= metrics['value']]
            elif metrics['type'] == 'max':
                df = df[df[col] <= metrics['value']]
            elif metrics['type'] == 'range':
                df = df[(df[col] >= metrics['min']) & (df[col] <= metrics['max'])]
        
        removed = initial_len - len(df)
        print(f"Removed {removed} low-quality samples")
        
        return df
    
    @staticmethod
    def remove_outlier_samples(df, column, method='iqr', threshold=3.0):
        """Remove statistical outliers."""
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[column]))
            return df[z_scores < threshold]
        
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(df[[column]])
            
            return df[outlier_labels == 1]

# Example
df = pd.DataFrame({
    'text': [
        'This is good sample',
        'This is good sample',  # duplicate
        'This is a very very very very very very long text that might be spam' * 10,
        'Another good sample',
        ''  # empty
    ],
    'score': [0.9, 0.9, 0.1, 0.85, np.nan]
})

# Remove duplicates
df_clean = DatasetCleaning.remove_exact_duplicates(df)
print(f"After removing duplicates: {len(df_clean)} rows")

# Remove low quality
quality_metrics = {
    'score': {'type': 'min', 'value': 0.3}
}
df_clean = DatasetCleaning.remove_low_quality_samples(df_clean, quality_metrics)
print(f"After quality filtering: {len(df_clean)} rows")
```

## 4. Dataset Balancing and Stratification

### 4.1 Handling Class Imbalance

```python
class DatasetBalancing:
    """Balance dataset classes."""
    
    @staticmethod
    def analyze_class_distribution(y, verbose=True):
        """Analyze class imbalance."""
        unique, counts = np.unique(y, return_counts=True)
        
        distribution = dict(zip(unique, counts))
        imbalance_ratio = counts.max() / counts.min()
        
        if verbose:
            print("Class Distribution:")
            for label, count in distribution.items():
                pct = 100 * count / len(y)
                print(f"  Class {label}: {count} samples ({pct:.1f}%)")
            print(f"Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
        return distribution, imbalance_ratio
    
    @staticmethod
    def stratified_train_test_split(X, y, test_size=0.2, random_state=42):
        """
        Split data while preserving class distribution.
        CRITICAL for imbalanced datasets.
        """
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def stratified_k_fold(y, n_splits=5):
        """
        K-fold cross-validation preserving class distribution.
        """
        from sklearn.model_selection import StratifiedKFold
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        folds = []
        for train_idx, val_idx in skf.split(np.zeros(len(y)), y):
            folds.append((train_idx, val_idx))
        
        return folds

# Example
y = np.concatenate([np.zeros(950), np.ones(50)])  # 95:5 imbalance

distribution, ratio = DatasetBalancing.analyze_class_distribution(y)
print(f"\nImbalance ratio: {ratio:.1f}:1")

# Stratified split
X = np.random.randn(1000, 10)
X_train, X_test, y_train, y_test = DatasetBalancing.stratified_train_test_split(
    X, y, test_size=0.2
)

print(f"Training class distribution: {np.bincount(y_train.astype(int))}")
print(f"Test class distribution: {np.bincount(y_test.astype(int))}")
```

## 5. Dataset Documentation and Versioning

### 5.1 Dataset Metadata and Datasheets

```python
import json
from datetime import datetime

class DatasetDocumentation:
    """Document and manage datasets."""
    
    @staticmethod
    def create_dataset_card(
        dataset_name,
        description,
        source,
        size,
        features,
        target,
        license='CC-BY-4.0',
        authors=['Shuvam Banerji Seal']
    ):
        """
        Create a comprehensive dataset card (metadata).
        Following Hugging Face Dataset Card format.
        """
        
        dataset_card = {
            'metadata': {
                'name': dataset_name,
                'version': '1.0.0',
                'created_date': datetime.now().isoformat(),
                'description': description,
                'source': source,
                'authors': authors,
                'license': license
            },
            'dataset_statistics': {
                'total_samples': size,
                'train_samples': int(size * 0.7),
                'val_samples': int(size * 0.15),
                'test_samples': int(size * 0.15)
            },
            'features': features,
            'target': target,
            'quality_metrics': {
                'completeness': 'To be measured',
                'consistency': 'To be measured',
                'accuracy': 'To be validated'
            },
            'preprocessing_steps': [],
            'known_limitations': [],
            'citations': []
        }
        
        return dataset_card
    
    @staticmethod
    def save_dataset_card(dataset_card, filepath):
        """Save dataset card as JSON."""
        with open(filepath, 'w') as f:
            json.dump(dataset_card, f, indent=2)
    
    @staticmethod
    def create_dataset_version(
        dataset_path,
        version,
        changes,
        author
    ):
        """
        Create versioned dataset snapshot.
        """
        import shutil
        
        version_path = f"{dataset_path}_v{version}"
        shutil.copytree(dataset_path, version_path)
        
        version_log = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'author': author,
            'changes': changes
        }
        
        # Save version log
        log_path = f"{version_path}/VERSION_LOG.json"
        with open(log_path, 'w') as f:
            json.dump(version_log, f, indent=2)
        
        return version_path

# Example
dataset_card = DatasetDocumentation.create_dataset_card(
    dataset_name='Customer Reviews Dataset',
    description='Sentiment analysis dataset with customer reviews',
    source='Internal collection',
    size=10000,
    features=['text', 'length', 'date'],
    target='sentiment',
    authors=['Shuvam Banerji Seal']
)

print("Dataset Card:")
print(json.dumps(dataset_card, indent=2))
```

## 6. Quality Checklist for Dataset Curation

### Pre-Collection Phase
- [ ] Define data requirements and objectives
- [ ] Identify data sources
- [ ] Check data licensing and permissions
- [ ] Estimate required sample size
- [ ] Define quality standards

### During Collection Phase
- [ ] Track data collection process
- [ ] Monitor data quality in real-time
- [ ] Maintain data lineage
- [ ] Document any issues or anomalies
- [ ] Ensure proper data storage and backup

### Post-Collection Phase
- [ ] Remove duplicates and corrupted samples
- [ ] Handle missing values
- [ ] Check for outliers
- [ ] Validate data against requirements
- [ ] Create dataset documentation
- [ ] Establish version control

### Final Steps
- [ ] Create dataset card/metadata
- [ ] Document preprocessing steps
- [ ] Set up data governance
- [ ] Create usage guidelines
- [ ] Archive raw data

## 7. Authoritative Sources

1. Gebru, T., et al. (2021). "Datasheets for Datasets." *arXiv:1803.09010*
2. Buolamwini, B., & Gebru, T. (2018). "Gender Shades: Intersectional Accuracy Disparities." *FAccT*, 77-91.
3. "Data Curation Best Practices" - Kaggle Dataset Guidelines
4. "Hugging Face Dataset Cards" - https://huggingface.co/docs/datasets/dataset_card
5. "Datasets for Machine Learning" - Machine Learning Mastery
6. Polyzotis, N., et al. (2017). "Data Management Challenges in Production Machine Learning." *SIGMOD Record*

---

**Citation Format:**
Banerji Seal, S. (2026). "Dataset Construction and Curation: Building High-Quality Training Datasets." LLM-Whisperer Skills Library.

**Version:** 1.0  
**Status:** Production Ready
