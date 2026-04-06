# Categorical Encoding: Techniques for Handling Categorical Variables

**Author:** Shuvam Banerji Seal  
**Last Updated:** April 2026  
**Category:** Data Preprocessing & Feature Engineering

## 1. Overview and Importance

Categorical encoding converts categorical (non-numeric) variables into numerical formats that machine learning algorithms can process. Different encoding strategies have varying impacts on model performance, interpretability, and memory usage.

### Why Categorical Encoding Matters

- **Algorithm Compatibility:** Most ML algorithms require numerical input
- **Information Preservation:** Preserves semantic relationships between categories
- **Memory Efficiency:** Optimizes storage for large categorical datasets
- **Model Performance:** Appropriate encoding can significantly boost model accuracy
- **Interpretability:** Some encodings make models more interpretable

### Categorical Variable Types

- **Ordinal:** Ordered categories (low, medium, high)
- **Nominal:** Unordered categories (red, green, blue)
- **Binary:** Two categories (yes/no, true/false)
- **High-Cardinality:** Many unique values (100+ categories)

## 2. One-Hot Encoding

### 2.1 Theory and Implementation

**Use Case:** Nominal categorical variables with low cardinality (< 10 categories)

**Mathematical Representation:**
```
Original: color ∈ {red, green, blue}

One-Hot:
  red:   [1, 0, 0]
  green: [0, 1, 0]
  blue:  [0, 0, 1]
```

**Implementation:**

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class OneHotEncodingImpl:
    """One-Hot Encoding implementation."""
    
    @staticmethod
    def manual_one_hot_encoding(categories_list):
        """Manually implement one-hot encoding."""
        unique_categories = list(set(categories_list))
        encoding_dict = {cat: np.zeros(len(unique_categories)) 
                        for cat in unique_categories}
        
        for i, cat in enumerate(unique_categories):
            encoding_dict[cat][i] = 1
        
        encoded = np.array([encoding_dict[cat] for cat in categories_list])
        return encoded, unique_categories
    
    @staticmethod
    def pandas_one_hot_encoding(df, column, prefix=None, drop_original=True):
        """One-hot encoding using pandas."""
        encoded = pd.get_dummies(df[column], prefix=prefix, dtype=int)
        
        if drop_original:
            df = df.drop(column, axis=1)
        
        return pd.concat([df, encoded], axis=1)
    
    @staticmethod
    def sklearn_one_hot_encoding(X, categorical_features=None):
        """One-hot encoding using scikit-learn."""
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore',
            dtype=int
        )
        
        return encoder.fit_transform(X)
    
    @staticmethod
    def one_hot_encoding_sparse(X, categorical_features=None):
        """One-hot encoding with sparse output (memory efficient)."""
        encoder = OneHotEncoder(
            sparse_output=True,  # Keep sparse format
            handle_unknown='ignore'
        )
        
        return encoder.fit_transform(X)

# Example
colors = ['red', 'blue', 'green', 'red', 'blue']
encoded, categories = OneHotEncodingImpl.manual_one_hot_encoding(colors)

print("Original categories:", colors)
print("Unique categories:", categories)
print("Encoded (manual):\n", encoded)

# Using pandas
df = pd.DataFrame({'color': colors, 'value': [1, 2, 3, 4, 5]})
df_encoded = OneHotEncodingImpl.pandas_one_hot_encoding(df, 'color', prefix='color')
print("\nPandas one-hot encoding:\n", df_encoded)
```

### 2.2 Handling High-Cardinality with One-Hot

```python
def one_hot_encode_high_cardinality(df, column, max_categories=10):
    """
    One-hot encode high-cardinality categorical with 'Other' category.
    """
    value_counts = df[column].value_counts()
    
    # Keep top N categories, group others as 'Other'
    top_categories = value_counts.head(max_categories).index
    df[column] = df[column].apply(
        lambda x: x if x in top_categories else 'Other'
    )
    
    return pd.get_dummies(df, columns=[column], dtype=int)

# Example
df = pd.DataFrame({
    'city': np.random.choice(['NY', 'LA', 'Chicago', 'Boston', 'Denver', 
                              'Seattle', 'Portland', 'Austin', 'Denver', 
                              'Houston', 'Miami'] * 10),
    'value': np.random.randn(110)
})

df_encoded = one_hot_encode_high_cardinality(df, 'city', max_categories=5)
print("Encoded shape:", df_encoded.shape)
print("Columns:", df_encoded.columns.tolist())
```

## 3. Label Encoding

### 3.1 Theory and Implementation

**Use Case:** Ordinal categorical variables (has inherent order)

**Mathematical Representation:**
```
Original: size ∈ {small, medium, large}

Label Encoded:
  small:  0
  medium: 1
  large:  2
```

**Implementation:**

```python
from sklearn.preprocessing import LabelEncoder

class LabelEncodingImpl:
    """Label Encoding implementation."""
    
    @staticmethod
    def manual_label_encoding(categories_list, category_order=None):
        """Manually implement label encoding."""
        if category_order is None:
            unique_categories = sorted(list(set(categories_list)))
        else:
            unique_categories = category_order
        
        encoding_dict = {cat: i for i, cat in enumerate(unique_categories)}
        encoded = np.array([encoding_dict[cat] for cat in categories_list])
        
        return encoded, encoding_dict
    
    @staticmethod
    def sklearn_label_encoding(categories_list):
        """Label encoding using scikit-learn."""
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(categories_list)
        
        return encoded, encoder
    
    @staticmethod
    def ordinal_label_encoding(df, column, order):
        """
        Label encoding with specified order (preserves ordinal relationship).
        
        Args:
            order: List of categories in ascending order
        """
        mapping = {cat: i for i, cat in enumerate(order)}
        return df[column].map(mapping)

# Example
sizes = ['small', 'large', 'medium', 'small', 'large', 'medium']
encoded, mapping = LabelEncodingImpl.manual_label_encoding(sizes)

print("Original:", sizes)
print("Label Encoded:", encoded)
print("Mapping:", mapping)

# Ordinal encoding with custom order
df = pd.DataFrame({'size': sizes, 'value': [1, 2, 3, 4, 5, 6]})
order = ['small', 'medium', 'large']
df['size_encoded'] = LabelEncodingImpl.ordinal_label_encoding(df, 'size', order)
print("\nOrdinal encoded:\n", df)
```

## 4. Target Encoding (Mean Encoding)

### 4.1 Theory and Implementation

**Use Case:** High-cardinality categorical variables, when you have a target variable

**Concept:** Replace each category with the mean target value for that category

**Mathematical Definition:**
```
Target encoding of category c:
  encoded_value = mean(y | x = c)
  
where y is the target variable
```

**Implementation:**

```python
class TargetEncodingImpl:
    """Target Encoding implementation."""
    
    @staticmethod
    def target_encode(X, y, categorical_column, smoothing=1.0):
        """
        Apply target encoding with optional smoothing to prevent overfitting.
        
        Smoothing formula:
          encoded_value = (count * mean + smoothing * global_mean) / (count + smoothing)
        """
        global_mean = y.mean()
        
        # Calculate mean target for each category
        category_means = X.groupby(categorical_column)[y.name].agg(['mean', 'count'])
        
        # Apply smoothing
        category_means['smoothed_mean'] = (
            (category_means['count'] * category_means['mean'] + smoothing * global_mean) /
            (category_means['count'] + smoothing)
        )
        
        # Create mapping
        encoding_dict = category_means['smoothed_mean'].to_dict()
        
        # Apply encoding
        X[f"{categorical_column}_encoded"] = X[categorical_column].map(encoding_dict)
        
        return X, encoding_dict
    
    @staticmethod
    def cross_validated_target_encoding(X, y, categorical_column, n_splits=5):
        """
        Cross-validated target encoding to prevent target leakage.
        """
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        encoded_column = np.zeros(len(X))
        
        for train_idx, test_idx in kf.split(X):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            
            # Fit on training fold
            category_means = X_train.groupby(categorical_column)[y_train.name].mean()
            
            # Apply to test fold
            encoded_column[test_idx] = X_test[categorical_column].map(category_means).values
        
        return encoded_column

# Example
df = pd.DataFrame({
    'city': ['NY', 'LA', 'NY', 'LA', 'NY', 'Chicago', 'Chicago', 'NY'],
    'price': [100, 150, 110, 140, 105, 80, 85, 115]
})

y = pd.Series(df['price'].values, name='target')
X = df[['city']]

X_encoded, encoding_dict = TargetEncodingImpl.target_encode(X, y, 'city', smoothing=1.0)

print("Original data:\n", df)
print("\nTarget encoding mapping:", encoding_dict)
print("\nEncoded data:\n", X_encoded)
```

## 5. Embedding-Based Encoding

### 5.1 Word Embeddings for Categories

```python
class EmbeddingEncodingImpl:
    """Embedding-based categorical encoding."""
    
    @staticmethod
    def embed_categories_word2vec(categories, embedding_dim=50):
        """
        Learn embeddings for categories using Word2Vec-like approach.
        Useful for high-cardinality categorical variables.
        """
        from gensim.models import Word2Vec
        
        # Treat each category occurrence as a sentence
        sentences = [[cat] for cat in categories]
        
        model = Word2Vec(
            sentences=sentences,
            vector_size=embedding_dim,
            window=1,
            min_count=1
        )
        
        embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
        return embeddings, model
    
    @staticmethod
    def learned_embeddings_from_nn(categories, target_variable, embedding_dim=10):
        """
        Learn embeddings from a neural network (embedding layer).
        """
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Create mapping for categories
        unique_categories = pd.factorize(categories)[0]
        n_categories = len(pd.unique(categories))
        
        # Create embedding layer
        embedding_layer = layers.Embedding(
            input_dim=n_categories + 1,
            output_dim=embedding_dim,
            input_length=1
        )
        
        # Build simple model
        model = keras.Sequential([
            embedding_layer,
            layers.Flatten(),
            layers.Dense(1)
        ])
        
        return model, embedding_layer
    
    @staticmethod
    def categorical_embeddings_pretrained(categories, embedding_type='fasttext'):
        """
        Use pre-trained embeddings (FastText, GloVe) for categories.
        """
        import fasttext
        
        # Load pre-trained FastText model
        # Note: Requires downloading model first
        try:
            model = fasttext.load_model('cc.en.300.bin')
            embeddings = {cat: model.get_vector(cat) for cat in categories}
            return embeddings
        except:
            print("Pre-trained model not available. Install fasttext and download model.")
            return {}

# Example with Word2Vec
categories = ['cat', 'dog', 'cat', 'bird', 'dog', 'cat']
embeddings, model = EmbeddingEncodingImpl.embed_categories_word2vec(
    categories, 
    embedding_dim=5
)

print("Category embeddings:")
for cat, embedding in embeddings.items():
    print(f"  {cat}: {embedding}")
```

## 6. Handling Unknown Categories

```python
class UnknownCategoryHandler:
    """Strategies for handling unknown categories in production."""
    
    @staticmethod
    def handle_unknown_with_none(known_categories, unknown_value=-1):
        """Assign unknown categories to a reserved value."""
        def encoder(cat):
            return cat if cat in known_categories else unknown_value
        return encoder
    
    @staticmethod
    def handle_unknown_with_frequency(category_counts, unknown_value=None):
        """Assign unknown categories to most frequent category."""
        most_frequent = category_counts.idxmax()
        
        def encoder(cat):
            return cat if cat in category_counts.index else most_frequent
        return encoder
    
    @staticmethod
    def handle_unknown_with_grouping(X, y, categorical_column, 
                                     min_frequency=10, other_label='Other'):
        """
        Group rare categories as 'Other' before encoding.
        Prevents too many low-frequency categories.
        """
        value_counts = X[categorical_column].value_counts()
        frequent_categories = value_counts[value_counts >= min_frequency].index
        
        X[categorical_column] = X[categorical_column].apply(
            lambda x: x if x in frequent_categories else other_label
        )
        
        return X

# Example
training_categories = ['A', 'B', 'C', 'D']
test_categories = ['A', 'B', 'E', 'C']  # 'E' is unknown

encoder = UnknownCategoryHandler.handle_unknown_with_none(
    training_categories, 
    unknown_value=-1
)

print("Encoded test categories:")
for cat in test_categories:
    print(f"  {cat}: {encoder(cat)}")
```

## 7. Complete Categorical Encoding Pipeline

```python
class CategoricalEncodingPipeline:
    """Complete pipeline for categorical encoding."""
    
    def __init__(self, encoding_strategy='onehot', categorical_features=None):
        self.encoding_strategy = encoding_strategy
        self.categorical_features = categorical_features
        self.encoders = {}
    
    def fit(self, X, y=None):
        """Fit encoders on training data."""
        if self.categorical_features is None:
            self.categorical_features = X.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()
        
        for col in self.categorical_features:
            if self.encoding_strategy == 'onehot':
                self.encoders[col] = OneHotEncoder(handle_unknown='ignore')
                self.encoders[col].fit(X[[col]])
            elif self.encoding_strategy == 'label':
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(X[col])
            elif self.encoding_strategy == 'target' and y is not None:
                target_means = X.groupby(col)[y.name].mean()
                self.encoders[col] = target_means.to_dict()
        
        return self
    
    def transform(self, X):
        """Apply fitted encoders."""
        X_transformed = X.copy()
        
        for col in self.categorical_features:
            if self.encoding_strategy == 'onehot':
                encoded = self.encoders[col].transform(X[[col]])
                # Handle the return format
                if hasattr(encoded, 'toarray'):
                    encoded = encoded.toarray()
                # Add encoded columns
                feature_names = self.encoders[col].get_feature_names_out([col])
                for i, fname in enumerate(feature_names):
                    X_transformed[fname] = encoded[:, i]
                X_transformed = X_transformed.drop(col, axis=1)
            
            elif self.encoding_strategy == 'label':
                X_transformed[col] = X_transformed[col].map(
                    lambda x: self.encoders[col].transform([x])[0]
                    if x in self.encoders[col].classes_
                    else -1
                )
            
            elif self.encoding_strategy == 'target':
                X_transformed[f"{col}_encoded"] = X_transformed[col].map(
                    self.encoders[col]
                ).fillna(X_transformed[col].map(self.encoders[col]).mean())
                X_transformed = X_transformed.drop(col, axis=1)
        
        return X_transformed

# Example
df = pd.DataFrame({
    'color': ['red', 'blue', 'red', 'green', 'blue'],
    'size': ['S', 'M', 'L', 'S', 'M'],
    'price': [10, 20, 30, 15, 25]
})

pipeline = CategoricalEncodingPipeline(
    encoding_strategy='onehot',
    categorical_features=['color', 'size']
)

df_encoded = pipeline.fit(df).transform(df)
print("Encoded dataframe:\n", df_encoded)
```

## 8. Comparison and Selection Guide

```python
class CategoricalEncodingAdvisor:
    """Recommend encoding strategy based on characteristics."""
    
    @staticmethod
    def recommend_encoding(df, column, target=None):
        """Recommend encoding strategy."""
        n_unique = df[column].nunique()
        cardinality_ratio = n_unique / len(df)
        
        recommendations = []
        
        # High cardinality
        if n_unique > 50:
            recommendations.append("Target Encoding (if target available)")
            recommendations.append("Embedding-based Encoding")
            recommendations.append("Frequency Encoding")
        
        # Medium cardinality
        elif n_unique > 10:
            if target is not None:
                recommendations.append("Target Encoding (best)")
            recommendations.append("One-Hot with frequency binning")
        
        # Low cardinality
        else:
            recommendations.append("One-Hot Encoding (best)")
            recommendations.append("Label Encoding (if ordinal)")
        
        return {
            'column': column,
            'n_unique': n_unique,
            'cardinality_ratio': cardinality_ratio,
            'recommendations': recommendations
        }

# Example
df = pd.DataFrame({
    'city': np.random.choice(['NY', 'LA', 'Chicago'] * 100),
    'color': np.random.choice(['red', 'blue'] * 150)
})

for col in ['city', 'color']:
    rec = CategoricalEncodingAdvisor.recommend_encoding(df, col)
    print(f"{col}: {rec['recommendations']}")
```

## 9. Quality Checklist

### Categorical Encoding Best Practices
- [ ] Identify all categorical variables
- [ ] Determine if categorical is ordinal or nominal
- [ ] Check cardinality for each variable
- [ ] Choose encoding strategy based on cardinality and algorithm
- [ ] Handle missing values before encoding
- [ ] Fit encoder on training data only
- [ ] Apply fitted encoder to test data
- [ ] Handle unknown categories in production
- [ ] Document encoding mappings
- [ ] Monitor category distribution in production

## 10. Authoritative Sources

1. **Scikit-learn Preprocessing Documentation** - https://scikit-learn.org/stable/modules/preprocessing.html
2. Micci-Barreca, D. (2001). "A preprocessing scheme for high-cardinality categorical attributes." *ACM SIGKDD Explorations Newsletter*, 3(1), 83-102.
3. "Target Encoding: A Primer" - Kaggle Blog
4. "Handling Categorical Data" - Fast.ai Practical Deep Learning Course
5. Peters, J., & Gromping, U. (2020). "Categorical regression for biological sciences." *Statistical Science*, 35(3), 439-461.

---

**Citation Format:**
Banerji Seal, S. (2026). "Categorical Encoding: Techniques for Handling Categorical Variables." LLM-Whisperer Skills Library.

**Version:** 1.0  
**Status:** Production Ready
