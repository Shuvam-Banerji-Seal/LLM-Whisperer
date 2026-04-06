# Dataset Versioning and Management: Reproducibility and Governance

**Author:** Shuvam Banerji Seal  
**Last Updated:** April 2026  
**Category:** Dataset Engineering & MLOps

## 1. Overview and Importance

Dataset versioning tracks changes to datasets over time, enabling reproducibility, collaboration, and governance. Without versioning, you can't guarantee that results are reproducible—a critical requirement in ML.

### Why Dataset Versioning Matters

- **Reproducibility:** Exact recreation of results from any point in time
- **Accountability:** Track who changed what and when
- **Collaboration:** Multiple teams working on same datasets
- **Compliance:** Audit trail for regulatory requirements
- **Debugging:** Roll back to previous versions if issues arise
- **Impact Analysis:** Understand how data changes affect models

### The Dataset Versioning Problem

Unlike code versioning (git), datasets are:
- **Large:** Can't efficiently store every version
- **Binary:** Can't compute meaningful diffs
- **Sensitive:** Privacy concerns with storage
- **Complex:** Multiple formats and dependencies

## 2. Dataset Versioning Strategies

### 2.1 Snapshot-Based Versioning

```python
import hashlib
import json
from datetime import datetime
import shutil
from pathlib import Path

class DatasetVersioning:
    """Dataset versioning and management utilities."""
    
    @staticmethod
    def calculate_dataset_hash(filepath):
        """
        Calculate SHA-256 hash of dataset for integrity checking.
        
        Mathematical: SHA-256 produces 256-bit (32-byte) cryptographic hash
        Collision resistant: Probability of collision ≈ 2^-128
        """
        sha256_hash = hashlib.sha256()
        
        # Read file in chunks to handle large files
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    @staticmethod
    def create_dataset_snapshot(dataset_path, version_path):
        """
        Create versioned snapshot of dataset.
        
        Strategy: Copy or link data with metadata
        """
        
        # Create version directory
        Path(version_path).mkdir(parents=True, exist_ok=True)
        
        # Copy dataset
        if Path(dataset_path).is_file():
            shutil.copy2(dataset_path, version_path)
        else:
            shutil.copytree(dataset_path, f"{version_path}/data")
        
        # Create version metadata
        metadata = {
            'version': version_path.split('_v')[-1] if '_v' in version_path else '1.0.0',
            'created_date': datetime.now().isoformat(),
            'hash': DatasetVersioning.calculate_dataset_hash(dataset_path),
            'size_bytes': sum(
                f.stat().st_size for f in Path(dataset_path).rglob('*') if f.is_file()
            ),
            'file_count': len(list(Path(dataset_path).rglob('*')))
        }
        
        # Save metadata
        with open(f"{version_path}/VERSION_METADATA.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    @staticmethod
    def create_version_log(version_log_path, version, author, changes, data_hash):
        """
        Create comprehensive version log entry.
        """
        
        log_entry = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'author': author,
            'changes': changes,
            'data_hash': data_hash,
            'parent_version': None,  # Link to previous version
            'tags': []
        }
        
        # Append to log
        log_file = Path(version_log_path)
        logs = []
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
        
        logs.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        return log_entry

# Example
dataset_path = 'data/train_data.csv'
version_path = 'versions/dataset_v1.0.0'

# Note: Would need actual data for full example
# metadata = DatasetVersioning.create_dataset_snapshot(dataset_path, version_path)
# print("Snapshot created:", metadata)
```

### 2.2 Delta-Based Versioning

```python
class DeltaVersioning:
    """
    Efficient versioning using deltas (changes) instead of full snapshots.
    Works well for tabular data where rows change over time.
    """
    
    @staticmethod
    def create_row_level_changes(df_old, df_new, key_column='id'):
        """
        Calculate row-level changes between versions.
        Returns: added, modified, deleted rows
        """
        
        old_keys = set(df_old[key_column])
        new_keys = set(df_new[key_column])
        
        added_keys = new_keys - old_keys
        deleted_keys = old_keys - new_keys
        modified_keys = old_keys & new_keys
        
        changes = {
            'added': len(added_keys),
            'deleted': len(deleted_keys),
            'modified': 0,
            'added_rows': df_new[df_new[key_column].isin(added_keys)],
            'deleted_rows': df_old[df_old[key_column].isin(deleted_keys)]
        }
        
        # Check modified rows
        for key in modified_keys:
            old_row = df_old[df_old[key_column] == key].iloc[0]
            new_row = df_new[df_new[key_column] == key].iloc[0]
            
            if not old_row.equals(new_row):
                changes['modified'] += 1
        
        return changes
    
    @staticmethod
    def create_column_level_changes(df_old, df_new):
        """Track column-level changes."""
        
        old_cols = set(df_old.columns)
        new_cols = set(df_new.columns)
        
        return {
            'added_columns': list(new_cols - old_cols),
            'removed_columns': list(old_cols - new_cols),
            'column_modifications': {}  # Type changes, etc.
        }
    
    @staticmethod
    def apply_delta(df_base, delta_record):
        """Apply delta changes to recreate version."""
        
        df_result = df_base.copy()
        
        # Remove deleted rows
        if 'deleted_keys' in delta_record:
            df_result = df_result[~df_result['id'].isin(delta_record['deleted_keys'])]
        
        # Add new rows
        if 'added_rows' in delta_record:
            df_result = pd.concat([df_result, delta_record['added_rows']])
        
        return df_result

# Example (conceptual)
import pandas as pd

df_v1 = pd.DataFrame({
    'id': [1, 2, 3],
    'value': [10, 20, 30]
})

df_v2 = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'value': [10, 25, 30, 40]  # Row 2 modified, Row 4 added
})

changes = DeltaVersioning.create_row_level_changes(df_v1, df_v2)
print(f"Changes: {changes['added']} added, {changes['deleted']} deleted, {changes['modified']} modified")
```

## 3. Data Lineage and Provenance

### 3.1 Data Lineage Tracking

```python
class DataLineageTracking:
    """
    Track data lineage (provenance) through processing pipeline.
    Critical for understanding where data comes from and transformations applied.
    """
    
    @staticmethod
    def create_data_lineage_record(
        dataset_name,
        source_type,
        source_location,
        transformations,
        output_location
    ):
        """
        Create comprehensive lineage record.
        
        Standard: W3C PROV Data Model
        """
        
        lineage = {
            'dataset': dataset_name,
            'source': {
                'type': source_type,  # 'api', 'database', 'file', 'generated'
                'location': source_location,
                'access_date': datetime.now().isoformat()
            },
            'transformations': transformations,  # List of processing steps
            'output': {
                'location': output_location,
                'format': 'parquet',  # or csv, json, etc.
                'size_mb': 0  # Will be calculated
            },
            'metadata': {
                'created_by': 'data_pipeline',
                'created_date': datetime.now().isoformat(),
                'last_modified': datetime.now().isoformat()
            }
        }
        
        return lineage
    
    @staticmethod
    def track_transformation_step(
        step_name,
        input_shape,
        output_shape,
        parameters,
        duration_seconds
    ):
        """Record a transformation step in data pipeline."""
        
        step = {
            'name': step_name,
            'timestamp': datetime.now().isoformat(),
            'input': {
                'rows': input_shape[0],
                'columns': input_shape[1]
            },
            'output': {
                'rows': output_shape[0],
                'columns': output_shape[1]
            },
            'parameters': parameters,
            'duration_seconds': duration_seconds,
            'row_change_percent': ((output_shape[0] - input_shape[0]) / input_shape[0] * 100) if input_shape[0] > 0 else 0
        }
        
        return step
    
    @staticmethod
    def create_data_lineage_graph(lineage_records):
        """
        Create visual representation of data lineage.
        Can be visualized as DAG (Directed Acyclic Graph)
        """
        
        # Conceptual: Would use networkx for real implementation
        graph = {
            'nodes': [],  # Data artifacts
            'edges': []   # Transformations
        }
        
        for record in lineage_records:
            graph['nodes'].append({
                'id': record['dataset'],
                'type': 'dataset',
                'location': record['output']['location']
            })
            
            for transform in record['transformations']:
                graph['edges'].append({
                    'from': record['dataset'],
                    'to': transform['name'],
                    'type': 'transformation'
                })
        
        return graph

# Example
lineage = DataLineageTracking.create_data_lineage_record(
    dataset_name='customer_reviews_v2.1.0',
    source_type='api',
    source_location='https://api.example.com/reviews',
    transformations=[
        {
            'name': 'Remove duplicates',
            'params': {'subset': ['text', 'user_id']}
        },
        {
            'name': 'Remove null values',
            'params': {'columns': ['review_text']}
        }
    ],
    output_location='s3://datasets/customer_reviews/v2.1.0/'
)

print("Data Lineage Record:")
print(json.dumps(lineage, indent=2, default=str))
```

## 4. Data Catalog and Metadata Management

### 4.1 Data Catalog System

```python
class DataCatalog:
    """
    Central registry of all datasets with metadata.
    Enables discoverability and governance.
    """
    
    def __init__(self):
        self.catalog = {}
    
    def register_dataset(self, dataset_id, metadata):
        """Register dataset in catalog."""
        
        catalog_entry = {
            'id': dataset_id,
            'registered_date': datetime.now().isoformat(),
            'metadata': metadata,
            'versions': [],
            'access_logs': [],
            'quality_scores': {},
            'tags': metadata.get('tags', []),
            'owner': metadata.get('owner'),
            'description': metadata.get('description')
        }
        
        self.catalog[dataset_id] = catalog_entry
        return catalog_entry
    
    def add_version(self, dataset_id, version_info):
        """Add version to dataset."""
        
        if dataset_id not in self.catalog:
            raise ValueError(f"Dataset {dataset_id} not found in catalog")
        
        self.catalog[dataset_id]['versions'].append({
            'version': version_info['version'],
            'created_date': datetime.now().isoformat(),
            'hash': version_info.get('hash'),
            'size_mb': version_info.get('size_mb'),
            'changes': version_info.get('changes')
        })
    
    def search_datasets(self, query, search_type='tags'):
        """Search catalog for datasets."""
        
        results = []
        
        for dataset_id, entry in self.catalog.items():
            if search_type == 'tags':
                if any(tag.lower() == query.lower() for tag in entry['tags']):
                    results.append(entry)
            elif search_type == 'description':
                if query.lower() in entry['description'].lower():
                    results.append(entry)
            elif search_type == 'owner':
                if entry['owner'].lower() == query.lower():
                    results.append(entry)
        
        return results
    
    def get_dataset_quality_report(self, dataset_id):
        """Get quality metrics for dataset."""
        
        if dataset_id not in self.catalog:
            return None
        
        return self.catalog[dataset_id].get('quality_scores', {})
    
    def export_catalog(self, filepath):
        """Export catalog as JSON."""
        
        with open(filepath, 'w') as f:
            json.dump(self.catalog, f, indent=2, default=str)

# Example
catalog = DataCatalog()

catalog.register_dataset('customer_reviews', {
    'description': 'Customer reviews for e-commerce platform',
    'owner': 'Data Science Team',
    'tags': ['nlp', 'sentiment', 'customer_feedback'],
    'format': 'parquet',
    'schema': {
        'review_id': 'string',
        'review_text': 'string',
        'rating': 'integer',
        'user_id': 'string'
    }
})

# Add versions
catalog.add_version('customer_reviews', {
    'version': '1.0.0',
    'hash': 'abc123',
    'size_mb': 250,
    'changes': 'Initial version'
})

# Search
results = catalog.search_datasets('nlp', search_type='tags')
print(f"Datasets tagged with 'nlp': {len(results)}")
```

## 5. DVC (Data Version Control)

### 5.1 Using DVC for Dataset Versioning

```python
class DVCIntegration:
    """
    Integration with DVC (Data Version Control) for dataset versioning.
    DVC is purpose-built for ML/data versioning.
    """
    
    @staticmethod
    def dvc_initialization_script():
        """
        DVC setup commands to run in terminal.
        """
        
        commands = [
            '# Initialize Git repository',
            'git init',
            '',
            '# Initialize DVC',
            'dvc init',
            '',
            '# Add dataset to DVC',
            'dvc add data/dataset.csv',
            '',
            '# Commit to Git',
            'git add data/dataset.csv.dvc .gitignore',
            'git commit -m "Add dataset v1.0.0"',
            '',
            '# Create Git tag for version',
            'git tag -a v1.0.0 -m "Dataset version 1.0.0"',
            '',
            '# Push to remote storage (S3, GCS, etc.)',
            'dvc remote add -d myremote s3://mybucket/dvc-storage',
            'dvc push'
        ]
        
        return '\n'.join(commands)
    
    @staticmethod
    def dvc_python_api():
        """Use DVC Python API for programmatic access."""
        
        code_example = '''
from dvc.repo import Repo

# Initialize DVC repo
repo = Repo()

# Add file/directory
repo.add("data/dataset.csv")

# Add with metadata
repo.add(
    "data/dataset.csv",
    desc="Customer reviews dataset",
    meta={
        "version": "1.0.0",
        "source": "API",
        "rows": 10000
    }
)

# Get file info
metrics = repo.metrics.show()
print(metrics)

# Checkout different version
repo.checkout("data/dataset.csv.dvc")
'''
        
        return code_example

# Example usage (commands only, not executable)
print(DVCIntegration.dvc_initialization_script())
```

## 6. Best Practices and Governance

### 6.1 Dataset Governance Framework

```python
class DatasetGovernance:
    """
    Framework for dataset governance and compliance.
    """
    
    @staticmethod
    def create_data_governance_policy():
        """
        Create comprehensive data governance policy.
        """
        
        policy = {
            'data_ownership': {
                'responsibility': 'Team lead',
                'accountability': 'Ensure data quality and compliance',
                'documentation': 'Required for all datasets'
            },
            'data_access': {
                'requirements': 'Data governance approval required',
                'audit_logging': 'All access logged',
                'encryption': 'Sensitive data must be encrypted at rest and in transit'
            },
            'data_quality': {
                'minimum_completeness': '95%',
                'maximum_latency': '24 hours',
                'freshness_requirement': 'Daily for prod datasets',
                'validation': 'Automated quality checks required'
            },
            'data_retention': {
                'default_retention': '3 years',
                'sensitive_data': '1 year',
                'audit_logs': '5 years',
                'deletion_process': 'Secure deletion required'
            },
            'versioning': {
                'requirement': 'All datasets must be versioned',
                'metadata': 'Comprehensive metadata required',
                'lineage': 'Data lineage must be tracked',
                'documentation': 'Changes must be documented'
            },
            'compliance': {
                'gdpr': 'Right to deletion implemented',
                'ccpa': 'Consumer privacy rights enforced',
                'hipaa': 'If applicable, encryption and access controls enforced',
                'audit': 'Regular audits required'
            }
        }
        
        return policy

# Example
policy = DatasetGovernance.create_data_governance_policy()
print("Data Governance Policy:")
print(json.dumps(policy, indent=2))
```

## 7. Quality Checklist

### Dataset Versioning Best Practices
- [ ] Choose versioning strategy (snapshot, delta, or hybrid)
- [ ] Implement automatic metadata collection
- [ ] Track data lineage through pipelines
- [ ] Create comprehensive version logs
- [ ] Use semantic versioning (major.minor.patch)
- [ ] Automate version tagging
- [ ] Set up data catalog
- [ ] Implement access logging
- [ ] Document versioning procedures
- [ ] Regular audit of versions

## 8. Authoritative Sources

1. "Data Version Control (DVC) Documentation" - https://dvc.org/doc
2. "PROV Data Model" - W3C Recommendation
3. "Dataset Cards for Datasets" - Gebru et al. (2021)
4. "Data Governance: What You Need to Know" - Enterprise Management Associates
5. "Machine Learning Operations (MLOps) Maturity Model" - Gartner
6. "Reproducible Research in Science" - Nature Editorial (2021)

---

**Citation Format:**
Banerji Seal, S. (2026). "Dataset Versioning and Management: Reproducibility and Governance." LLM-Whisperer Skills Library.

**Version:** 1.0  
**Status:** Production Ready
