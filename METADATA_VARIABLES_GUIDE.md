# Metadata Variables System Guide

## Overview
The upload procedure has been refactored to use **Monaco editor variables as form fields** instead of separate HTML form inputs. This approach:

- ✅ Eliminates duplicate form fields
- ✅ Keeps all metadata in one place (the editor)
- ✅ Allows version control of metadata alongside code
- ✅ Simplifies the UI and form submission
- ✅ Makes deployment info transparent in your scripts

## How It Works

### Step 1: Define Metadata Variables in Your Script

At the top of your editor code, define variables for object metadata:

```python
# Object metadata variables
name = "my_dataset"
description = "What this dataset contains"
dataset_type = "csv"
connection_string = "postgresql://localhost/mydb"

# Your actual data processing code below
import pandas as pd
df = pd.read_csv("data.csv")
# ... process data ...
```

### Step 2: Extract and Display Metadata

Click the **"Extract Variables"** button to parse metadata from your code. The system will:
1. Find all metadata variable assignments in your code
2. Display them in a formatted metadata table
3. Validate required fields (like `name`)

You'll see a panel showing:
| Field | Value |
|-------|-------|
| name | my_dataset |
| description | What this dataset contains |
| dataset_type | csv |

### Step 3: Create the Database Object

Click the **"Create"** button to upload the dataset/model. The form will:
1. Extract metadata variables from your editor
2. Validate that `name` is defined
3. Send the script + metadata to the backend
4. Create the database object with the specified metadata

## Supported Metadata Fields

### Required Fields
- **`name`**: Unique identifier for your object
  - Type: string
  - Example: `name = "customer_data_2024"`

### Optional Fields
- **`description`**: Human-readable description
  - Type: string
  - Example: `description = "Monthly customer transaction records"`

- **`dataset_type`**: Format or type of data
  - Type: string
  - Examples: `"csv"`, `"json"`, `"parquet"`, `"sql"`, `"database"`

- **`connection_string`**: Database connection URL (for SQL sources)
  - Type: string
  - Example: `connection_string = "postgresql://user:password@localhost:5432/mydb"`

- **`model_type`**: Type of ML model (for models)
  - Type: string
  - Examples: `"sklearn"`, `"pytorch"`, `"tensorflow"`, `"xgboost"`

- **`type`**: Generic type field (alternative to dataset_type/model_type)
  - Type: string
  - Example: `type = "classification_model"`

## Complete Example

### Dataset Creation Script

```python
# ============================================================================
# OBJECT METADATA
# ============================================================================
name = "sales_2024"
description = "Complete sales data for 2024 with regional breakdown"
dataset_type = "csv"
connection_string = ""

# ============================================================================
# DATA PROCESSING
# ============================================================================
import pandas as pd
import numpy as np

# Load raw data
df = pd.read_csv("sales_raw.csv")

# Process: clean, filter, aggregate
df['date'] = pd.to_datetime(df['date'])
df = df[df['amount'] > 0]  # Remove invalid entries

# Create summary
summary = {
    "total_records": len(df),
    "date_range": f"{df['date'].min()} to {df['date'].max()}",
    "regions": df['region'].unique().tolist()
}
```

### Model Creation Script

```python
name = "customer_churn_predictor"
description = "XGBoost classifier for predicting customer churn"
model_type = "xgboost"

import xgboost as xgb
import pandas as pd

# Load training data
X = pd.read_csv("features.csv")
y = pd.read_csv("targets.csv")

# Train model
model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1)
model.fit(X, y)

# Performance metrics
accuracy = model.score(X, y)
print(f"Model accuracy: {accuracy:.2%}")
```

## Using the Template System

### Load Template
Click the **"Load Template"** button to populate your editor with a pre-formatted metadata template:

```python
# ============================================================================
# OBJECT METADATA (Define these variables for database object creation)
# ============================================================================

# REQUIRED: Object name (must be unique)
name = "my_dataset"

# OPTIONAL: Description of what this object does
description = "Description of the dataset or model"

# OPTIONAL: Type of dataset/model
dataset_type = "csv"  # or: json, parquet, database, etc.

# OPTIONAL: Database connection string (for SQL-based datasets)
connection_string = ""

# ============================================================================
# YOUR DATA PROCESSING CODE BELOW
# ============================================================================
# ... write your code here ...
```

## Backend Integration

The refactored upload handler sends the following data to the backend:

```json
{
  "operationId": "datasets",
  "objectName": "my_dataset",
  "description": "...",
  "datasetType": "csv",
  "connectionString": "...",
  "script": "# Python code..."
}
```

The backend should extract metadata from the request and store it with the dataset/model object.

## Workflow Summary

```
1. Write code in editor
   ↓
2. Define metadata variables at the top
   ↓
3. Click "Extract Variables"
   ↓
4. Review metadata in the panel
   ↓
5. Click "Create"
   ↓
6. Object created with metadata!
```

## Tips & Tricks

### Tip 1: Use Load Template
Start with the **"Load Template"** button to get the correct structure.

### Tip 2: Auto-Extract on Execute
When you click **"Execute"** button, it runs your script on the backend. The **"Extract Variables"** button parses metadata locally without executing.

### Tip 3: Validate Before Creating
Always click **"Extract Variables"** and review the metadata panel before clicking **"Create"** to catch any formatting issues.

### Tip 4: Comment Variables
Keep descriptive comments in your code:
```python
# Unique name for this dataset (required)
name = "product_sales_q1_2024"

# What analysis was done
description = "Q1 2024 product sales aggregated by region and category"
```

### Tip 5: Multi-line Strings
For longer descriptions, use triple quotes:
```python
description = """
Complete sales dataset for 2024.
Includes:
- Daily transaction records
- Regional aggregations
- Customer demographics
"""
```

## Troubleshooting

### Issue: "Define 'name' variable in editor"
**Solution**: Add a `name` variable at the top of your script:
```python
name = "my_object_name"
```

### Issue: Metadata not showing in panel
**Solution**: 
1. Make sure variables are on separate lines
2. Use proper Python syntax: `variable_name = "value"`
3. Click "Extract Variables" button
4. Check browser console (F12) for any parsing errors

### Issue: Special characters in values
**Solution**: Use quotes around values and escape special characters:
```python
connection_string = "postgresql://user:pass@host:5432/db_name"
description = "Data with \"quotes\" and special chars"
```

## Advanced Usage

### Conditional Metadata
You can compute metadata dynamically:
```python
import datetime

name = "daily_report_" + datetime.date.today().isoformat()
description = f"Auto-generated report for {datetime.date.today()}"
```

### Environment-based Metadata
```python
import os

name = os.environ.get('DATASET_NAME', 'default_dataset')
connection_string = os.environ.get('DB_CONNECTION', '')
```

## Migration from Old System

If you have existing forms with HTML inputs, here's how to migrate:

**Old approach:**
```html
<input id="datasetObjectName" value="my_name" />
<input id="datasetDescription" value="my_description" />
```

**New approach:**
```python
name = "my_name"
description = "my_description"
```

Simply define these variables in your editor code instead of HTML fields!

---

**Enjoy the cleaner, more streamlined workflow!** 🚀
