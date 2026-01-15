# Spider Dataset Knowledge Base

> A comprehensive guide to the Spider text-to-SQL dataset from Yale University.
> This document serves as both human-readable documentation and LLM context.

---

## 1. Dataset Overview

### What is Spider?

Spider is a large-scale, cross-domain text-to-SQL benchmark dataset created by Yale University. It is designed to train and evaluate models that translate natural language questions into SQL queries.

**Citation:**
```
@inproceedings{Yu&al.18c,
  year = 2018,
  title = {Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  booktitle = {EMNLP},
  author = {Tao Yu and Rui Zhang and Kai Yang and Michihiro Yasunaga and Dongxu Wang and Zifan Li and James Ma and Irene Li and Qingning Yao and Shanelle Roman and Zilin Zhang and Dragomir Radev}
}
```

### Key Characteristics

| Property | Description |
|----------|-------------|
| **Task** | Natural Language → SQL |
| **Domains** | Cross-domain (166+ databases spanning restaurants, music, flights, colleges, etc.) |
| **Languages** | English (questions), SQL (targets) |
| **Complexity** | Simple SELECTs to complex JOINs, subqueries, set operations |
| **Evaluation** | Exact match, execution accuracy |
| **Challenge** | Zero-shot generalization to unseen database schemas |

### Why Spider is Important

1. **Cross-domain**: Unlike earlier datasets, Spider tests generalization to completely new database schemas
2. **Complexity**: Includes complex SQL constructs (JOINs, GROUP BY, HAVING, nested queries, INTERSECT/UNION/EXCEPT)
3. **Scale**: 10,000+ examples across 200+ databases
4. **Real schemas**: Uses realistic database schemas with foreign key relationships

---

## 2. File Structure

### Directory Layout

```
spider_data/
├── spider_data/
│   ├── train_spider.json      # Primary training data (7,000 examples, 140 DBs)
│   ├── train_others.json      # Additional training data (1,659 examples, 6 DBs)
│   ├── dev.json               # Development/validation set (1,034 examples, 20 DBs)
│   ├── test.json              # Hidden test set
│   ├── tables.json            # Schema definitions for train/dev databases (166 DBs)
│   ├── test_tables.json       # Schema definitions for test databases
│   ├── train_gold.sql         # Gold SQL queries with DB IDs for training
│   ├── dev_gold.sql           # Gold SQL queries with DB IDs for dev
│   ├── test_gold.sql          # Gold SQL queries with DB IDs for test
│   ├── database/              # 166 SQLite databases for train/dev
│   │   ├── academic/
│   │   │   ├── academic.sqlite
│   │   │   └── schema.sql
│   │   ├── concert_singer/
│   │   │   ├── concert_singer.sqlite
│   │   │   └── schema.sql
│   │   └── ... (166 database folders)
│   ├── test_database/         # ~520 databases for testing
│   └── README.txt             # Original dataset documentation
```

### File Descriptions

| File | Size | Purpose |
|------|------|---------|
| `train_spider.json` | ~25 MB | Primary training set created by Spider authors |
| `train_others.json` | ~8.5 MB | Training examples from prior datasets (Restaurants, GeoQuery, Scholar, Academic, IMDB, Yelp) |
| `dev.json` | ~3.6 MB | Development/validation set for hyperparameter tuning |
| `test.json` | ~7.8 MB | Hidden test set (labels not publicly available) |
| `tables.json` | ~811 KB | Database schema definitions (tables, columns, types, keys) |
| `*_gold.sql` | varies | Plain text SQL with database IDs (format: `SQL\tdb_id`) |

---

## 3. Data Format

### Example Entry (JSON)

Each entry in `train_spider.json`, `train_others.json`, or `dev.json` contains:

```json
{
    "db_id": "department_management",
    "question": "How many heads of the departments are older than 56?",
    "question_toks": ["How", "many", "heads", "of", "the", "departments", "are", "older", "than", "56", "?"],
    "query": "SELECT count(*) FROM head WHERE age > 56",
    "query_toks": ["SELECT", "count", "(", "*", ")", "FROM", "head", "WHERE", "age", ">", "56"],
    "query_toks_no_value": ["select", "count", "(", "*", ")", "from", "head", "where", "age", ">", "value"],
    "sql": {
        "select": [...],
        "from": {...},
        "where": [...],
        "groupBy": [],
        "having": [],
        "orderBy": [],
        "limit": null,
        "intersect": null,
        "union": null,
        "except": null
    }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `db_id` | string | Database identifier (matches folder name in `database/`) |
| `question` | string | Natural language question |
| `question_toks` | list[string] | Tokenized question |
| `query` | string | Gold SQL query |
| `query_toks` | list[string] | Tokenized SQL query |
| `query_toks_no_value` | list[string] | Tokenized SQL with literal values replaced by "value" |
| `sql` | object | Parsed SQL as an abstract syntax tree (AST) |

### Schema Definition Format (tables.json)

Each database schema in `tables.json` contains:

```json
{
    "db_id": "department_management",
    "table_names": ["department", "head", "management"],
    "table_names_original": ["department", "head", "management"],
    "column_names": [
        [-1, "*"],
        [0, "department id"],
        [0, "name"],
        [0, "creation"],
        [1, "head id"],
        [1, "name"],
        [1, "born state"],
        [2, "department id"],
        [2, "head id"]
    ],
    "column_names_original": [
        [-1, "*"],
        [0, "Department_ID"],
        [0, "Name"],
        ...
    ],
    "column_types": ["text", "number", "text", "text", "number", "text", "text", "number", "number"],
    "primary_keys": [1, 5],
    "foreign_keys": [[7, 1], [8, 5]]
}
```

### Schema Field Descriptions

| Field | Description |
|-------|-------------|
| `db_id` | Database identifier |
| `table_names` | Human-readable table names (lowercased, spaces) |
| `table_names_original` | Original table names as in the database |
| `column_names` | List of [table_index, column_name] pairs. Index -1 means "*" wildcard |
| `column_names_original` | Original column names as in the database |
| `column_types` | Data type for each column ("text", "number", etc.) |
| `primary_keys` | Column indices that are primary keys |
| `foreign_keys` | Pairs of [column_index, referenced_column_index] |

### Gold SQL Format (*_gold.sql)

Plain text file with one example per line:

```
SELECT count(*) FROM head WHERE age > 56	department_management
SELECT name, born_state, age FROM head ORDER BY age	department_management
...
```

Format: `SQL_QUERY<tab>database_id`

---

## 4. Dataset Statistics

### Size Summary

| Split | Examples | Databases | Source |
|-------|----------|-----------|--------|
| train_spider | 7,000 | 140 | Original Spider |
| train_others | 1,659 | 6 | Prior datasets (Restaurants, GeoQuery, etc.) |
| dev | 1,034 | 20 | Original Spider |
| test | ~2,147 | ~40 | Original Spider (hidden) |
| **Total** | **~11,840** | **200+** | — |

**Note:** Official training set = train_spider + train_others (8,659 examples)

### Schema Size Statistics

Based on analysis of 166 database schemas:

| Metric | Mean | Median | Min | Max | Std Dev |
|--------|------|--------|-----|-----|---------|
| Tables per database | 5.28 | 4.0 | 2 | 26 | 3.87 |
| Columns per database | 27.13 | 19.5 | 6 | 352 | 30.80 |
| Columns per table | 5.17 | 4.94 | — | — | — |

**Table Count Distribution:**
- 2 tables: 23 databases (13.9%)
- 3 tables: 55 databases (33.1%)
- 4 tables: 26 databases (15.7%)
- 5-6 tables: 18 databases (10.8%)
- 7+ tables: 44 databases (26.5%)

### SQL Length Statistics

Based on analysis of 8,659 training examples:

**Character Length:**
| Metric | Value |
|--------|-------|
| Mean | 122.9 |
| Median | 104 |
| Min | 18 |
| Max | 577 |
| 25th percentile | 64 |
| 75th percentile | 163 |
| 90th percentile | 228 |

**Token Length:**
| Metric | Value |
|--------|-------|
| Mean | 21.0 |
| Median | 18 |
| Min | 4 |
| Max | 90 |

**Token Count Distribution:**
| Range | Count | Percentage |
|-------|-------|------------|
| 1-10 tokens | 1,756 | 20.3% |
| 11-20 tokens | 3,148 | 36.4% |
| 21-30 tokens | 2,157 | 24.9% |
| 31-50 tokens | 1,298 | 15.0% |
| 51-75 tokens | 296 | 3.4% |
| 76-100 tokens | 4 | 0.05% |

### JOIN Frequency Statistics

Based on analysis of 9,693 examples (train + dev):

| Metric | Value |
|--------|-------|
| Queries with JOINs | 4,220 (43.5%) |
| Mean JOINs per query | 0.69 |
| Max JOINs in a query | 8 |
| Total JOINs | 6,660 |

**Join Count Distribution:**
| JOINs | Count | Percentage |
|-------|-------|------------|
| 0 | 5,473 | 56.5% |
| 1 | 2,553 | 26.3% |
| 2 | 1,114 | 11.5% |
| 3 | 369 | 3.8% |
| 4+ | 184 | 1.9% |

**Join Types:**
- All JOINs in Spider use simple/implicit `JOIN` syntax (equivalent to INNER JOIN)
- No LEFT JOIN, RIGHT JOIN, or OUTER JOIN variants

**Set Operations:**
| Operation | Count | Percentage |
|-----------|-------|------------|
| INTERSECT | 290 | 3.0% |
| EXCEPT | 240 | 2.5% |
| UNION | 78 | 0.8% |

**Subqueries:** 1,417 queries (14.6%) contain subqueries

---

## 5. SQL Complexity Levels

Spider categorizes queries into four difficulty levels:

| Level | Description | Examples |
|-------|-------------|----------|
| **Easy** | Single table, no aggregation | `SELECT name FROM singer` |
| **Medium** | JOINs, GROUP BY, simple aggregation | `SELECT count(*) FROM singer GROUP BY country` |
| **Hard** | Multiple JOINs, nested aggregation, HAVING | `SELECT T1.name FROM artist T1 JOIN album T2 ON T1.id = T2.artist_id GROUP BY T1.id HAVING count(*) > 3` |
| **Extra Hard** | Subqueries, set operations (INTERSECT/UNION/EXCEPT), complex nesting | Queries with INTERSECT, deeply nested subqueries |

### SQL Components Coverage

| Component | Present in Dataset |
|-----------|-------------------|
| SELECT (columns) | ✅ All queries |
| SELECT DISTINCT | ✅ Common |
| COUNT/SUM/AVG/MIN/MAX | ✅ Common |
| FROM (single table) | ✅ 56.5% of queries |
| JOIN (1+) | ✅ 43.5% of queries |
| WHERE | ✅ Very common |
| GROUP BY | ✅ Common |
| HAVING | ✅ Present |
| ORDER BY | ✅ Common |
| LIMIT | ✅ Common |
| INTERSECT | ✅ 3.0% |
| UNION | ✅ 0.8% |
| EXCEPT | ✅ 2.5% |
| Subqueries | ✅ 14.6% |
| LIKE | ✅ Present |
| BETWEEN | ✅ Present |
| IN | ✅ Present |
| NOT IN | ✅ Present |
| EXISTS | ❌ Rare/absent |
| CASE WHEN | ❌ Rare/absent |
| Window functions | ❌ Not present |

---

## 6. Database Domains

Spider covers diverse domains to test cross-domain generalization:

### Sample Domains

| Domain | Example DB | Description |
|--------|------------|-------------|
| Music | concert_singer, music_1 | Singers, concerts, albums |
| Academic | college_2, student_1 | Universities, courses, students |
| Government | department_management | Departments, heads, management |
| Transportation | flight_1, bike_1 | Flights, bike sharing |
| Sports | soccer_1, basketball | Teams, players, matches |
| Business | store_1, customers_card_transactions | Retail, transactions |
| Healthcare | hospital_1, allergy_1 | Patients, allergies |
| Geography | geo, world_1 | Cities, countries, rivers |
| Entertainment | tvshow, movie_1 | TV shows, movies |
| Social | twitter_1, epinions_1 | Social networks |

### Domain Characteristics

- **Train/Dev separation**: Databases in training and development sets are completely different
- **Real-world schemas**: Foreign keys, realistic naming, normalized structures
- **Varying complexity**: From 2-table toy schemas to 26-table enterprise schemas

---

## 7. Input/Output Format for Fine-Tuning

### Recommended Input Format

For fine-tuning an LLM on text-to-SQL, construct inputs as:

```
### Database Schema:
Table: department
Columns: department_id (number, primary key), name (text), creation (text), ranking (number), budget_in_billions (number), num_employees (number)

Table: head
Columns: head_id (number, primary key), name (text), born_state (text), age (number)

Table: management
Columns: department_id (number, foreign key -> department.department_id), head_id (number, foreign key -> head.head_id), temporary_acting (text)

### Question:
How many heads of the departments are older than 56?

### SQL:
```

### Expected Output Format

```sql
SELECT count(*) FROM head WHERE age > 56
```

### Alternative Formats

**Format 1: Compact Schema**
```
Schema: department(department_id*, name, creation, ranking, budget_in_billions, num_employees), head(head_id*, name, born_state, age), management(department_id→department, head_id→head, temporary_acting)
Question: How many heads of the departments are older than 56?
SQL:
```

**Format 2: CREATE TABLE Style**
```
CREATE TABLE department (department_id INT PRIMARY KEY, name TEXT, ...);
CREATE TABLE head (head_id INT PRIMARY KEY, name TEXT, born_state TEXT, age REAL);
CREATE TABLE management (department_id INT REFERENCES department, head_id INT REFERENCES head, ...);

Question: How many heads of the departments are older than 56?
SQL:
```

---

## 8. Evaluation Metrics

### Exact Match (EM)

Compares predicted SQL to gold SQL after normalization:
- Lowercase conversion
- Whitespace normalization
- Component-wise comparison (SELECT, FROM, WHERE, etc.)

### Execution Accuracy

Executes both predicted and gold SQL on the database:
- Compares result sets
- More lenient than exact match (allows semantically equivalent queries)

### Component-wise F1

Evaluates individual SQL components:
- SELECT columns
- FROM tables
- WHERE conditions
- GROUP BY columns
- etc.

---

## 9. Known Issues and Considerations

### Data Quality Notes

1. **Annotation errors**: The dataset documentation notes ~40 annotation corrections in dev.json (as of 06/2020)
2. **Schema mismatches**: Some column name inconsistencies were corrected in tables.json (08/2020)
3. **Value variations**: Same semantic query may have different literal values

### Preprocessing Considerations

1. **Schema serialization**: Choose a consistent format for representing schemas
2. **Value anonymization**: `query_toks_no_value` replaces literals with "value" for schema-agnostic training
3. **Case sensitivity**: SQL keywords and identifiers may have varying cases
4. **Alias handling**: Table aliases (T1, T2) are common; models must learn to use them

### Train/Test Leakage Prevention

- **No database overlap**: Training and test databases are completely separate
- **No question overlap**: Questions are unique per database
- **Schema similarity**: Some schemas may be structurally similar across domains

---

## 10. Related Datasets

| Dataset | Comparison to Spider |
|---------|---------------------|
| WikiSQL | Single-table only, simpler SQL |
| SParC | Multi-turn conversational extension of Spider |
| CoSQL | Dialogue-based SQL with clarification |
| Spider-Realistic | Spider with more natural, ambiguous questions |
| Spider-Syn | Spider with synthetic question perturbations |
| Spider-DK | Spider requiring domain knowledge |
| BIRD | Larger scale, more complex schemas, with external knowledge |

---

## 11. Quick Reference

### Loading the Data (Python)

```python
import json

# Load training examples
with open("spider_data/spider_data/train_spider.json") as f:
    train_spider = json.load(f)

with open("spider_data/spider_data/train_others.json") as f:
    train_others = json.load(f)

# Combine for full training set
train_data = train_spider + train_others  # 8,659 examples

# Load schemas
with open("spider_data/spider_data/tables.json") as f:
    schemas = {db["db_id"]: db for db in json.load(f)}

# Access example
example = train_data[0]
print(f"Question: {example['question']}")
print(f"SQL: {example['query']}")
print(f"Database: {example['db_id']}")

# Get schema for this example
schema = schemas[example['db_id']]
```

### Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Training examples | 8,659 |
| Dev examples | 1,034 |
| Total databases | 166 (train/dev) |
| Avg tables per DB | 5.28 |
| Avg columns per DB | 27.13 |
| Avg SQL length (tokens) | 21 |
| Queries with JOINs | 43.5% |
| Queries with subqueries | 14.6% |

---

## 12. Changelog

| Date | Update |
|------|--------|
| 2018-10 | Initial Spider release (EMNLP 2018) |
| 2020-06 | ~40 annotation errors corrected in dev.json |
| 2020-08 | Column name mismatches fixed in tables.json |

---

*Last updated: December 2024*
*Generated from dataset analysis on Spider 1.0*
