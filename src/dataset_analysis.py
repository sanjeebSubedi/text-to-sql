"""
Spider Dataset Analysis Script
Computes:
- Average schema size (tables and columns per database)
- SQL length distribution
- Join frequency
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import statistics

# Paths
DATA_DIR = Path(__file__).parent.parent / "spider_data" / "spider_data"
TABLES_JSON = DATA_DIR / "tables.json"
TRAIN_SPIDER_JSON = DATA_DIR / "train_spider.json"
TRAIN_OTHERS_JSON = DATA_DIR / "train_others.json"
DEV_JSON = DATA_DIR / "dev.json"


def load_json(filepath: Path) -> list:
    """Load a JSON file and return its contents."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_schema_sizes(tables_data: list) -> dict:
    """
    Analyze schema sizes across all databases.
    Returns statistics about tables and columns per database.
    """
    table_counts = []
    column_counts = []
    columns_per_table = []
    
    for db in tables_data:
        num_tables = len(db["table_names"])
        # Exclude the "*" column at index 0
        num_columns = len([c for c in db["column_names"] if c[0] != -1])
        
        table_counts.append(num_tables)
        column_counts.append(num_columns)
        
        if num_tables > 0:
            columns_per_table.append(num_columns / num_tables)
    
    return {
        "total_databases": len(tables_data),
        "tables": {
            "mean": statistics.mean(table_counts),
            "median": statistics.median(table_counts),
            "min": min(table_counts),
            "max": max(table_counts),
            "stdev": statistics.stdev(table_counts) if len(table_counts) > 1 else 0,
        },
        "columns": {
            "mean": statistics.mean(column_counts),
            "median": statistics.median(column_counts),
            "min": min(column_counts),
            "max": max(column_counts),
            "stdev": statistics.stdev(column_counts) if len(column_counts) > 1 else 0,
        },
        "columns_per_table": {
            "mean": statistics.mean(columns_per_table),
            "median": statistics.median(columns_per_table),
        },
        "distribution": {
            "table_counts": Counter(table_counts),
            "column_count_bins": bin_distribution(column_counts),
        }
    }


def bin_distribution(values: list, bins: list = None) -> dict:
    """Bin values into ranges for distribution analysis."""
    if bins is None:
        bins = [0, 5, 10, 15, 20, 30, 50, 100]
    
    result = defaultdict(int)
    for v in values:
        for i, upper in enumerate(bins[1:], 1):
            if v <= upper:
                result[f"{bins[i-1]+1}-{upper}"] = result.get(f"{bins[i-1]+1}-{upper}", 0) + 1
                break
        else:
            result[f">{bins[-1]}"] = result.get(f">{bins[-1]}", 0) + 1
    
    return dict(result)


def analyze_sql_lengths(examples: list) -> dict:
    """
    Analyze SQL query lengths.
    Returns distribution of query lengths (in tokens and characters).
    """
    char_lengths = []
    token_lengths = []
    word_lengths = []
    
    for ex in examples:
        query = ex["query"]
        tokens = ex["query_toks"]
        
        char_lengths.append(len(query))
        token_lengths.append(len(tokens))
        # Word count (splitting on whitespace)
        word_lengths.append(len(query.split()))
    
    return {
        "total_queries": len(examples),
        "character_length": {
            "mean": statistics.mean(char_lengths),
            "median": statistics.median(char_lengths),
            "min": min(char_lengths),
            "max": max(char_lengths),
            "stdev": statistics.stdev(char_lengths) if len(char_lengths) > 1 else 0,
            "percentiles": {
                "25th": sorted(char_lengths)[len(char_lengths) // 4],
                "75th": sorted(char_lengths)[3 * len(char_lengths) // 4],
                "90th": sorted(char_lengths)[9 * len(char_lengths) // 10],
            }
        },
        "token_length": {
            "mean": statistics.mean(token_lengths),
            "median": statistics.median(token_lengths),
            "min": min(token_lengths),
            "max": max(token_lengths),
            "stdev": statistics.stdev(token_lengths) if len(token_lengths) > 1 else 0,
        },
        "word_length": {
            "mean": statistics.mean(word_lengths),
            "median": statistics.median(word_lengths),
            "min": min(word_lengths),
            "max": max(word_lengths),
        },
        "distribution": {
            "token_bins": bin_sql_lengths(token_lengths),
            "char_bins": bin_sql_lengths(char_lengths, bins=[0, 50, 100, 150, 200, 300, 500, 1000]),
        }
    }


def bin_sql_lengths(lengths: list, bins: list = None) -> dict:
    """Bin SQL lengths into ranges."""
    if bins is None:
        bins = [0, 10, 20, 30, 50, 75, 100, 150]
    
    result = {}
    for i, upper in enumerate(bins[1:], 1):
        lower = bins[i-1] + 1 if i > 1 else 1
        count = sum(1 for l in lengths if lower <= l <= upper)
        result[f"{lower}-{upper}"] = count
    
    count_above = sum(1 for l in lengths if l > bins[-1])
    if count_above > 0:
        result[f">{bins[-1]}"] = count_above
    
    return result


def analyze_join_frequency(examples: list) -> dict:
    """
    Analyze JOIN usage in SQL queries.
    Returns frequency of different join types and counts.
    """
    join_pattern = re.compile(r'\bJOIN\b', re.IGNORECASE)
    left_join_pattern = re.compile(r'\bLEFT\s+JOIN\b', re.IGNORECASE)
    right_join_pattern = re.compile(r'\bRIGHT\s+JOIN\b', re.IGNORECASE)
    inner_join_pattern = re.compile(r'\bINNER\s+JOIN\b', re.IGNORECASE)
    outer_join_pattern = re.compile(r'\b(LEFT|RIGHT|FULL)\s+OUTER\s+JOIN\b', re.IGNORECASE)
    
    join_counts = []  # Number of JOINs per query
    queries_with_joins = 0
    join_type_counts = defaultdict(int)
    
    # Also track set operations
    intersect_count = 0
    union_count = 0
    except_count = 0
    subquery_count = 0
    
    for ex in examples:
        query = ex["query"]
        
        # Count JOINs
        joins = len(join_pattern.findall(query))
        join_counts.append(joins)
        
        if joins > 0:
            queries_with_joins += 1
        
        # Categorize join types
        left_joins = len(left_join_pattern.findall(query))
        right_joins = len(right_join_pattern.findall(query))
        inner_joins = len(inner_join_pattern.findall(query))
        outer_joins = len(outer_join_pattern.findall(query))
        
        # Simple JOINs (not explicitly LEFT, RIGHT, INNER, or OUTER)
        simple_joins = joins - left_joins - right_joins - inner_joins - outer_joins
        
        join_type_counts["simple_join"] += simple_joins
        join_type_counts["left_join"] += left_joins
        join_type_counts["right_join"] += right_joins
        join_type_counts["inner_join"] += inner_joins
        join_type_counts["outer_join"] += outer_joins
        
        # Set operations
        if re.search(r'\bINTERSECT\b', query, re.IGNORECASE):
            intersect_count += 1
        if re.search(r'\bUNION\b', query, re.IGNORECASE):
            union_count += 1
        if re.search(r'\bEXCEPT\b', query, re.IGNORECASE):
            except_count += 1
        
        # Subqueries (simplified detection)
        if query.count('(') > query.upper().count('COUNT(') + query.upper().count('SUM(') + \
           query.upper().count('AVG(') + query.upper().count('MAX(') + query.upper().count('MIN('):
            subquery_count += 1
    
    # Distribution of join counts
    join_distribution = Counter(join_counts)
    
    return {
        "total_queries": len(examples),
        "queries_with_joins": queries_with_joins,
        "join_percentage": (queries_with_joins / len(examples)) * 100 if examples else 0,
        "join_statistics": {
            "mean_joins_per_query": statistics.mean(join_counts),
            "max_joins_in_query": max(join_counts),
            "total_joins": sum(join_counts),
        },
        "join_type_breakdown": dict(join_type_counts),
        "join_count_distribution": {
            "0_joins": join_distribution.get(0, 0),
            "1_join": join_distribution.get(1, 0),
            "2_joins": join_distribution.get(2, 0),
            "3_joins": join_distribution.get(3, 0),
            "4+_joins": sum(v for k, v in join_distribution.items() if k >= 4),
        },
        "set_operations": {
            "intersect": intersect_count,
            "union": union_count,
            "except": except_count,
        },
        "queries_with_subqueries": subquery_count,
    }


def print_section(title: str, width: int = 60):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_stats(stats: dict, indent: int = 0):
    """Recursively print statistics dictionary."""
    prefix = "  " * indent
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_stats(value, indent + 1)
        elif isinstance(value, float):
            print(f"{prefix}{key}: {value:.2f}")
        else:
            print(f"{prefix}{key}: {value}")


def main():
    print("Loading Spider dataset...")
    
    # Load data
    tables_data = load_json(TABLES_JSON)
    train_spider = load_json(TRAIN_SPIDER_JSON)
    train_others = load_json(TRAIN_OTHERS_JSON)
    dev_data = load_json(DEV_JSON)
    
    # Combine training data
    all_train = train_spider + train_others
    all_data = all_train + dev_data
    
    print(f"Loaded {len(tables_data)} database schemas")
    print(f"Loaded {len(train_spider)} train_spider examples")
    print(f"Loaded {len(train_others)} train_others examples")
    print(f"Loaded {len(dev_data)} dev examples")
    print(f"Total examples: {len(all_data)}")
    
    # ========================================
    # 1. Schema Size Analysis
    # ========================================
    print_section("1. AVERAGE SCHEMA SIZE")
    schema_stats = analyze_schema_sizes(tables_data)
    
    print(f"\nTotal Databases: {schema_stats['total_databases']}")
    print("\nTables per Database:")
    print(f"  Mean:   {schema_stats['tables']['mean']:.2f}")
    print(f"  Median: {schema_stats['tables']['median']:.1f}")
    print(f"  Min:    {schema_stats['tables']['min']}")
    print(f"  Max:    {schema_stats['tables']['max']}")
    print(f"  Stdev:  {schema_stats['tables']['stdev']:.2f}")
    
    print("\nColumns per Database:")
    print(f"  Mean:   {schema_stats['columns']['mean']:.2f}")
    print(f"  Median: {schema_stats['columns']['median']:.1f}")
    print(f"  Min:    {schema_stats['columns']['min']}")
    print(f"  Max:    {schema_stats['columns']['max']}")
    print(f"  Stdev:  {schema_stats['columns']['stdev']:.2f}")
    
    print("\nAverage Columns per Table:")
    print(f"  Mean:   {schema_stats['columns_per_table']['mean']:.2f}")
    print(f"  Median: {schema_stats['columns_per_table']['median']:.2f}")
    
    print("\nTable Count Distribution:")
    for count, freq in sorted(schema_stats['distribution']['table_counts'].items()):
        print(f"  {count} tables: {freq} databases")
    
    # ========================================
    # 2. SQL Length Distribution
    # ========================================
    print_section("2. SQL LENGTH DISTRIBUTION")
    
    print("\n--- Training Data (train_spider + train_others) ---")
    train_sql_stats = analyze_sql_lengths(all_train)
    
    print(f"\nTotal Queries: {train_sql_stats['total_queries']}")
    print("\nCharacter Length:")
    print(f"  Mean:   {train_sql_stats['character_length']['mean']:.1f}")
    print(f"  Median: {train_sql_stats['character_length']['median']:.1f}")
    print(f"  Min:    {train_sql_stats['character_length']['min']}")
    print(f"  Max:    {train_sql_stats['character_length']['max']}")
    print(f"  25th percentile: {train_sql_stats['character_length']['percentiles']['25th']}")
    print(f"  75th percentile: {train_sql_stats['character_length']['percentiles']['75th']}")
    print(f"  90th percentile: {train_sql_stats['character_length']['percentiles']['90th']}")
    
    print("\nToken Length:")
    print(f"  Mean:   {train_sql_stats['token_length']['mean']:.1f}")
    print(f"  Median: {train_sql_stats['token_length']['median']:.1f}")
    print(f"  Min:    {train_sql_stats['token_length']['min']}")
    print(f"  Max:    {train_sql_stats['token_length']['max']}")
    
    print("\nToken Count Distribution:")
    for bin_range, count in train_sql_stats['distribution']['token_bins'].items():
        pct = (count / train_sql_stats['total_queries']) * 100
        bar = "█" * int(pct / 2)
        print(f"  {bin_range:>10} tokens: {count:>5} ({pct:>5.1f}%) {bar}")
    
    # ========================================
    # 3. Join Frequency Analysis
    # ========================================
    print_section("3. JOIN FREQUENCY ANALYSIS")
    
    print("\n--- All Data (Train + Dev) ---")
    join_stats = analyze_join_frequency(all_data)
    
    print(f"\nTotal Queries: {join_stats['total_queries']}")
    print(f"Queries with JOINs: {join_stats['queries_with_joins']} ({join_stats['join_percentage']:.1f}%)")
    
    print("\nJoin Statistics:")
    print(f"  Mean JOINs per query: {join_stats['join_statistics']['mean_joins_per_query']:.2f}")
    print(f"  Max JOINs in a query: {join_stats['join_statistics']['max_joins_in_query']}")
    print(f"  Total JOINs: {join_stats['join_statistics']['total_joins']}")
    
    print("\nJoin Type Breakdown:")
    total_joins = sum(join_stats['join_type_breakdown'].values())
    for join_type, count in sorted(join_stats['join_type_breakdown'].items(), key=lambda x: -x[1]):
        pct = (count / total_joins * 100) if total_joins > 0 else 0
        print(f"  {join_type:>12}: {count:>5} ({pct:>5.1f}%)")
    
    print("\nJoin Count Distribution:")
    for label, count in join_stats['join_count_distribution'].items():
        pct = (count / join_stats['total_queries']) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:>10}: {count:>5} ({pct:>5.1f}%) {bar}")
    
    print("\nSet Operations:")
    print(f"  INTERSECT: {join_stats['set_operations']['intersect']}")
    print(f"  UNION:     {join_stats['set_operations']['union']}")
    print(f"  EXCEPT:    {join_stats['set_operations']['except']}")
    
    print(f"\nQueries with Subqueries: {join_stats['queries_with_subqueries']}")
    
    print("\n" + "=" * 60)
    print(" Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
