# Data Pipeline

## Overview

This module implements a complete data pipeline:

Scrape → Clean → Convert → Store → Query → Compare

The data is collected from books.toscrape.com using Python requests and BeautifulSoup.

## Data Scope

- 100 books scraped
- 29 categories
- Data source: https://books.toscrape.com/

## Cleaning

The scraped data is cleaned into:

- price_gbp: float
- rating: integer from 1 to 5
- in_stock: boolean
- price_inr: float

Unexpected numeric parsing issues are handled using median imputation where required.

## Currency Conversion

The project uses the required fixed baseline:

**1 GBP = 105.50 INR**

This is a project-defined fixed conversion rate and does not use a live currency API.

## Database

SQLite is used with two normalized tables:

- categories
- books

The tables use a primary-key / foreign-key relationship.

## SQL Queries

The project demonstrates:

- WHERE
- ORDER BY
- LIMIT
- DISTINCT
- IN
- BETWEEN
- JOIN

SQL outputs are saved in `sql_query_outputs.txt`.

## Pandas Validation

At least two SQL query results are read into pandas using `pd.read_sql()`.

The JOIN result is independently reproduced using `pd.merge()`.

The SQL JOIN and pandas merge results are equivalent.

## Files

- `data_pipeline.ipynb` — scraping, cleaning, database loading and analysis
- `books.db` — SQLite database
- `sql_query_outputs.txt` — SQL queries and outputs
- `sql_vs_pandas_merge_comparison.csv` — SQL vs pandas comparison
