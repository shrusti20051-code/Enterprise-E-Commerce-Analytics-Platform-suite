CREATE TABLE olist_customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    customer_unique_id VARCHAR(50),
    customer_zip_code_prefix INT,
    customer_city VARCHAR(100),
    customer_state VARCHAR(10)
);

SELECT *
FROM olist_customers
LIMIT 5;

--Check how many customers we have
SELECT COUNT(*) AS total_customer_records
FROM olist_customers;

--Which Brazilian states have the most Olist customers
SELECT
    customer_state,
    COUNT(*) AS total_customers
FROM olist_customers
GROUP BY customer_state
ORDER BY total_customers DESC;

--Find the number of unique customers
SELECT
    COUNT(DISTINCT customer_unique_id) AS unique_customers
FROM olist_customers;

--Find the cities with the most customers
SELECT
    customer_city,
    customer_state,
    COUNT(*) AS total_customers
FROM olist_customers
GROUP BY customer_city, customer_state
ORDER BY total_customers DESC
LIMIT 10;

--Where is Olist's customer base concentrated

SELECT
    customer_state,
    COUNT(*) AS total_customers
FROM olist_customers
GROUP BY customer_state
ORDER BY total_customers DESC;

--Which cities have the largest customer base

SELECT
    customer_city,
    customer_state,
    COUNT(*) AS total_customers
FROM olist_customers
GROUP BY customer_city, customer_state
ORDER BY total_customers DESC
LIMIT 10;

--How many unique customers does Olist have

SELECT
    COUNT(DISTINCT customer_unique_id) AS unique_customers
FROM olist_customers;

--Are there customers with multiple customer records

SELECT
    customer_unique_id,
    COUNT(*) AS records
FROM olist_customers
GROUP BY customer_unique_id
HAVING COUNT(*) > 1
ORDER BY records DESC;

--How many customer records are duplicated

SELECT
    COUNT(*) AS duplicate_customer_records
FROM (
    SELECT customer_unique_id
    FROM olist_customers
    GROUP BY customer_unique_id
    HAVING COUNT(*) > 1
) AS duplicates;

--Which states have the fewest customers

SELECT
    customer_state,
    COUNT(*) AS total_customers
FROM olist_customers
GROUP BY customer_state
ORDER BY total_customers ASC;

--How many cities does Olist have customers in

SELECT
    COUNT(DISTINCT customer_city) AS total_cities
FROM olist_customers;

--How many cities are represented in each state

SELECT
    customer_state,
    COUNT(DISTINCT customer_city) AS total_cities
FROM olist_customers
GROUP BY customer_state
ORDER BY total_cities DESC;

--Which state has the highest number of different cities

SELECT
    customer_state,
    COUNT(DISTINCT customer_city) AS number_of_cities
FROM olist_customers
GROUP BY customer_state
ORDER BY number_of_cities DESC
LIMIT 10;

--Find customers from a particular state

SELECT *
FROM olist_customers
WHERE customer_state = 'SP';