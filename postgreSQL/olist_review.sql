drop TABLE IF EXISTS olist_order_reviews;
CREATE TABLE olist_order_reviews (
    review_id VARCHAR(50) PRIMARY KEY,
    order_id VARCHAR(50),
    review_score INT,
    review_comment_title VARCHAR(255),
    review_comment_message TEXT,
    review_creation_date TIMESTAMP
);

--What is the total number of reviews

SELECT COUNT(*) AS total_reviews
FROM olist_order_reviews;

--Are customers generally happy or unhappy?

SELECT
    review_score,
    COUNT(*) AS total_reviews
FROM olist_order_reviews
GROUP BY review_score
ORDER BY review_score DESC;

--What percentage of customers gave the highest rating?

SELECT
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE review_score = 5)
        / COUNT(*),
        2
    ) AS five_star_percentage
FROM olist_order_reviews;

--How large is the unhappy customer group?

SELECT
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE review_score = 1)
        / COUNT(*),
        2
    ) AS one_star_percentage
FROM olist_order_reviews;

--What is Olist's overall customer satisfaction level?

SELECT
    ROUND(AVG(review_score), 2) AS average_review_score
FROM olist_order_reviews;

--How many customers are unhappy

SELECT
    CASE
        WHEN review_score IN (1, 2) THEN 'Unhappy'
        WHEN review_score = 3 THEN 'Neutral'
        WHEN review_score IN (4, 5) THEN 'Happy'
    END AS customer_sentiment,
    COUNT(*) AS total_reviews
FROM olist_order_reviews
GROUP BY customer_sentiment
ORDER BY total_reviews DESC;

--What percentage of customers are unhappy?

SELECT
    ROUND(
        100.0 *
        COUNT(*) FILTER (WHERE review_score IN (1, 2))
        / COUNT(*),
        2
    ) AS unhappy_percentage
FROM olist_order_reviews;

--How does customer satisfaction change over time?

SELECT
    DATE_TRUNC('month', review_creation_date) AS review_month,
    ROUND(AVG(review_score), 2) AS average_rating
FROM olist_order_reviews
GROUP BY review_month
ORDER BY review_month;

--How many reviews are received each month?

SELECT
    DATE_TRUNC('month', review_creation_date) AS review_month,
    COUNT(*) AS total_reviews
FROM olist_order_reviews
GROUP BY review_month
ORDER BY review_month;

--How often do customers give written feedback?

SELECT
    COUNT(*) AS reviews_with_comments
FROM olist_order_reviews
WHERE review_comment_message IS NOT NULL
  AND TRIM(review_comment_message) <> '';

--What percentage of reviews contain comments?
SELECT
    ROUND(
        100.0 *
        COUNT(*) FILTER (
            WHERE review_comment_message IS NOT NULL
              AND TRIM(review_comment_message) <> ''
        )
        / COUNT(*),
        2
    ) AS comment_percentage
FROM olist_order_reviews;

--How many reviews have a missing score?

SELECT COUNT(*) AS missing_review_scores
FROM olist_order_reviews
WHERE review_score IS NULL;

--Are there duplicate review IDs?
SELECT
    review_id,
    COUNT(*) AS occurrences
FROM olist_order_reviews
GROUP BY review_id
HAVING COUNT(*) > 1
ORDER BY occurrences DESC;

--Do customers with low ratings tend to explain their problems more often?

SELECT
    review_score,
    COUNT(*) FILTER (
        WHERE review_comment_message IS NOT NULL
          AND TRIM(review_comment_message) <> ''
    ) AS reviews_with_comments
FROM olist_order_reviews
GROUP BY review_score
ORDER BY review_score DESC;

--Do 1-star customers leave more comments?

SELECT
    review_score,
    COUNT(*) AS total_reviews,
    COUNT(*) FILTER (
        WHERE review_comment_message IS NOT NULL
          AND TRIM(review_comment_message) <> ''
    ) AS reviews_with_comments
FROM olist_order_reviews
GROUP BY review_score
ORDER BY review_score;

--Find the most recent reviews

SELECT
    review_id,
    order_id,
    review_score,
    review_comment_message,
    review_creation_date
FROM olist_order_reviews
ORDER BY review_creation_date DESC
LIMIT 10;

--Find the latest unhappy reviews

SELECT
    review_id,
    order_id,
    review_score,
    review_comment_message,
    review_creation_date
FROM olist_order_reviews
WHERE review_score IN (1, 2)
ORDER BY review_creation_date DESC
LIMIT 20;

--Find the longest customer comments

SELECT
    review_id,
    review_score,
    LENGTH(review_comment_message) AS comment_length,
    review_comment_message
FROM olist_order_reviews
WHERE review_comment_message IS NOT NULL
ORDER BY comment_length DESC
LIMIT 10;

--Compare happy vs unhappy reviews

SELECT
    CASE
        WHEN review_score >= 4 THEN 'Positive'
        WHEN review_score = 3 THEN 'Neutral'
        ELSE 'Negative'
    END AS review_category,
    COUNT(*) AS total_reviews,
    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS percentage
FROM olist_order_reviews
GROUP BY review_category
ORDER BY total_reviews DESC;

--Does delivery affect customer satisfaction

SELECT
    CASE
        WHEN o.order_delivered_customer_date >
             o.order_estimated_delivery_date
        THEN 'Late'
        ELSE 'On Time'
    END AS delivery_status,

    COUNT(r.review_id) AS total_reviews,

    ROUND(AVG(r.review_score), 2) AS average_review_score

FROM olist_order_reviews r

JOIN olist_orders o
    ON r.order_id = o.order_id

WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL

GROUP BY delivery_status;

--Our Review Analysis Journey


--Review Count
     ↓
--Rating Distribution
     ↓
--Average Rating
     ↓
--Positive / Neutral / Negative
     ↓
--Unhappy Customer %
     ↓
--Comments Analysis
     ↓
--Monthly Satisfaction
     ↓
--Data Quality
     ↓
--Delivery vs Rating ⭐
     ↓
--Product Category vs Rating ⭐