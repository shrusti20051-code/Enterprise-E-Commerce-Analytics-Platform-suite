# Olist Raw Data Directory (Optional)

This directory can hold the 9 raw CSV files from the Kaggle Olist Brazilian E-Commerce dataset:

1. `olist_orders_dataset.csv`
2. `olist_order_items_dataset.csv`
3. `olist_customers_dataset.csv`
4. `olist_products_dataset.csv`
5. `olist_order_payments_dataset.csv`
6. `olist_order_reviews_dataset.csv`
7. `olist_sellers_dataset.csv`
8. `olist_geolocation_dataset.csv`
9. `product_category_name_translation.csv`

### Zero-Friction Setup:
- **No files? No problem!** If this directory is empty or the CSVs haven't been downloaded yet, the application automatically activates its **Calibrated Olist Benchmark Engine**. It generates a statistically accurate representation mirroring authentic Brazilian e-commerce distributions, seasonal sales, logistics delays, and review scores so the app runs immediately.
- **When you download the raw CSVs**: Simply place them inside this `data/` folder, restart the app, and the data loader will automatically detect and clean the real files.
