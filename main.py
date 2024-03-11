# Import all libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


brands = pd.read_csv("brands.csv")
finance = pd.read_csv("finance.csv")
info = pd.read_csv("info.csv")
reviews = pd.read_csv("reviews.csv")

# # Show dataset information
brands.info()
finance.info()
info.info()
reviews.info()

# Merge the seperated data into one and drop null values
merged_df = info.merge(finance, on="product_id")
merged_df = merged_df.merge(reviews, on="product_id")
merged_df = merged_df.merge(brands, on="product_id")
merged_df.dropna(inplace=True)


# 1) Consumer Preferences:
# 1) Q: Does a correlation exist between revenue and reviews?
# Calculate the correlation coefficient
revenue_reviews_corr = merged_df["revenue"].corr(merged_df["reviews"])

# Create a seaborn heatmap plot
correlation_matrix = merged_df[["revenue", "reviews"]].corr()
colors = sns.color_palette("pastel")
sns.heatmap(correlation_matrix, annot=True, cmap=colors)
plt.title("Correlation between Revenue and Reviews")
plt.show()




# 2): Is there an influence on a product's rating and reviews based on the length of its description?
# Create a new column to store the length of each description
merged_df["description_length"] = merged_df["description"].str.len()

# Check the longest description length of characters to decide the bin size
print(f"The longest description length is {max(merged_df['description_length'])} characters.\n")

# Split the product description length into bins of 100 characters
bins = [0, 100, 200, 300, 400, 500, 600, 700]
labels= [100, 200, 300, 400, 500, 600, 700]
merged_df["description_length"] = pd.cut(merged_df["description_length"], bins=bins, labels=labels)

# Group by the bins and calculate the average rating and number of reviews
description_length_df = merged_df.groupby("description_length", as_index=False).agg(
    average_rating=("rating", "mean"),
    num_reviews=("reviews", "count")
).round(2)
print(description_length_df)

# Visualize the correlation between description length and mean rating of each product using a Seaborn Regression Plot
# Convert the description_length columns from string into integer to prepare for correlation
merged_df["description_length"] = merged_df["description_length"].astype(int)

# Calculate the correlation coefficient between description length and average rating
correlation_coefficient = description_length_df["description_length"].corr(description_length_df["average_rating"])

# Create a regression plot to visualize the correlation between description length and mean rating
plt.figure(figsize=(8, 6))
sns.regplot(x="description_length", y="average_rating", data=description_length_df)
plt.title(f"Scatter Plot with Correlation Line (Correlation Coefficient: {correlation_coefficient:.2f})")
plt.xlabel("Description Length")
plt.ylabel("Average Rating")
plt.xlim(90)
sns.despine()
plt.show()


# 2) Product Offerings:
# 1) Q: Do Nike and Adidas offer differing discount amounts?
# Calculate the percentage of the amount of offered discounts
discount_comparison = merged_df.groupby("brand", as_index=False).agg(
    num_discounts=("discount","count"))
adidas_discount_ratio = round((discount_comparison["num_discounts"].values[0] / len(merged_df["discount"])) * 100, 2)
nike_discount_ratio = round((discount_comparison["num_discounts"].values[1] / len(merged_df["discount"])) * 100, 2)

# Display the number of discount rates by brand
labels = ["Adidas", "Nike"]
size = [adidas_discount_ratio, nike_discount_ratio]
sns.set_style("whitegrid")
colors = sns.color_palette("pastel")
plt.pie(size, labels=labels, colors=colors, autopct='%.0f%%')
plt.title("Distribution of Offered Discount Amounts between Adidas and Nike")
plt.show()




# 2) Q: What distinguishes the price points between Nike and Adidas products?
# Label products priced up to quartile one as "Budget", quartile two as "Average", quartile three as "Expensive", and quartile four as "Elite"
# Using pandas.qcut to discretize variable into equal-sized buckets
labels = ["Budget", "Average", "Expensive", "Premium"]
merged_df["price_category"] = pd.qcut(merged_df["listing_price"], q=4, labels=labels)

# Group by brand and price_category to do agg function, create new columns and get the volume and mean revenue
adidas_vs_nike = merged_df.groupby(["brand", "price_category"], as_index=False).agg(
    num_products=("price_category", "count"),
    mean_revenue=("revenue", "mean")
).round(2)

# Creating a Seaborn Barplot to visualize the number of different products in the Adidas_vs_Nike DataFrame
plt.figure(figsize=(10,6))
sns.set_style("whitegrid")
colors = sns.color_palette("pastel")
sns.barplot(data=adidas_vs_nike, x="brand", y="num_products", hue="price_category", palette=colors)
plt.title("Number of Different Products by Price Category between Adidas and Nike")
plt.xlabel("Brand")
plt.ylabel("Number of different products")
plt.legend(title="Price Category", loc="upper right")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# Creating a Seaborn Barplot to visualize the average revenue in the Adidas_vs_Nike DataFrame
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
colors = sns.color_palette("pastel")
sns.barplot(data=adidas_vs_nike, x="brand", y="mean_revenue", hue="price_category", palette=colors)
plt.title("Average Revenue by Price Category between Adidas and Nike")
plt.xlabel("Brand")
plt.ylabel("Average Revenue")
plt.legend(title = "Price Category", loc="upper right")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# 3) Financial Performance:
# 1) Q: How much of the company's stock consists of footwear items?

# There is no column stating the type of product, so I need to rely on the "description" column
# Challenge: pattern matching -> wildcard -> https://docs.python.org/3/library/re.html#regular-expression-syntax
footwear_keyword = "shoe*|trainer*|foot*"

# Filter for footwear products
shoes = merged_df[merged_df["description"].str.contains(footwear_keyword)]

# Filter for clothing products
# How to Filter Pandas DataFrame Using Boolean Columns https://www.statology.org/pandas-filter-by-boolean-column/
clothing = merged_df[~merged_df.isin(shoes["product_id"])]

# Since it still returns all, but non-matching ones get null -> drop null
clothing.dropna(inplace=True)

# Build the product_types DataFrame, containing the number of clothing and footwear products along with their associated median revenue
product_type = pd.DataFrame({
    "num_clothing_products": len(clothing),
    "clothing_revenue_median": clothing["revenue"].median(),
    "num_footwear_products": len(shoes),
    "footwear_revenue_median": shoes["revenue"].median(),
}, index=[0])
print(product_type)


# Visualize the data to answer: How much of the company's stock consists of footwear items?
# Calculate the total number of products
total_products = len(merged_df)

# Calculate the percentage of footwear products
footwear_percentage = len(shoes) / total_products * 100

# Calculate the percentage of clothing products
clothing_percentage = len(clothing) / total_products * 100

# Create a pie chart to visualize the distribution of product types
labels = ['Footwear', 'Clothing']
sizes = [footwear_percentage, clothing_percentage]
colors = ['#ff9999','#66b3ff']
explode = (0.1, 0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title("Total Number of Products Sold by Product Category")
plt.show()


# Visualize the data to answer: How does footwear's median revenue differ from clothing products?
# Create a bar plot to compare the median revenue of footwear and clothing products
plt.bar(["Footwear", "Clothing"], [shoes["revenue"].median(), clothing["revenue"].median()], color=colors)
plt.xlabel("Category")
plt.ylabel("Median Revenue")
plt.title("Comparison of Median Revenue: Footwear vs Clothing")
plt.show()