# Analysis of Sportswear Product Sales: Adidas vs. Nike

## Project Overview:
This data analysis project aims to support an online sports clothing company that exclusively sells Adidas or Nike-branded products in boosting its revenue by providing actionable recommendations.

![image](https://github.com/fangoaish/Python__Analysis-of-Sportswear-Product-Sales-Adidas-vs.-Nike/assets/51399519/2584a6fa-3c14-4538-9f6d-0a30f3118f7b)

## Business Objective:

The primary objective is to gain actionable insights into the market dynamics and product strategies employed by Nike and Adidas. By addressing specific questions related to pricing, discounts, customer reviews, product descriptions, review trends, and revenue breakdown, the goal is to equip stakeholders with valuable information for strategic decision-making.

The sports clothing and athleisure sector, valued at around $193 billion in 2021 according to [Statista](https://www.statista.com/statistics/254489/total-revenue-of-the-global-sports-apparel-market/), is a thriving industry expected to experience significant growth in the coming decade.

Within this document, the company is specifically focused on enhancing **_its revenue streams_**. The analysis will delve into various aspects of product data, including pricing, reviews, descriptions, ratings, as well as revenue, and website traffic.



## Data Sources:
These four datasets are provided to investigate:

### brands.csv

| Columns | Description |
|---------|-------------|
| `product_id` | Unique product identifier |
| `brand` | Brand of the product | 

### finance.csv

| Columns | Description |
|---------|-------------|
| `product_id` | Unique product identifier |
| `listing_price` | Original price of the product | 
| `sale_price` | Discounted price of the product |
| `discount` | Discount off the listing price, as a decimal | 
| `revenue` | Revenue generated by the product |

### info.csv

| Columns | Description |
|---------|-------------|
| `product_name` | Name of the product | 
| `product_id` | Unique product identifier |
| `description` | Description of the product |

### reviews.csv

| Columns | Description |
|---------|-------------|
| `product_id` | Unique product identifier |
| `rating` | Average product rating | 
| `reviews` | Number of reviews for the product |


## Data Preparation:
No data preparation tasks were required as all the data provided had already been cleaned out prior.

## Goal:
This analysis aims to assist in identifying potential areas of improvement, understanding **consumer preferences**, and optimizing **product offerings** to enhance the overall competitiveness and **financial performance** of the respective brands.

## Exploratory Data Analysis
Before diving into the data sea, I'll categorize the hypotheses systematically based on our goal, establishing a structured and logical framework for thoughtful analysis.

## _1) Consumer Preferences:_
- ### **Q:** Does a correlation exist between revenue and reviews?
    - Why do I want to know? 
        - Explore the strength of any correlation that may exist between a product's revenue and its reviews
    - So what?     
        - Implement initiatives to encourage and incentivize customer reviews, fostering increased engagement and potentially driving higher revenues through positive customer feedback
    - Measure by?
        - revenue // reviews

- ### **Q:** Is there an influence on a product's rating and reviews based on the length of its description?
    -  Why do I want to know?
        - Explore the potential impact of product description length on customer ratings and reviews, addressing the relationship between product information and consumer perception.
    - So what?
        - If positive, highlight product features within the optimal description length to attract and engage customers effectively
    - Measure by?
        - description // reviews // rating

![Correlation between Revenue and Reviews](https://github.com/fangoaish/Python__Analysis-of-Sportswear-Product-Sales-Adidas-vs.-Nike/assets/51399519/43de922e-bb14-4e04-8ed0-9638154a2c54)

![Correlation Between Description Length and Mean Rating](https://github.com/fangoaish/Python__Analysis-of-Sportswear-Product-Sales-Adidas-vs.-Nike/assets/51399519/23ff228c-94d6-49e5-8190-ec332522cdb2)

## _1) Consumer Preferences - Findings_
1. A correlation coefficient of 0.65 could be interpreted as either a "good" or "moderate" correlation. Therefore, there is a positive correlation between revenue and reviews, suggesting that products with higher reviews tend to generate higher revenue.
2. The correlation coefficient of 0.73 indicates the strength and direction of the linear relationship between description length and the average rating.

## _1) Consumer Preferences - Recommendations_
- Leverage the positive correlation observed between revenue and reviews to enhance marketing strategies, emphasizing the importance of customer reviews in promotional campaigns and product positioning.
- Consider optimizing product descriptions to a length that resonates well with customers. This could potentially lead to higher average ratings.

## _2) Product Offerings:_
- ### **Q:** Do Nike and Adidas offer differing discount amounts?
    - Why do I want to know? 
        - Aim to investigate and compare the discount strategies employed by Nike and Adidas, providing insights into their promotional approaches.
    - So what?
        - Navigate the complexities of differing discount amounts, optimizing their strategies to achieve a balance between attracting customers, maintaining brand value, and sustaining profitable operations
    - Measure by?
        - brand // discount


- ### **Q:** What distinguishes the price points between Nike and Adidas products?
    - Why do I want to know?: 
        - Focus on understanding the comparative pricing strategies of Nike and Adidas, exploring the differences in their product price points.
    - So what? 
        - Analyze how the variations in price points may impact consumer perception, market positioning, and business strategies for Nike and Adidas
    - Measure by?
        - brand // listing_price

![Distribution of Offered Discount Amounts between Adidas and Nike](https://github.com/fangoaish/Python__Analysis-of-Sportswear-Product-Sales-Adidas-vs.-Nike/assets/51399519/0b5bfc01-4bdd-4b17-a874-a0f43df0dced)

![Number of Different Products by Price Category between Adidas and Nike](https://github.com/fangoaish/Python__Analysis-of-Sportswear-Product-Sales-Adidas-vs.-Nike/assets/51399519/d0d04744-dfc4-46db-828d-1bb2540aaf8d)

![Average Revenue by Price Category between Adidas and Nike](https://github.com/fangoaish/Python__Analysis-of-Sportswear-Product-Sales-Adidas-vs.-Nike/assets/51399519/469f76fc-5173-43c1-ada9-843f919c3be6)
