import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.manifold as TSNE
from sklearn.preprocessing import MinMaxScaler

"""
This repository provides a Jupyter notebook for customer segmentation in online retail, 
utilizing KMeans clustering and t-SNE visualization.Explore and analyze distinct customer segments based on transaction behavior with accompanying visualizations.
"""

transactions_df = pd.read_excel('online_retail_II.xlsx')

print(transactions_df.head())

transactions_df_cp = transactions_df.copy()
transactions_df_cp = transactions_df_cp.dropna()
transactions_df_cp.columns = ['invoice_id', 'product_id', 'description', "quantity", 'invoice_date', 'price',
                              'customer_id', 'country']
transactions_df_cp['customer_id'] = transactions_df_cp['customer_id'].astype(int)

CUSTOMER_ID_EXAMPLE = 13885
print(transactions_df_cp[transactions_df_cp['customer_id'] == CUSTOMER_ID_EXAMPLE])


def is_weekend(transaction_date):
    # Check if the day of the week is either Saturday (5) or Sunday (6) return transaction_date.weekday() in [5, 6]
    return transaction_date.weekday() in [5, 6]


def generate_customer_features(df):
    df['is_weekend'] = df['invoice_date'].apply(is_weekend)
    # Group the DataFrame by 'Customer ID'
    grouped = df.groupby('customer_id')

    # Calculate the average basket size by customer
    avg_basket_size = grouped['quantity'].mean()

    # Calculate the average monthly transaction price by customer (excluding the refunds)
    df['transaction_price'] = df['quantity'] * df['price']
    df['transaction_price'] = df.apply(lambda row: row['quantity'] * row['price']
    if row['quantity'] > 0 and row['price'] > 8 else 8, axis=1)

    df['year_month'] = df['invoice_date'].dt.to_period('M')

    avg_monthly_transaction_price = df.groupby(['customer_id', 'year_month'])['transaction_price'].sum().groupby(
        'customer_id').mean()

    # Calculate the average quantity price on weekends by customer
    avg_weekend_quantity_price = grouped.apply(lambda x: (x['is_weekend'] * x['transaction_price']).mean())

    # Calculate the average monthly returns by customer
    df['returns'] = df['quantity'] * df['price'] < 0

    avg_monthly_returns = df.groupby(['customer_id', 'year_month'])['returns'].mean().groupby('customer_id').mean()
    # Create a DataFrame to store the customer features
    customer_features = pd.DataFrame({'average_basket_size': avg_basket_size,
                                      'average_monthly_transaction_price': avg_monthly_transaction_price,
                                      'average_weekend_transaction_price': avg_weekend_quantity_price,
                                      'average_monthly_returns': avg_monthly_returns})
    return customer_features


customer_features = generate_customer_features(transactions_df_cp)

# Instantiate the Min-Max Scaler
scaler = MinMaxScaler()

# Fit the scaler on the data and transform the data
scaled_customer_features = scaler.fit_transform(customer_features)

# Convert the scaled data back to a DataFrame
scaled_customer_features_df = pd.DataFrame(scaled_customer_features, columns=customer_features.columns)
scaled_customer_features_df2 = scaled_customer_features_df.copy()
scaled_customer_features_df2['customer_id'] = customer_features.index

# Number of clusters (K)
n_clusters = 3

# Instantiate the KMeans model kmeans
kmeans = KMeans(n_clusters=n_clusters, random_state=43)

# Fit the model to the scaled data
kmeans.fit(scaled_customer_features_df)
# Add cluster labels to the original customer features DataFrame
customer_features['Cluster'] = kmeans.labels_

# merge the clusters with the original dataframe
transactions_with_clusters_df = pd.merge(transactions_df_cp, customer_features, on='customer_id')

# ---------------------------------------


# Set the style of the plots
sns.set(style="whitegrid")

# Create subplats for both the regular and log-transformed plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Number of customers in each cluster
sns.countplot(data=transactions_with_clusters_df, x='Cluster', palette='Setl', ax=axes(81))
axes[0].set_title('Number of Customers in Each Cluster')
axes[0].set_xlabel('Cluster')
axes[0].set_ylabel('Count')

# Annotate the bars with the count values on top
for p in axes[0].patches:
    axes[0].annotate(f'(int(p.get_height()))', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                     va='center')

# Plot 2: Log-transformed number of customers in each cluster

sns.countplot(data=transactions_with_clusters_df, x='Cluster', palette='Set1', ax=axes(11))
axes[1].set_yscale('log')

# Apply log scale to the y-axis
axes[1].set_title('Log Number of Customers in Each Cluster')
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel("Log Count")

# Annotate the bars with the count values on top (log scale)
for p in axes[1].patches:
    axes[1].annotate(f'(int(p.get_height()))', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                     va='center')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
# -------------------------------------------------
# Set the style of the plots
sns.set(style="whitegrid")

# Define a list of features to visualize
features_to_visualize = ['average_monthly_transaction_price', 'average_monthly_returns', 'average_basket_size',
                         'average_weekend_transaction']

# Create subplots for each feature
plt.figure(figsize=(14, 10))

for i, feature in enumerate(features_to_visualize, 1):
    plt.subplot(2, 2, 1)  # Create a 2x2 grid of subplots
    sns.barplot(data=transactions_with_clusters_df, x='Cluster', y=feature, palette='Set1')
    plt.title(f' (feature) vs. Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(feature)

# Adjust layout
plt.tight_layout()

sns.set(style="whitegrid")

# create a box plot for AverageMonthlyTransactionPrice by cluster
plt.figure(figsize=(8, 6))

sns.boxplot(data=transactions_with_clusters_df, x='Cluster', y='average_monthly_transaction_price', palette='Set1')
plt.title('AverageMonthlyTransactionPrice by cluster')
plt.xlabel('Cluster')
plt.ylabel('AverageMonthlyTransactionPrice')

plt.show()

tsne = TSNE.TSNE(n_components=2, random_state=42, n_iter=2000)
x_tsne = tsne.fit_transform(scaled_customer_features)

df_tsne = pd.DataFrame(data=x_tsne, columns=['t-SNE Component 1', 't-SNE Component 2'])
df_tsne['cluster'] = kmeans.labels_

sns.scatterplot(x='t-SNE Component 1', y='t-SNE Component 2', hue='cluster', data=df_tsne, palette='viridis')
plt.title('Clusters Visualized with t-SNE')
plt.show()
