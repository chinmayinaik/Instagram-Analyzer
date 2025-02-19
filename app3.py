import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import ScalarFormatter, MaxNLocator

# Function to load data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin1')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['post_hashtags'] = df['post_hashtags'].str.lower().fillna('')
    df['account'] = df['account'].str.lower().fillna('')
    df['posts_count'] = pd.to_numeric(df['posts_count'], errors='coerce').fillna(0)
    df['avg_engagement'] = pd.to_numeric(df['avg_engagement'], errors='coerce').fillna(0)
    df['post_likes'] = pd.to_numeric(df['post_likes'], errors='coerce').fillna(0)
    df['post_comments'] = pd.to_numeric(df['post_comments'], errors='coerce').fillna(0)
    df['post_shares'] = pd.to_numeric(df['post_shares'], errors='coerce').fillna(0)
    df['followers'] = pd.to_numeric(df['followers'], errors='coerce').fillna(0)
    return df

# Load sample data
file_path = r"C:\Users\chinm\OneDrive\Desktop\ig analyzer\Instagram Profiles - Github Hashtag.csv"
df = load_data(file_path)

# Sidebar for user inputs
st.sidebar.title('Instagram Engagement Analyzer')

option = 'Hashtags'  # Removed the option to select account, default is now 'Hashtags'

user_input = st.sidebar.text_area('Enter hashtags to analyze (separate multiple entries with commas):')

# Filter data based on user input
if user_input:
    user_input_list = [item.strip().lower() for item in user_input.split(',')]

    filtered_df = df[df['post_hashtags'].apply(lambda x: any(hashtag in x for hashtag in user_input_list))]

    if not filtered_df.empty:
        st.write(f'Analysis for Hashtags: {", ".join(user_input_list)}')

        # Split data into features and target variable
        X = filtered_df[['followers']]  # Adjust features as per your data
        y = filtered_df['avg_engagement']  # Adjust target variable as per your data

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize RandomForestRegressor
        model = RandomForestRegressor(random_state=42)

        # Fit the model on training data
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Calculate Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse:.2f}")

        # Calculate R-squared
        #r2 = r2_score(y_test, y_pred)
        #st.write(f"R-squared: {r2:.2f}")

        # Plotting
        st.subheader('Engagement Metrics')

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.scatterplot(
            x=X_test['followers'],
            y=y_test,
            ax=ax,
            color='blue',
            label='Actual'
        )

        sns.scatterplot(
            x=X_test['followers'],
            y=y_pred,
            ax=ax,
            color='red',
            label='Predicted'
        )

        ax.set_title(f'Followers vs Average Engagement for Hashtags: {", ".join(user_input_list)}')
        ax.set_xlabel('Followers')
        ax.set_ylabel('Average Engagement')

        # Set x-axis to use ScalarFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='x')

        # Use MaxNLocator to automatically set a reasonable number of ticks
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

        # Add legend
        ax.legend()

        # Adjust layout
        plt.tight_layout()

        st.pyplot(fig)
      
        # Engagement Counts: Likes, Comments, Shares
        st.subheader('Engagement Counts: Likes, Comments, Shares')
        engagement_counts = filtered_df[['post_likes', 'post_comments', 'post_shares']].sum().reset_index()
        engagement_counts.columns = ['Engagement Type', 'Count']
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Engagement Type', y='Count', data=engagement_counts, palette='viridis', ax=ax)
        ax.set_title(f'Engagement Counts for Hashtags: {", ".join(user_input_list)}')
        ax.set_ylabel('Count')
        st.pyplot(fig)

        # Time Series Analysis of Engagement Over Time
        st.subheader('Time Series Analysis of Engagement Over Time')
        fig, ax = plt.subplots(figsize=(10, 6))
        filtered_df.set_index('timestamp')['avg_engagement'].resample('M').mean().plot(ax=ax, color='green')
        ax.set_title('Monthly Average Engagement Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Engagement')
        st.pyplot(fig)

        # Correlation Heatmap of Engagement Metrics
        st.subheader('Correlation Heatmap of Engagement Metrics')
        corr_matrix = filtered_df[['followers', 'avg_engagement', 'post_likes', 'post_comments', 'post_shares']].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)

        # Top Accounts by Engagement
        st.subheader('Top Accounts by Engagement')
        top_accounts = filtered_df.groupby('account')['avg_engagement'].mean().sort_values(ascending=False).head(10).reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='avg_engagement', y='account', data=top_accounts, palette='plasma', ax=ax)
        ax.set_title('Top 10 Accounts by Average Engagement')
        ax.set_xlabel('Average Engagement')
        ax.set_ylabel('Account')
        st.pyplot(fig)

        # List all users with the specified parameter
        st.subheader(f'All Users for Hashtags: {", ".join(user_input_list)}')
        users_df = filtered_df[['account', 'business_category_name']].drop_duplicates().reset_index(drop=True)
        st.table(users_df)
                
    else:
        st.write(f'No data found for Hashtags: {", ".join(user_input_list)}')
else:
    st.write('Please enter one or more hashtags to analyze.')

# Display the data for reference
st.write("### Sample Data")
st.dataframe(df)
