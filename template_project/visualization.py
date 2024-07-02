# visualization.py
import matplotlib.pyplot as plt
import pandas as pd

def visualize_class_distribution(before_count, after_count, class_names):
    before_df = pd.DataFrame(list(before_count.items()), columns=['Class', 'Count Before'])
    after_df = pd.DataFrame(list(after_count.items()), columns=['Class', 'Count After'])

    # Map class indices to class names
    before_df['Class'] = before_df['Class'].map(lambda x: class_names[x])
    after_df['Class'] = after_df['Class'].map(lambda x: class_names[x])

    df = before_df.merge(after_df, on='Class')

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    before_df.plot(kind='bar', x='Class', y='Count Before', ax=ax[0], color='skyblue')
    after_df.plot(kind='bar', x='Class', y='Count After', ax=ax[1], color='salmon')
    
    ax[0].set_title('Class Distribution Before Downsampling')
    ax[0].set_xlabel('Class')
    ax[0].set_ylabel('Count')
    
    ax[1].set_title('Class Distribution After Downsampling')
    ax[1].set_xlabel('Class')
    ax[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

    # Display the table
    print(df)
