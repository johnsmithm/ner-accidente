# here will be the functions for data visualization


from wordcloud import WordCloud
from matplotlib import pyplot as plt
import nltk
import seaborn as sns
sns.set_style("dark")


# pass it a list for it to generate a word-cloud with all the words in that list
def generate_cloud(data_frame):
    all_words = ''
    for arg in data_frame:
        tokens = arg.split()  
        all_words += " ".join(tokens)+" "
    wordcloud = WordCloud(width = 1000, height = 1000, background_color ='white', min_font_size = 10).generate(all_words) 
    # plot the WordCloud image                        
    plt.figure(figsize = (5, 5), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show()

# pass it a list so that it generate an (x,y) plot where x is the words and y is the occurance of that word
def plot_word_freq(text_to_plot, title='Placeholder title', num_words=55):
    fd2 = nltk.FreqDist(word for word in text_to_plot)
    print('lenghts of the text_to_plot is',len(text_to_plot))
    x=[fd2.most_common(num_words)[i][0] for i in range(num_words)]
    y=[fd2.most_common(num_words)[i][1] for i in range(num_words)]
    #palette=sns.color_palette("PuBuGn_d",100)
    palette= sns.light_palette("crimson",100,reverse=True)
    plt.figure(figsize=(65,25))
    ax= sns.barplot(x, y, alpha=0.8,palette=palette)
    plt.title(title, fontsize=110)
    plt.ylabel('Occurrences', fontsize=190)
    plt.xlabel(' Word ', fontsize=110)
    #adding the text labels
    rects = ax.patches
    labels = y
    sns.set(font_scale=3)
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 1, label, ha='center', va='bottom')
        plt.xticks(rotation=90, fontsize=50)
    #plt.savefig('Toxic_Word_count1.png')    
    plt.show()