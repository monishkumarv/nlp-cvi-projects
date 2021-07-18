import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

def DisplayAttentionWeights(attention_weights, germ_word_index, eng_word_index, actual, predicted, index, scale):
    
    def germ_convert_int2word(data):
        text = []
        for line in data:
            sentence = ' '.join([germ_word_index.get(i) if germ_word_index.get(i) is not None else '<NONE>' for i in line])
            # sentence = ' '.join([i for i in sentence.split() if i != 'EOS'])
            text.append(sentence)
        return text

    def eng_convert_int2word(data):
        text = []
        for line in data:
            sentence = ' '.join([eng_word_index.get(i) if eng_word_index.get(i) is not None else '<NONE>' for i in line])        
            # sentence = ' '.join([i for i in sentence.split() if i != 'EOS'])
            text.append(sentence)
        return text
    
    german = germ_convert_int2word(actual)[index].split()
    english = eng_convert_int2word(predicted)[index].split()
    
    xrange = len(german)
    yrange = len(english)

    fig = plt.figure(figsize=(xrange * scale, yrange * scale))
    ax = fig.add_subplot(1, 1, 1)
    img = sns.heatmap(np.flipud(attention_weights[index]), linewidths = 0.25)

    # ax.set_xticks(range(xrange))
    ax.set_xticklabels(german, rotation=90)
    ax.set_xlabel('German', labelpad=20)
    ax.xaxis.tick_top()

    # ax.set_yticks(range(yrange))
    ax.set_yticklabels(english, rotation=0)
    ax.set_ylabel('English', labelpad=20)

    fig.show()
