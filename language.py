"""
Language Modeling Project
Name:
Roll No:
"""

import language_tests as test

project = "Language" # don't edit this

### WEEK 1 ###

'''
loadBook(filename)
#1 [Check6-1]
Parameters: str
Returns: 2D list of strs
'''
def loadBook(filename):
    data=open(filename,"r").read().splitlines()
    list_2d=[data[i].split(" ") for i in range(len(data)) if len(data[i])!=0]
    return list_2d


'''
getCorpusLength(corpus)
#2 [Check6-1]
Parameters: 2D list of strs
Returns: int
'''
def getCorpusLength(corpus):
    corpus_length=sum([len(i) for i in corpus])
    return corpus_length


'''
buildVocabulary(corpus)
#3 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def buildVocabulary(corpus):
    # sort_names=list(set(i for j in corpus for i in j))
    uniq_names=[]
    for j in corpus:
        for i in j:
            if i not in uniq_names:
                uniq_names.append(i)
    return uniq_names


'''
countUnigrams(corpus)
#4 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countUnigrams(corpus):
    unigram_freq={}
    # data=[j[i] for j in corpus for i in range(len(j))]
    # for i in data:
    #     dicts[i]=data.count(i)
    for i in corpus:
        for j in i:
            if j not in unigram_freq:
                unigram_freq[j]=1
            else:
                unigram_freq[j]+=1
    return unigram_freq


'''
getStartWords(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def getStartWords(corpus):
    start_words=[]
    for i in corpus:
        if i[0] not in start_words:
            start_words.append(i[0])
    return start_words


'''
countStartWords(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countStartWords(corpus):
    start_words_freq={}
    start_words_list=getStartWords(corpus)
    Total_startwords=[corpus[i][0] for i in range(len(corpus))]
    for i in start_words_list:
        start_words_freq[i]=Total_startwords.count(i)
    return start_words_freq


'''
countBigrams(corpus)
#6 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def countBigrams(corpus):
    bigrams_freq={}
    for i in range(len(corpus)):
        for j in range(len(corpus[i])-1):
            first_word,second_word= corpus[i][j],corpus[i][j+1]
            if first_word not in bigrams_freq:
                bigrams_freq[first_word]={}
            if second_word not in bigrams_freq[first_word]:
                bigrams_freq[first_word][second_word]=0
            bigrams_freq[first_word][second_word]+=1
    return bigrams_freq


### WEEK 2 ###

'''
buildUniformProbs(unigrams)
#1 [Check6-2]
Parameters: list of strs
Returns: list of floats
'''
def buildUniformProbs(unigrams):
    uniform_probs=[1/len(unigrams) for i in unigrams]
    return  uniform_probs


'''
buildUnigramProbs(unigrams, unigramCounts, totalCount)
#2 [Check6-2]
Parameters: list of strs ; dict mapping strs to ints ; int
Returns: list of floats
'''
def buildUnigramProbs(unigrams, unigramCounts, totalCount):
    unigram_probs=[j/totalCount for i,j in unigramCounts.items()]
    return unigram_probs


'''
buildBigramProbs(unigramCounts, bigramCounts)
#3 [Check6-2]
Parameters: dict mapping strs to ints ; dict mapping strs to (dicts mapping strs to ints)
Returns: dict mapping strs to (dicts mapping strs to (lists of values))
'''
def buildBigramProbs(unigramCounts, bigramCounts):
    bigrams_probs={}
    for i in bigramCounts:
        word=[]
        prob=[]
        for j in bigramCounts[i]:
            word.append(j)
            prob.append(bigramCounts[i][j]/unigramCounts[i])
        bigrams_probs[i]={"words":word,"probs":prob}
    return bigrams_probs


'''
getTopWords(count, words, probs, ignoreList)
#4 [Check6-2]
Parameters: int ; list of strs ; list of floats ; list of strs
Returns: dict mapping strs to floats
'''
def getTopWords(count, words, probs, ignoreList):
    # dicts={}
    # max_prob=sorted(probs,reverse=True)
    # for i in range(len(words)):
    #     for j in range(len(probs)):
    #         if words[j] not in ignoreList:
    #             if probs[j]==max_prob[i]:
    #                 dicts[words[j]]=probs[j]
    # dicts=dict(list(dicts.items())[:count])
    no_ignorelist_words={}
    for i in range(len(words)):
        if words[i] not in ignoreList:
            no_ignorelist_words[words[i]]=probs[i]
    Top_words=dict(sorted(no_ignorelist_words.items(),key=lambda x:x[1],reverse=True)[:count])
    return Top_words


'''
generateTextFromUnigrams(count, words, probs)
#5 [Check6-2]
Parameters: int ; list of strs ; list of floats
Returns: str
'''
from random import choices
def generateTextFromUnigrams(count, words, probs):
    word=choices(words,weights=probs,k=count)
    string=""
    for i in word:
        string+=" "+i
    return string


'''
generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs)
#6 [Check6-2]
Parameters: int ; list of strs ; list of floats ; dict mapping strs to (dicts mapping strs to (lists of values))
Returns: str
'''
def generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs):
    string=""
    templist=[]
    for i in range(count):
        if len(templist)==0 or templist[-1]==".":
            word=choices(startWords,startWordProbs)
            templist+=word
        else:
            last_word=templist[-1]
            words=bigramProbs[last_word]["words"]
            probs=bigramProbs[last_word]["probs"]
            templist+=choices(words,probs)
    for i in templist:
        string+=" "+i
    return string


### WEEK 3 ###

ignore = [ ",", ".", "?", "'", '"', "-", "!", ":", ";", "by", "around", "over",
           "a", "on", "be", "in", "the", "is", "on", "and", "to", "of", "it",
           "as", "an", "but", "at", "if", "so", "was", "were", "for", "this",
           "that", "onto", "from", "not", "into" ]

'''
graphTop50Words(corpus)
#3 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTop50Words(corpus):
    import matplotlib.pyplot as plt
    no_dups=buildVocabulary(corpus)
    count=countUnigrams(corpus)
    length=getCorpusLength(corpus)
    probs=buildUnigramProbs(no_dups,count,length)
    Top_50=getTopWords(50, no_dups, probs, ignore)
    names=[i for i in Top_50.keys()]
    values=[j for j in Top_50.values()]
    plt.bar(names, values, width=0.6)
    plt.xticks(rotation='vertical')
    plt.xlabel("Top 50 Words")
    plt.ylabel("Probs")
    plt.title("Top 50 Words")
    plt.show()
    
    return


'''
graphTopStartWords(corpus)
#4 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTopStartWords(corpus):
    import matplotlib.pyplot as plt
    start_words=getStartWords(corpus)
    count=countStartWords(corpus)
    length=getCorpusLength(corpus)
    probs=buildUnigramProbs(start_words,count,length)
    Top_50=getTopWords(50, start_words, probs, ignore)
    names=[i for i in Top_50.keys()]
    values=[j for j in Top_50.values()]
    plt.bar(names, values, width=0.6)
    plt.xticks(rotation='vertical')
    plt.xlabel("Top Start Words")
    plt.ylabel("Probs")
    plt.title("Top 50 StartWords")
    plt.show()
    return


'''
graphTopNextWords(corpus, word)
#5 [Hw6]
Parameters: 2D list of strs ; str
Returns: None
'''
def graphTopNextWords(corpus, word):
    import matplotlib.pyplot as plt
    count=countUnigrams(corpus)
    bigrm_count=countBigrams(corpus)
    probs=buildBigramProbs(count, bigrm_count)
    Top_50=getTopWords(10, probs[word]["words"], probs[word]["probs"], ignore)
    names=[i for i in Top_50.keys()]
    values=[j for j in Top_50.values()]
    plt.bar(names, values, width=0.6)
    plt.xticks(rotation='vertical')
    plt.xlabel("Top Words")
    plt.ylabel("Probs")
    plt.title("Top 10 Next Words")
    plt.show()

    return


'''
setupChartData(corpus1, corpus2, topWordCount)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int
Returns: dict mapping strs to (lists of values)
'''
def setupChartData(corpus1, corpus2, topWordCount):
    top_words=[]
    corpus1_probs=[]
    corpus2_probs=[]
    chart_data_dicts={}

    corp1=buildVocabulary(corpus1)
    corp1_unicount=countUnigrams(corpus1)
    corp1_probs=buildUnigramProbs(corp1,corp1_unicount, getCorpusLength(corpus1))
    Top_crop1=getTopWords(topWordCount, corp1, corp1_probs, ignore)

    corp2=buildVocabulary(corpus2)
    corp2_unicount=countUnigrams(corpus2)
    corp2_probs=buildUnigramProbs(corp2,corp2_unicount, getCorpusLength(corpus2))
    Top_crop2=getTopWords(topWordCount, corp2, corp2_probs, ignore)

    top_words+=list(Top_crop1.keys())
    for i in Top_crop2.keys():
        if i not in top_words:
            top_words.append(i)
    
    for i in top_words:
        if i in corp1:
            index=corp1.index(i)
            corpus1_probs.append(corp1_probs[index])
        else:
            corpus1_probs.append(0)
        if i in corp2:
            index=corp2.index(i)
            corpus2_probs.append(corp2_probs[index])
        else:
            corpus2_probs.append(0)
    
    chart_data_dicts["topWords"]=top_words
    chart_data_dicts["corpus1Probs"]=corpus1_probs
    chart_data_dicts["corpus2Probs"]=corpus2_probs
    return chart_data_dicts


'''
graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; str ; 2D list of strs ; str ; int ; str
Returns: None
'''
def graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title):
    data=setupChartData(corpus1, corpus2, numWords)
    xValues=data["topWords"]
    values1=data["corpus1Probs"]
    values2=data["corpus2Probs"]
    sideBySideBarPlots(xValues, values1, values2, name1, name2, title)

    return


'''
graphTopWordsInScatterplot(corpus1, corpus2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int ; str
Returns: None
'''
def graphTopWordsInScatterplot(corpus1, corpus2, numWords, title):
    data=setupChartData(corpus1, corpus2, numWords)
    labels=data["topWords"]
    xs=data["corpus1Probs"]
    ys=data["corpus2Probs"]
    scatterPlot(xs, ys, labels, title)
    return


### WEEK 3 PROVIDED CODE ###

"""
Expects a dictionary of words as keys with probabilities as values, and a title
Plots the words on the x axis, probabilities as the y axis and puts a title on top.
"""
def barPlot(dict, title):
    import matplotlib.pyplot as plt

    names = []
    values = []
    for k in dict:
        names.append(k)
        values.append(dict[k])

    plt.bar(names, values)

    plt.xticks(rotation='vertical')
    plt.title(title)

    plt.show()

"""
Expects 3 lists - one of x values, and two of values such that the index of a name
corresponds to a value at the same index in both lists. Category1 and Category2
are the labels for the different colors in the graph. For example, you may use
it to graph two categories of probabilities side by side to look at the differences.
"""
def sideBySideBarPlots(xValues, values1, values2, category1, category2, title):
    import matplotlib.pyplot as plt

    w = 0.35  # the width of the bars

    plt.bar(xValues, values1, width=-w, align='edge', label=category1)
    plt.bar(xValues, values2, width= w, align='edge', label=category2)

    plt.xticks(rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Note that this limits the graph to go from 0x0 to 0.02 x 0.02.
"""
def scatterPlot(xs, ys, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xs, ys)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xs[i], ys[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.xlim(0, 0.02)
    plt.ylim(0, 0.02)

    # a bit of advanced code to draw a y=x line
    ax.plot([0, 1], [0, 1], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    # print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    # test.week1Tests()
    # print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    #test.runWeek1()
    test.testLoadBook()
    test.testGetCorpusLength()
    test.testBuildVocabulary()
    test.testCountUnigrams()
    test.testGetStartWords()
    test.testCountStartWords()
    test.testCountBigrams()
    # Uncomment these for Week 2 ##
    test.testBuildUniformProbs()
    test.testBuildUnigramProbs()
    test.testBuildBigramProbs()
    test.testGetTopWords()
    test.testGenerateTextFromUnigrams()
    test.testGenerateTextFromBigrams()
    #test.runWeek2()
    test.testSetupChartData()
    test.runWeek3()
    
"""
    print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()
"""
    ## Uncomment these for Week 3 ##
    
"""
    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()
"""
