# -*- coding: utf-8 -*-
import os 


def normalizelines(filee,outname):  

    f=open(filee,'r')
    fileout = open(outname, 'w') 
    for line in f: 
        print(line) 
        normalizedline = line.lower() 
        fileout.write(normalizedline) 


    fileout.close() 
 


class Lang: 
    def __init__(self,name): 
        self.name=name 
        self.word2index={}
        self.index2word={0:"SOS",1:"EOS"} 
        self.n_words = 2

    def addSentence(self,sentence):
        for word in sentence.split(' '): 
            self.addWord(word) 

    def addWord(self,word):
        if word not in self.word2index: 
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word 
            self.n_words +=1 


def readLangs(file1,file2,langi,lango,pairs): 
    
    f1 = open(file1,'r') 
    f2 = open(file2,'r') 

    for line1,line2 in zip(f1,f2): 
        pairs.append([line1.strip('\n'),line2.strip('\n') ])  

    inputlang = Lang(langi) 
    outputlang = Lang(lango)  
    return inputlang, outputlang, pairs 


def preparedata(file1,file2,langi,lango,pairs): 
    pairs = [] 
    inputlang,outputlang, pairs = readLangs(file1,file2,langi,lango,pairs) 

    print("Read %s sentence pairs" % len(pairs)) 

    for pair in pairs: 
        inputlang.addSentence(pair[0]) 
        outputlang.addSentence(pair[1]) 

    print("Counted Words:") 
    print(inputlang.name,inputlang.n_words) 
    print(outputlang.name, outputlang.n_words) 



    return inputlang, outputlang,pairs 

'''
normalizelines('msgs/train_greng.txt', 'norm_train_greng.txt')
normalizelines('msgs/train_gr.txt', 'norm_train_gr.txt')   

pairs = [] 
inputlang, outputlang,pairs = preparedata('norm_train_greng.txt','norm_train_gr.txt','greeklish','greek') 


print(pairs[0]) 
print(inputlang.word2index) 

'''








 

