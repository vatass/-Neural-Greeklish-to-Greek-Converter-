# -*- coding: utf-8 -*-
import os 


def preprocess(file1,file2,i):  

    f1=open(file1,'r') 
    f2=open(file2,'r')
    if i==0:
        fileout=open('gr-greng.txt','w') 
    else: 
        fileout=open('greng_test.txt','w') 

    for line1,line2 in zip(f1,f2): 
        print(line1) 
        print(line2) 
        totalline=line1.rstrip() + '\t' + line2.rstrip() + '\n'  
        fileout.write(totalline) 


    fileout.close() 


dirr = 'msgs/' 
#file_path1 = '/files/greng_train.txt' 
#file_path2 = '/files/greng_test.txt' 
#newdir1 = os.path.dirname(file_path1) 
#newdir2 = os.path.dirname(file_path2)  
preprocess('msgs/train_greng.txt','msgs/train_gr.txt',0) 
#preprocess('msgs/test_greng.txt', 'msgs/test_gr.txt',1)  


