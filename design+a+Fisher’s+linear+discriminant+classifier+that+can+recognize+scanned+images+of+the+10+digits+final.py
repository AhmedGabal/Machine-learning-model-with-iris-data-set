
# coding: utf-8

# In[29]:



# coding: utf-8

# In[1]:


import numpy as np 
import cv2 
from functools import reduce
import os
import pandas as pd  
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 


# In[34]:



# ## loaed images_train and preprocessing

# In[2]:


#load image train 
images_train = []
for root, dirnames, i in os.walk(r"C:\Users\Gabal\Anaconda projects\nile\week 5\train\Train"):
    for i in range(2400):
        filepath = os.path.join(root,str(i+1)+'.jpg')
        image = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
        images_train.append(image/255) 
#visulization images 
#index=1000
#plt.imshow(images_train[index])
#print('image dimention' ,images_train[index].shape ) 
#plt.show() 



# In[35]:



# In[3]:


# convert all data to matrices to able to deal easliy with it to flatten and slicing and another operations .
images_train = np.array(images_train, dtype=np.float32) 
# can me detect dtype or not 
print(images_train.shape,'\n')

images_flatten= images_train.reshape(2400,784)
print(images_flatten[0].shape)



# In[36]:



# In[4]:


# images_train division to  classes

c0=images_flatten[0:240]
c1=images_flatten[240:480]
c2=images_flatten[480:720]
c3=images_flatten[720:960]
c4=images_flatten[960:1200]
c5=images_flatten[1200:1440]
c6=images_flatten[1440:1680]
c7=images_flatten[1680:1920]
c8=images_flatten[1920:2160]
c9=images_flatten[2160:2400]
classes=[c0,c1,c2,c3,c4,c5,c6,c7,c8,c9]
#for i in range (10):
    #print(classes[i].shape)
    
#clas=[]
#for i in range(2400):
#     for n in range(10):
  #      clas.append(images_flatten.index(0,240)
    



# In[37]:



# In[5]:
#means
m=[]
for i in range (10):
    m_clas = np.mean(classes[i], axis=0,keepdims=True)
    #print(m[i].shape)
    m.append(m_clas)
  #  print(m[i].shape)

#calculate all means minus each means and put it in list
print(m[0].shape)
all_means=sum(m)
means_others=[]
for i in range (len(m)):
    mean=np.subtract((all_means),(m[i]))
    means_others.append(mean/9)
  #  print(means_others[i].shape)


#all class minus each classe and put it in list
class_others=[]
all_classes=sum(classes)
for i in range (len(classes)):
    cls= [x for ind,x in enumerate(classes) if ind!=i] #np.subtract((all_classes),(classes[i]))
    flattendedCls =[]
    for c in cls:
        for v in c:
            flattendedCls.append(v)
    class_others.append(flattendedCls)
#all_classes.shape


# In[39]:



# In[ ]:
# calculat sw
sw=[]
for n in range(10):
    #for n in range(len(classes)):
    res = np.zeros((784,784))
    for cv in classes[n]:
        s= np.subtract (cv , m[n]) 
        #print(s.shape)
        ss= s.T
        sw_one=np.dot(ss,s)
        res += sw_one
        #print(sw1.shape)
    for cv in class_others[n]:  
        c=np.subtract(cv,means_others[n])    #c=np.subtract((sum(classes,classes[n])),(sum(m,m[n])))
        cc=c.T
        sw_others= np.dot(cc,c)
        res += sw_others
    #sw + sw_all_class
    #sw_n=sum(sw_one,sw_others)
    sw.append(res)
#for i in range (10):
 #   print(sw.shape)
#print(sw.shape)
#print(sw,'\n','\n')



# In[40]:


# In[22]:

# inv. sW
SW_inv =[]
from numpy.linalg import pinv
for i in range(len(sw)):
    Sw=pinv(sw[i])
    SW_inv.append(Sw) 
    
# to visulizing lists shapes
for i in range(len(sw)):
    print(SW_inv[i].shape)
#print(all_means[0])



# In[41]:



# ## weights 

# In[23]:
W=[] 
for i in range(10):
    m_subtract =(np.subtract(means_others[i],m[i]))
   # print(m_subtract .shape)
    weight = np.dot((SW_inv[i]),m_subtract.T)
    W.append(weight) 
#print(W)
print(W[9].shape) 
# we have a list w0,w1,w2,w3,w4,w5,w6,w7,w8,w9

# bais term 
# (W.T)(sum(m))/2
# bais 
W0 =[]
for i in range(10):
    #b=(W[i].T)
    #print(b.shape)
   # bais=-(np.dot(all_means,b))//2
    bais=-.5*(np.dot(sum(m[i],means_others[i]),W[i]))
   # print(all_means.shape)
   # print(bais.shape)
    W0.append(bais)
# to print all bais values 
#for c in range (10):
 #  print(W0[c])


# In[42]:


# ## Test images and preprocessing
# In[24]:

images_test= []
for root, dirnames, i in os.walk(r"C:\Users\Gabal\Anaconda projects\nile\week 5\train\Test"):
    for i in range(200):
            filepath = os.path.join(root, str(i+1)+'.jpg')
            image = ndimage.imread(filepath)
            images_test.append(image)
#print(images_test)
#convert images to arrays 
X = np.array(images_test) 
# can me detect dtype or not 
print(X.shape,'\n')
#read labels teast.
t =pd.read_csv('Test Labels.txt')
print(t.shape)
#print(t.head(200))

# flatten 
X=X.reshape(200,28*28)
print(X.shape)
#visulization images 
#index=100
#plt.imshow(images_test[index])
#plt.show()
#print(images_test[index].shape)



# In[43]:



# ## predict

# In[25]:

predict=[]
for j in range (len(t)):             #num of images 
    y_list=[]
    for i in range (10):           #number of classes
        y=(np.dot(X[j],W[i].flatten()))+W0[i]
        y_list.append(y) 
        #for n in range(10):
    c= np.amin(y_list)             
    #print(c)                     
    #print(y_list)               
    #print(y_list.index(np.min(y_list)))    #print number of index
    predict.append(y_list.index(min(y_list)))
    #print((np.asarray(predict)).shape)
print(predict)


# In[44]:



# ## confution matrix and Accurcy

# In[26]:
from sklearn.metrics import confusion_matrix
array=confusion_matrix(t, predict)
array


# In[45]:



# In[27]:
from sklearn.metrics import accuracy_score
print(accuracy_score(t, predict)*100)


# In[14]:



# In[169]:
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

df_cm = pd.DataFrame(array, range(10),
                  range(10))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16}) # font size


