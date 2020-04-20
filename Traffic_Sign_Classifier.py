import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import cv2
import random
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import os
import matplotlib.image as mpimg
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "./datasets/train.p"
validation_file = "./datasets/valid.p"
testing_file = "./datasets/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
#%matplotlib inline
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_valid = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = max(y_train) + 1

print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


hist, bins = np.histogram(y_train, bins=n_classes)
plt.hist(y_train, bins, color='red', label ='Train')
plt.hist(y_test, bins, color='green', label = 'Valid')
plt.hist(y_valid, bins, color='blue', label = 'Test')
plt.legend(loc='upper right')
plt.show()



signname_list = pd.read_csv("./signnames.csv")
signname_list.set_index("ClassId")
signname_list.head(n=43)


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
#import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#%matplotlib inline


def categorizeImgs(idx_list, name_list):
    array = []
    for i in range(0, idx_list.shape[0]):
        idx = idx_list[i]
        name = name_list[name_list["ClassId"]==idx]["SignName"].values[0]
        array.append({"No.": i, "Index": idx, "Name": name})
    return pd.DataFrame(array)

def countCategories(categories):
    return pd.pivot_table(categories, index=["Index","Name"], values=["No."], aggfunc='count')

X_train_categories = categorizeImgs(y_train, signname_list)
#X_train_categories.head(n=10)
X_train_counts = countCategories(X_train_categories)
#X_train_counts.head(n=10)
X_train_counts.max
X_train_counts.plot(kind='bar', figsize=(16,9))
X_train_counts.max()





def rgb2gray(img_org):
    return cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY)


def affineTransImg(img, range_angle, range_shear, range_trans):
    # Rotation
    angle = np.random.uniform(range_angle) - range_angle/2
    # updated to reflect gray pipeline
    #print(img.shape)
    rows, cols, chs = img.shape    
    M_rotat = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)

    # Translation
    tr_x = range_trans*np.random.uniform() - range_trans/2
    tr_y = range_trans*np.random.uniform() - range_trans/2
    M_trans = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5  + range_shear*np.random.uniform() - range_shear/2
    pt2 = 20 + range_shear*np.random.uniform() - range_shear/2
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    M_shear = cv2.getAffineTransform(pts1,pts2)
        
    img = cv2.warpAffine(img,M_rotat,(cols,rows))
    img = cv2.warpAffine(img,M_trans,(cols,rows))
    img = cv2.warpAffine(img,M_shear,(cols,rows))
    
    return img



#Data augmentation to avoid overfitting
#Increase the number of data so that all classes has the same number of data
#ndata_goal = X_train_counts.max()   # The target number of data
ndata_goal = 2500   # The target number of data
range_angle = 15  # Range of angles for rotation
range_shear = 2   # Range of values to apply affine transform to
range_trans = 2   # Range of values to apply translations over.

nbins_before = np.bincount(y_train)

print("Generating additional data.")


#from pandas.io.parsers import read_csv
signnames = pd.read_csv("signnames.csv").values[:, 1]
unique_labels = np.unique(y_train)
#print(len(X_train))
#i=0  
for idx in range(len(unique_labels)):
    
    unique_labels = np.unique(y_train)
    #Print update to feature tracking.
    print("Current label name: ", signnames[idx])
    print("Current label index: ", idx)
    
    #Print feature currently being generate    
    y_labels = np.where(y_train == idx)
    
    ndata_orgs = len(X_train[y_labels])
    print("# of data before augmentation: ", ndata_orgs)
    ndata_diff = ndata_goal - ndata_orgs
    
    # Set features to generate to 0 if less than 0
    if ndata_diff > 0:
        ndata_togen = ndata_diff
    else:
        ndata_togen = 0
    print("# of data to generate: ", ndata_togen)
    
    if ndata_togen > 0:
       
        print("Generate data for ", signnames[idx])
        new_dataset = []
        new_indices = []
        
        # Start actually generated features while there are features to be generated
#        while i <= ndata_togen:
        while ndata_togen > 0:
            for img in X_train[y_labels]:
                
                # Graceful stopping if > 1 passes through loop
                if ndata_togen == 0: 
                    break
                
                else:
                    # generate image
                    new_img = affineTransImg(img,range_angle,range_shear,range_trans)
                    
                    new_dataset.append(new_img)
                    new_indices.append(idx)
                    
                    ndata_togen = ndata_togen - 1
#        i = i + 1

        # Append image to data
        # IMPORTANT axis=0 must be set or strange issues even though supposedly default is axis=0
        
        X_train = np.append(X_train, new_dataset, axis=0)
        y_train = np.append(y_train, new_indices, axis=0)
        
    else:
        print("Data augmentation done")
        
    # update y labels
    y_labels = np.where(y_train == idx)
    x = np.array(y_labels)
    x_min = x[0, -200]
    x_max = x[0, -1]
    random_index = random.sample(range(x_min, x_max), 5)
    
    # graphing function concepts from http://navoshta.com/traffic-signs-classification/
    fig = plt.figure(figsize = (6, 1))
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)
    
    for i in range(5):
        axis = fig.add_subplot(1, 10, i + 1, xticks=[], yticks=[])
        axis.imshow(X_train[random_index[i]], cmap="gray")
    plt.show()
    print("-----------------------------------------------------\n")

nbins_after = np.bincount(y_train)
print("Before augmentation: ", nbins_before)
print("After augmentation: ", nbins_after)


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

img_list = list()
lab_list = list()
for i in range(20):
    idx = random.randint(0, len(X_train))
    img_list.append(X_train[idx])
    lab_list.append(y_train[idx])

def normalize(img_org):
    return ((img_org - np.mean(img_org)) / np.std(img_org))


def claheHist(img_org):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img_org)


def preProcData(imgs, labs):
    imgs_gray = list(map(rgb2gray, imgs))
    imgs_equ = list(map(claheHist, imgs_gray))
    imgs_pre = list(map(normalize, imgs_equ))
    x_imgs, x_labs = shuffle(imgs_pre, labs)
    return x_imgs, x_labs


X_train_preproc, y_train_preproc = preProcData(X_train, y_train)
X_valid_preproc, y_valid_preproc = preProcData(X_train, y_train)
X_test_preproc, y_test_preproc = preProcData(X_train, y_train)


img_pre_list = list()
lab_pre_list = list()
for i in range(20):
    idx = random.randint(0, len(X_train_preproc))
    img_pre_list.append(X_train_preproc[idx])
    img_pre_list.append(y_train_preproc[idx])



### Define your architecture here.
### Feel free to use as many code cells as needed.


initializer = tf.contrib.layers.xavier_initializer()

def LeNet(img, keep_prob, dropout=True):
    mu = 0
    sigma =0.1
    
    x = tf.reshape(img, [-1,32,32,1])

    #Conv1 - input:32x32x1, output:28x28x6
    conv1_w = tf.Variable(initializer(shape=(5,5,1,6)))
    #conv1_w = tf.Variable(tf.truncated_normal(shape=(5,5,1,6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1,1,1,1], padding='VALID') + conv1_b
    #ReLU1 & MaxPool1 - input: 28x28x6, output:14x14x6
    conv1 = tf.nn.relu(tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID'))
    if dropout:
        conv1 = tf.nn.dropout(conv1, keep_prob)

    #Conv2 - input:14x14x6, output: 10x10x6
    conv2_w = tf.Variable(initializer(shape=(5,5,6,16)))
    #conv2_w = tf.Variable(tf.truncated_normal(shape=(5,5,6,16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1,1,1,1], padding='VALID') + conv2_b
    #ReLU2 & MaxPool2 - input:10x10x6, output: 5x5x16
    conv2 = tf.nn.relu(tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID'))
    if dropout:
        conv2 = tf.nn.dropout(conv2, keep_prob)

    #FC3 - flatten, input:5x5x16, output:400
    fc3 = flatten(conv2)

    #FC4 - input:400, output:120
    fc4_w = tf.Variable(initializer(shape=(400,120)))
    #fc4_w = tf.Variable(tf.truncated_normal(shape=(400,120), mean=mu, stddev=sigma))
    fc4_b = tf.Variable(tf.zeros(120))
    fc4 = tf.add(tf.matmul(fc3, fc4_w), fc4_b)
    #ReLU4
    fc4 = tf.nn.relu(fc4)
    if dropout:
        fc4 = tf.nn.dropout(fc4, keep_prob)

    #FC5 - input:120, output:84
    fc5_w = tf.Variable(initializer(shape=(120,84)))
    #fc5_w = tf.Variable(tf.truncated_normal(shape=(120,84), mean=mu, stddev=sigma))
    fc5_b = tf.Variable(tf.zeros(84))
    fc5 = tf.add(tf.matmul(fc4, fc5_w), fc5_b)
    #ReLU5
    fc5 = tf.nn.relu(fc5)
    if dropout:
        fc5 = tf.nn.dropout(fc5, keep_prob)

    #FC6 - input:84, output: n_classes
    n_classes = max(y_train) + 1
    fc6_w = tf.Variable(initializer(shape=(84,n_classes)))
    #fc6_w = tf.Variable(tf.truncated_normal(shape=(84,n_classes), mean=mu, stddev=sigma))
    fc6_b = tf.Variable(tf.zeros(n_classes))
    logit = tf.add(tf.matmul(fc5, fc6_w), fc6_b)

    return logit



### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
n_epochs = 150
n_batches = 128
r_learn = 0.0005

x = tf.placeholder(tf.float32, (None, 32, 32))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)

# added this to fix bug CUDA_ERROR_ILLEGAL_ADDRESS / kernal crash
with tf.device('/cpu:0'):
    one_hot_y = tf.one_hot(y, 43)

logits = LeNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(r_learn)
training_operation = optimizer.minimize(loss_operation)

print("Model loaded")


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data, prob=0):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, n_batches):
        batch_x, batch_y = X_data[offset:offset+n_batches], y_data[offset:offset+n_batches]
        if prob > 0:
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:prob})
            loss = sess.run(loss_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:prob})
        else:
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(loss_operation, feed_dict={x: batch_x, y: batch_y})
            
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples, loss

print("Training setting done")

global best_acc
best_acc = 0
train_accu_stack = []
valid_accu_stack = []
train_loss_stack = []
valid_loss_stack = []


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_preproc)
    
    print("Training...")
    print()
    train_loss = 0.0
    train_accu = 0.0
    for i in range(n_epochs):
        X_train_preproc, y_train_preproc = shuffle(X_train_preproc, y_train_preproc)
        for offset in range(0, num_examples, n_batches):
            end = offset + n_batches
            batch_x, batch_y = X_train_preproc[offset:end], y_train_preproc[offset:end]
            #batch_x = np.reshape(batch_x, (-1,32,32,1))
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: .8})
            train_loss = sess.run(loss_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: .8})
            train_accu = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: .8})
            
        valid_accu, valid_loss = evaluate(X_valid_preproc, y_valid_preproc, prob=1.0)

        train_accu_stack.append(train_accu)
        train_loss_stack.append(train_loss)
        
        valid_accu_stack.append(valid_accu)
        valid_loss_stack.append(valid_loss)
        
        print("EPOCH {} ...".format(i+1))
        print("Training Loss: {:.3f}, Training Accuracy = {:.3f}".format(train_loss, train_accu*100))
        print("Validation Loss: {:.3f}, Validation Accuracy = {:.3f}".format(valid_loss, valid_accu*100))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
    
print("Training done")





#Test data

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accu, test_loss = evaluate(X_test_preproc, y_test_preproc, prob=1.0)
    print("Test Accuracy = {:.3f}".format(test_accu))
    print("Test Loss = {:.3f}".format(test_loss))




#Visualize training & validation accuracy w.r.t. epochs
plt.plot(train_accu_stack, color='red', label='Train')
plt.plot(valid_accu_stack, color='blue', label='Valid')
plt.title('Accuracy')
plt.legend(loc='best')
plt.show()




#Visualize training & validation loss w.r.t. epochs
plt.plot(train_loss_stack, color='red', label='Train')
plt.plot(valid_loss_stack, color='blue', label='Valid')
plt.title('Loss')
plt.legend(loc='best')
plt.show()





### Load the images and plot them here.
### Feel free to use as many code cells as needed.
list_imgs = []

imgs = ['traffic_sign1.jpg','traffic_sign2.jpg','traffic_sign3.jpg','traffic_sign4.jpg','traffic_sign5.jpg']

for name_img in imgs:
    img_org = mpimg.imread('./test_images/'+name_img)
    img_rev = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    print(img_rev.shape)
    list_imgs.append(img_rev)

    
    
for img_test in list_imgs:
    img_resz = cv2.resize(img_test, (32,32))
    img_proc = normalize(claheHist(rgb2gray(img_resz)))
    fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(4,4))
    axis[0].imshow(img_resz)
    axis[1].imshow(img_proc, cmap='gray')






### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
def classifyImg(img_list, top_k=5):
    n_test = len(img_list)
    y_pred = np.zeros((n_test, top_k), dtype=np.int32)
    y_prob = np.zeros((n_test, top_k))
    top_5 = tf.nn.top_k(tf.nn.softmax(logits), k=top_k, sorted=True)
    
    with tf.Session() as sess:
        saver.restore(sess, './lenet')
        y_prob, y_pred = sess.run(top_5, feed_dict={x:img_list, keep_prob:1.0})
    return y_prob, y_pred



signname_list = dict()
with open('./signnames.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    signname_list = {int(row[0]):row[1] for row in reader}
    csvfile.close()


list_label = []
list_preds = []
list_probs = []

for img_test in list_imgs:
    img_resz = cv2.resize(img_test, (32,32), interpolation=cv2.INTER_NEAREST)
    img_proc = normalize(cv2.equalizeHist(rgb2gray(img_resz)))
    img_rshp = np.reshape(img_proc, (-1,32,32))
    probs,preds = classifyImg(img_rshp, 5)
    list_preds.append(np.ndarray.flatten(preds))
    labels=[]
    for pred in np.ndarray.flatten(preds):
        labels.append(signname_list[pred])
    list_label.append(labels)
    list_probs.append(np.ndarray.flatten(probs))
    
ans_labels = np.array([33, 40, 14, 25, 13])
list_label
list_probs





### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
def predictImgSoftmax(list_imgs, list_label, list_probs, fig_size=(20, 10)):
    rows = len(list_imgs)
    fg, ax = plt.subplots(nrows=rows, ncols=2, figsize=fig_size)
    for i, prob_lb in enumerate(list_label):
        img = list_imgs[i]
        ax[i,0].imshow(img)

        y_pos = np.arange(len(prob_lb))
        for j in range(0, len(prob_lb)):
            if j == 0:
                color = 'green'
            else:
                color = 'red'
            ax[i, 1].barh(j, list_probs[i][j], color=color, label="{0}".format(prob_lb[j]))

        ax[i, 1].set_yticks(y_pos)
        ax[i, 1].set_yticklabels(prob_lb)
        ax[i, 1].invert_yaxis()
        ax[i, 1].set_xlabel('Class')
        ax[i, 1].set_title('Softmax')  
        #ax[i, 1].set_xscale('log')
    
    fg.tight_layout()
    plt.show()

predictImgSoftmax(list_imgs, list_label, list_probs)





### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
top5prediction = np.array(list_preds)
accuracy = 0.0
for i, prediction in enumerate(top5prediction):
    if ans_labels[i] in prediction:
        accuracy = accuracy+1
top5accuracy = accuracy / len(ans_labels) * 100
top5accuracy





### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
