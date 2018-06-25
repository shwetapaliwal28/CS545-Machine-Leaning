#import libraries
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Load training and test data
data_path = "/Users/shweta/Downloads/Machine Learning/"
print("Loading data\n")
spambase_data = np.loadtxt(data_path + "spambase_data.csv", delimiter=",")

#Create training and test set by equally diving spam and not-spam among train and test
spam, not_spam = [], []
for x in spambase_data:
    if x[-1] == 1:
        spam.append(x)
    else:
        not_spam.append(x)
        
#shuffle spam and not spam dataset
np.random.shuffle(spam)
np.random.shuffle(not_spam)       
half_spam = np.array_split(spam,2)
half_not_spam= np.array_split(not_spam,2) 

#Split the data into a training and test set 
train_data = np.concatenate((half_spam[0], half_not_spam[0]),axis=0)
test_data = np.concatenate((half_spam[1], half_not_spam[1]),axis=0)

Xtrain = np.asfarray(train_data[:, :-1]) 
ytrain = np.asfarray(train_data[:,-1]) 
Xtest = np.asfarray(test_data[:, :-1])
ytest = np.asfarray(test_data[:, -1]) 

#Compute the prior probability for each class, 1 (spam) and 0 (not-spam) in the training data.
def prior_probablility():
    prior_spam = round((np.count_nonzero(ytrain)) / float(len(ytrain)),1)
    prior_not_spam = round((np.count_nonzero(ytrain == 0.0)) / float(len(ytrain)),1)
    return prior_spam,prior_not_spam
    
#If any of the features has zero standard deviation, assign it a “minimal” standard deviation(0.0001)
def check_sd(value):
    if(value == 0):
        return 0.0001
    else:
        return value    

def naiveBayesTrain():
    mean_spam,mean_not_spam,sd_spam,sd_not_spam=list(),list(),list(),list()
    for i in Xtrain.T:
        s, ns = list(),list()
        for j in range(0,len(ytrain)):
            if(ytrain[j]==1):
                s.append(i[j])
            else:
                ns.append(i[j])
        #calculate mean and standard deviation for not-spam
        mean_not_spam.append(np.mean(ns))
        sd_not_spam.append(check_sd(np.std(ns)))  
        #calculate mean and standard deviation for spam 
        mean_spam.append(np.mean(s))
        sd_spam.append(check_sd(np.std(s)))
        
    return mean_spam, mean_not_spam, sd_spam, sd_not_spam

# Gaussian Naïve Bayes algorithm to classify the instances in test set
def gaussian(value, mean, sd):
    prob = (1 / (np.sqrt(2*np.pi) * sd)) * np.exp((-1)*np.square(value-mean) / (2*np.square(sd)))
    return prob


# Run Naïve Bayes on the test data.
def naiveBayesTest(mean_spam, mean_not_spam, sd_spam, sd_not_spam, prior_spam, prior_not_spam):
    prob_spam,prob_not_spam,probability= list(),list(),list()
    vector = np.vectorize(gaussian)
    for i in Xtest:
        prob_spam.append(vector(i, mean_spam, sd_spam))
        prob_not_spam.append(vector(i, mean_not_spam, sd_not_spam))
        
    #Calculate the probability of spam and not-spam for all features in test set
    for s,ns in zip(prob_spam, prob_not_spam):
        class_spam = np.log(prior_spam) + np.sum(np.log(s))
        class_not_spam = np.log(prior_not_spam) + np.sum(np.log(ns))
        #Finding  argmax
        probability.append(float(np.argmax([ class_not_spam ,class_spam])))
    return probability

#Main function
def main():
    mean_spam, mean_not_spam, sd_spam, sd_not_spam, probability=list(),list(),list(),list(),list()
    prior_spam=0.0
    prior_not_spam=0.0
    prior_spam,prior_not_spam = prior_probablility()
    
    print "\n Prior(Spam): ", prior_spam
    print "\n Prior(Not Spam): ", prior_not_spam
    
    mean_spam, mean_not_spam, sd_spam, sd_not_spam = naiveBayesTrain()
    probability = naiveBayesTest(mean_spam,mean_not_spam,sd_spam,sd_not_spam, prior_spam, prior_not_spam)
    
    #Print the accuracy, precision, and recall on the test set
    print "\n Accuracy Score: ", accuracy_score(ytest, probability)
    print "\n Precision Score: ", precision_score(ytest, probability)
    print "\n Recall: ", recall_score(ytest, probability)
    
    #Print confusion matrix for the test set.
    print "\n Confusion Matrix: \n", confusion_matrix(ytest, probability)

#Calling main function 
main()