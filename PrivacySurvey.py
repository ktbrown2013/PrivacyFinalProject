import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#readin results and remove timestamp column
surveyResults = pd.read_csv('598 Privacy Survey.csv')
surveyResults = surveyResults.iloc[:,1:]

#replace responses on integer scale
#never-always is 1-5
surveyResults.iloc[:,:].mask(surveyResults.iloc[:,:] == 'Never', 1, inplace = True)
surveyResults.iloc[:,:].mask(surveyResults.iloc[:,:] == 'Infrequently', 2, inplace = True)
surveyResults.iloc[:,:].mask(surveyResults.iloc[:,:] == 'Sometimes', 3, inplace = True)
surveyResults.iloc[:,:].mask(surveyResults.iloc[:,:] == 'Often', 4, inplace = True)
surveyResults.iloc[:,:].mask(surveyResults.iloc[:,:] == 'Always', 5, inplace = True)
#comfortable-Concerned is 1-5
surveyResults.iloc[:,:].mask(surveyResults.iloc[:,:] == 'Comfortable', 1, inplace = True)
surveyResults.iloc[:,:].mask(surveyResults.iloc[:,:] == 'Somewhat comfortable', 2, inplace = True)
surveyResults.iloc[:,:].mask(surveyResults.iloc[:,:] == 'Neither comfortable or concered', 3, inplace = True)
surveyResults.iloc[:,:].mask(surveyResults.iloc[:,:] == 'Somewhat concerned', 4, inplace = True)
surveyResults.iloc[:,:].mask(surveyResults.iloc[:,:] == 'Concerned', 5, inplace = True)
#Strongly Disagree-Strongly Agree is 1-5
surveyResults.iloc[:,:].mask(surveyResults.iloc[:,:] == 'Strongly Disagree', 1, inplace = True)
surveyResults.iloc[:,:].mask(surveyResults.iloc[:,:] == 'Disagree', 2, inplace = True)
surveyResults.iloc[:,:].mask(surveyResults.iloc[:,:] == 'Neither Agree or Disagree', 3, inplace = True)
surveyResults.iloc[:,:].mask(surveyResults.iloc[:,:] == 'Agree', 4, inplace = True)
surveyResults.iloc[:,:].mask(surveyResults.iloc[:,:] == 'Strongly Agree', 5, inplace = True)

#categorization of users
#add rows of users in a category to a new dataframe for that category
fitResults = surveyResults.iloc[:,0]
fitUsers = surveyResults.loc[fitResults >= 3]
alwaysFitUsers = surveyResults.loc[fitResults == 5]
oftenFitUsers = surveyResults.loc[fitResults == 4]
someFitUsers = surveyResults.loc[fitResults == 3]
infreqFitUsers = surveyResults.loc[fitResults == 2]
neverFitUsers = surveyResults.loc[fitResults == 1]

commResults = surveyResults.iloc[:,1]
commUsers = surveyResults.loc[commResults >= 3]
alwaysCommUsers = surveyResults.loc[commResults == 5]
oftenCommUsers = surveyResults.loc[commResults >= 4]
someCommUsers = surveyResults.loc[commResults >= 3]
infreqCommUsers = surveyResults.loc[commResults >= 2]
neverCommUsers = surveyResults.loc[commResults >= 1]

convResults = surveyResults.iloc[:,2]
convUsers = surveyResults.loc[convResults >= 3]

styleResults = surveyResults.iloc[:,3]
styleUsers = surveyResults.loc[styleResults >= 3]

otherResults = surveyResults.iloc[:,4]
otherUsers = surveyResults.loc[otherResults >= 3]

understandResults = surveyResults.iloc[:,5]
stronglyDisagreeUsers = surveyResults.loc[understandResults == 1]
disagreeUsers = surveyResults.loc[understandResults == 2]
neitherUsers = surveyResults.loc[understandResults == 3]
agreeUsers = surveyResults.loc[understandResults == 4]
stronglyAgreeUsers = surveyResults.loc[understandResults == 5]



#graph user categorization based on question 2(prior understanding)
categoryData = {'Strongly Disagree':len(stronglyDisagreeUsers), 'Disagree':len(disagreeUsers), 'Neither Agree or Disagree':len(neitherUsers), 'Agree':len(agreeUsers), 'Strongly Agree':len(stronglyAgreeUsers)}
category = list(categoryData.keys())
catValues = list(categoryData.values())
fig = plt.figure(figsize = (10,5))
plt.bar(category,catValues,width = 0.4)
plt.xlabel('Category of User')
plt.ylabel('No. of Users')
plt.title('Categorization of Users Based on Perceived Understanding of their Smartwatch Data Collection')
plt.show()

#graph user categorization
categoryData = {'Fitness':len(fitUsers), 'Communication':len(commUsers), 'Convenience':len(convUsers), 'Style/Fashion':len(styleUsers), 'Other':len(otherUsers)}
category = list(categoryData.keys())
catValues = list(categoryData.values())
fig = plt.figure(figsize = (10,5))
plt.bar(category,catValues,width = 0.4)
plt.xlabel('Category of User')
plt.ylabel('No. of Users')
plt.title('Categorization of Users')
plt.show()

#returns num of responses for each answer to a question from a group of users
#df is user group and i is index of question
def findNumUsers(df, i):
    #iterate through column of answer
    numResponses = []
    category = df.iloc[:,i]
    numResponses.append(len(df.loc[category == 1]))
    numResponses.append(len(df.loc[category == 2]))
    numResponses.append(len(df.loc[category == 3]))
    numResponses.append(len(df.loc[category == 4]))
    numResponses.append(len(df.loc[category == 5]))
    return numResponses

#scatter based on utilization of features and comfort with topisc
#print(surveyResults.iloc[0:5].sum())
utilizationScores = []
biometricComfortScores = []
locationComfortScores = []
audioComfortScores = []
fitUserBiometricComfortScores = []
partyComfortScores = []
for i in range(len(surveyResults)):
    #store utilization scores for all users
    utilizationScore = surveyResults.iloc[i,0:5].sum()
    utilizationScores.append(utilizationScore)
    #store biometric comfort score for all users
    biometricComfortScore = surveyResults.iloc[i,15:23].sum()
    biometricComfortScores.append(biometricComfortScore/8)
    #store location comfort score for all users
    locationComfortScore = surveyResults.iloc[i,14]
    locationComfortScores.append(locationComfortScore)
    #store audio comfort score for all users
    audioComfortScore = surveyResults.iloc[i,26]
    audioComfortScores.append(audioComfortScore)
    #store third-party comfort score for all users
    partyComfortScore = surveyResults.iloc[i,23] + surveyResults.iloc[i,24]
    partyComfortScores.append(partyComfortScore/2)



plt.scatter(utilizationScores, partyComfortScores)
plt.xlabel('Utilization scores(less = uses less features)')
plt.ylabel('Comfort Scores(less = more comfortable)')
plt.title('User comfortableness with tracked biometric data')
plt.show()

utilizationScores = np.array(utilizationScores)
comfortScores = np.array(biometricComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
print(utilizationScores)
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilBiometricComfortData.xlsx")

comfortScores = np.array(locationComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilLocationComfortData.xlsx")

comfortScores = np.array(audioComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilAudioComfortData.xlsx")

comfortScores = np.array(partyComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilPartyComfortData.xlsx")

#do utilization vs comfort scatter plots for fitness users
utilizationScores = []
biometricComfortScores = []
locationComfortScores = []
audioComfortScores = []
fitUserBiometricComfortScores = []
partyComfortScores = []
for i in range(len(fitUsers)):
    #store utilization scores for all users
    utilizationScore = fitUsers.iloc[i,0:5].sum()
    utilizationScores.append(utilizationScore)
    #store biometric comfort score for all users
    biometricComfortScore = fitUsers.iloc[i,15:23].sum()
    biometricComfortScores.append(biometricComfortScore/8)
    #store location comfort score for all users
    locationComfortScore = fitUsers.iloc[i,14]
    locationComfortScores.append(locationComfortScore)
    #store audio comfort score for all users
    audioComfortScore = fitUsers.iloc[i,26]
    audioComfortScores.append(audioComfortScore)
    #store third-party comfort score for all users
    partyComfortScore = fitUsers.iloc[i,23] + fitUsers.iloc[i,24]
    partyComfortScores.append(partyComfortScore/2)

utilizationScores = np.array(utilizationScores)
comfortScores = np.array(biometricComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
print(utilizationScores)
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilBiometricComfortData.xlsx")

comfortScores = np.array(locationComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilLocationComfortData.xlsx")

comfortScores = np.array(audioComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilAudioComfortData.xlsx")

comfortScores = np.array(partyComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilPartyComfortData.xlsx")

#do utilization vs comfort scatter plots for communication users
utilizationScores = []
biometricComfortScores = []
locationComfortScores = []
audioComfortScores = []
fitUserBiometricComfortScores = []
partyComfortScores = []
for i in range(len(commUsers)):
    #store utilization scores for all users
    utilizationScore = commUsers.iloc[i,0:5].sum()
    utilizationScores.append(utilizationScore)
    #store biometric comfort score for all users
    biometricComfortScore = commUsers.iloc[i,15:23].sum()
    biometricComfortScores.append(biometricComfortScore/8)
    #store location comfort score for all users
    locationComfortScore = commUsers.iloc[i,14]
    locationComfortScores.append(locationComfortScore)
    #store audio comfort score for all users
    audioComfortScore = commUsers.iloc[i,26]
    audioComfortScores.append(audioComfortScore)
    #store third-party comfort score for all users
    partyComfortScore = commUsers.iloc[i,23] + commUsers.iloc[i,24]
    partyComfortScores.append(partyComfortScore/2)

utilizationScores = np.array(utilizationScores)
comfortScores = np.array(biometricComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
print(utilizationScores)
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilBiometricComfortData.xlsx")

comfortScores = np.array(locationComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilLocationComfortData.xlsx")

comfortScores = np.array(audioComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilAudioComfortData.xlsx")

comfortScores = np.array(partyComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilPartyComfortData.xlsx")

#do utilization vs comfort scatter plots for convenience users
utilizationScores = []
biometricComfortScores = []
locationComfortScores = []
audioComfortScores = []
fitUserBiometricComfortScores = []
partyComfortScores = []
for i in range(len(convUsers)):
    #store utilization scores for all users
    utilizationScore = convUsers.iloc[i,0:5].sum()
    utilizationScores.append(utilizationScore)
    #store biometric comfort score for all users
    biometricComfortScore = convUsers.iloc[i,15:23].sum()
    biometricComfortScores.append(biometricComfortScore/8)
    #store location comfort score for all users
    locationComfortScore = convUsers.iloc[i,14]
    locationComfortScores.append(locationComfortScore)
    #store audio comfort score for all users
    audioComfortScore = convUsers.iloc[i,26]
    audioComfortScores.append(audioComfortScore)
    #store third-party comfort score for all users
    partyComfortScore = convUsers.iloc[i,23] + convUsers.iloc[i,24]
    partyComfortScores.append(partyComfortScore/2)

utilizationScores = np.array(utilizationScores)
comfortScores = np.array(biometricComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
print(utilizationScores)
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilBiometricComfortData.xlsx")

comfortScores = np.array(locationComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilLocationComfortData.xlsx")

comfortScores = np.array(audioComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilAudioComfortData.xlsx")

comfortScores = np.array(partyComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilPartyComfortData.xlsx")

#do utilization vs comfort scatter plots for Style users
utilizationScores = []
biometricComfortScores = []
locationComfortScores = []
audioComfortScores = []
fitUserBiometricComfortScores = []
partyComfortScores = []
for i in range(len(styleUsers)):
    #store utilization scores for all users
    utilizationScore = styleUsers.iloc[i,0:5].sum()
    utilizationScores.append(utilizationScore)
    #store biometric comfort score for all users
    biometricComfortScore = styleUsers.iloc[i,15:23].sum()
    biometricComfortScores.append(biometricComfortScore/8)
    #store location comfort score for all users
    locationComfortScore = styleUsers.iloc[i,14]
    locationComfortScores.append(locationComfortScore)
    #store audio comfort score for all users
    audioComfortScore = styleUsers.iloc[i,26]
    audioComfortScores.append(audioComfortScore)
    #store third-party comfort score for all users
    partyComfortScore = styleUsers.iloc[i,23] + styleUsers.iloc[i,24]
    partyComfortScores.append(partyComfortScore/2)

utilizationScores = np.array(utilizationScores)
comfortScores = np.array(biometricComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
print(utilizationScores)
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilBiometricComfortData.xlsx")

comfortScores = np.array(locationComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilLocationComfortData.xlsx")

comfortScores = np.array(audioComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilAudioComfortData.xlsx")

comfortScores = np.array(partyComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilPartyComfortData.xlsx")

#do utilization vs comfort scatter plots for Other users
utilizationScores = []
biometricComfortScores = []
locationComfortScores = []
audioComfortScores = []
fitUserBiometricComfortScores = []
partyComfortScores = []
for i in range(len(otherUsers)):
    #store utilization scores for all users
    utilizationScore = otherUsers.iloc[i,0:5].sum()
    utilizationScores.append(utilizationScore)
    #store biometric comfort score for all users
    biometricComfortScore = otherUsers.iloc[i,15:23].sum()
    biometricComfortScores.append(biometricComfortScore/8)
    #store location comfort score for all users
    locationComfortScore = otherUsers.iloc[i,14]
    locationComfortScores.append(locationComfortScore)
    #store audio comfort score for all users
    audioComfortScore = otherUsers.iloc[i,26]
    audioComfortScores.append(audioComfortScore)
    #store third-party comfort score for all users
    partyComfortScore = otherUsers.iloc[i,23] + otherUsers.iloc[i,24]
    partyComfortScores.append(partyComfortScore/2)

utilizationScores = np.array(utilizationScores)
comfortScores = np.array(biometricComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
print(utilizationScores)
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilBiometricComfortData.xlsx")

comfortScores = np.array(locationComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilLocationComfortData.xlsx")

comfortScores = np.array(audioComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilAudioComfortData.xlsx")

comfortScores = np.array(partyComfortScores)
utilComfortData = np.vstack([utilizationScores, comfortScores])
df = pd.DataFrame(utilComfortData)
df.to_excel(excel_writer = "utilPartyComfortData.xlsx")

#correlate 2 with last 4 questions

x = ['Never','Infrequently','Sometimes','Often','Always']

plt.plot(x, findNumUsers(alwaysFitUsers, 9), color = 'red', label = 'Always')
plt.plot(x, findNumUsers(oftenFitUsers, 9), color = 'blue', label = 'Often')
plt.plot(x, findNumUsers(someFitUsers, 9), color = 'green', label = 'Sometimes')
plt.plot(x, findNumUsers(infreqFitUsers, 9), color = 'orange', label = 'Infrequently')
plt.plot(x, findNumUsers(neverFitUsers, 9), color = 'black', label = 'Never')
plt.legend(title = 'Type of Fitness User')
plt.xlabel('Response to Heart Info Perceptiveness')
plt.ylabel('Number of Responses')
plt.title('Fitness Users Tracking Heart Info Perceptiveness')
plt.show()

#12-26 are comfort questions
#12steps, 13distance traveled, 14gps location, 15calories, 16sleeptime, 17sleepquality, 18physical exertion, 19heart rate, 20ekg, 21breath rate, 22blood oxygen, 23call/text data,
#24notifications, 25driving data, 26audio data

x = ['Comfortable', 'Somewhat comfortable', 'Neither comfortable or concerned', 'Somewhat concerned', 'Concerned']
plt.plot(x, findNumUsers(alwaysFitUsers, 19), color = 'red', label = 'Always')
plt.plot(x, findNumUsers(oftenFitUsers, 19), color = 'blue', label = 'Often')
plt.plot(x, findNumUsers(someFitUsers, 19), color = 'green', label = 'Sometimes')
plt.plot(x, findNumUsers(infreqFitUsers, 19), color = 'orange', label = 'Infrequently')
plt.plot(x, findNumUsers(neverFitUsers, 19), color = 'black', label = 'Never')
plt.legend(title = 'Type of Fitness User')
plt.xlabel('Response to Comfort with Tracking Heart Rate')
plt.ylabel('Number of Responses')
plt.title('Fitness Users Comfort Levels with Heart Rate Tracking')
plt.show()





#graph perceptiveness of location, sleep, heart info, audio, third-party data
#takes dataframe of users, column index of question (7-11), and type of user
def graphPerceptiveness(df,i, user, type):
    category = df.iloc[:,i]
    neverResults = df.loc[category == 1]
    infreqResults = df.loc[category == 2]
    someResults = df.loc[category == 3]
    oftenResults = df.loc[category == 4]
    alwaysResults = df.loc[category == 5]

    data = {'Never':len(neverResults), 'Infrequently':len(infreqResults), 'Sometimes':len(someResults), 'Often':len(oftenResults), 'Always':len(alwaysResults)}
    category = list(data.keys())
    catValues = list(data.values())
    fig = plt.figure(figsize = (10,5))
    plt.bar(category,catValues,width = 0.4)
    plt.xlabel('User Response')
    plt.ylabel('No. of Users')
    plt.title('User Perceptiveness to ' + type + ' for ' + user)
    plt.show()

# graphPerceptiveness(surveyResults, 7, 'All Users', 'Location Tracking')
# graphPerceptiveness(surveyResults, 8, 'All Users', 'Sleep Tracking')
# graphPerceptiveness(surveyResults, 9, 'All Users', 'Heart Info Tracking')
# graphPerceptiveness(surveyResults, 10, 'All Users', 'Audio Tracking')
# graphPerceptiveness(surveyResults, 11, 'All Users', 'Third-Party Sharing')

# graphPerceptiveness(fitUsers, 7, 'Fitness Users', 'Location Tracking')
# graphPerceptiveness(fitUsers, 8, 'Fitness Users', 'Sleep Tracking')
# graphPerceptiveness(fitUsers, 9, 'Fitness Users', 'Heart Info Tracking')
# graphPerceptiveness(fitUsers, 10, 'Fitness Users', 'Audio Tracking')
# graphPerceptiveness(fitUsers, 11, 'Fitness Users', 'Third-Party Sharing')

# graphPerceptiveness(commUsers, 7, 'Communication Users', 'Location Tracking')
# graphPerceptiveness(commUsers, 8, 'Communication Users', 'Sleep Tracking')
# graphPerceptiveness(commUsers, 9, 'Communication Users', 'Heart Info Tracking')
# graphPerceptiveness(commUsers, 10, 'Communication Users', 'Audio Tracking')
# graphPerceptiveness(commUsers, 11, 'Communication Users', 'Third-Party Sharing')

# graphPerceptiveness(convUsers, 7, 'Convenience Users', 'Location Tracking')
# graphPerceptiveness(convUsers, 8, 'Convenience Users', 'Sleep Tracking')
# graphPerceptiveness(convUsers, 9, 'Convenience Users', 'Heart Info Tracking')
# graphPerceptiveness(convUsers, 10, 'Convenience Users', 'Audio Tracking')
# graphPerceptiveness(convUsers, 11, 'Convenience Users', 'Third-Party Sharing')

# graphPerceptiveness(styleUsers, 7, 'Style Users', 'Location Tracking')
# graphPerceptiveness(styleUsers, 8, 'Style Users', 'Sleep Tracking')
# graphPerceptiveness(styleUsers, 9, 'Style Users', 'Heart Info Tracking')
# graphPerceptiveness(styleUsers, 10, 'Style Users', 'Audio Tracking')
# graphPerceptiveness(styleUsers, 11, 'Style Users', 'Third-Party Sharing')