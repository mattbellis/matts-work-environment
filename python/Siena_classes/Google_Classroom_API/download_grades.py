# https://developers.google.com/classroom/quickstart/python

from __future__ import print_function
import pickle
import os.path
import os
from googleapiclient import errors
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import simplejson

import pandas as pd
import numpy as np
import matplotlib,pylab as plt
import seaborn as sns

# So weird. I need this when I have multiple oauths because otherwise, it 
# changes the order. 
# https://stackoverflow.com/questions/53176162/google-oauth-scope-changed-during-authentication-but-scope-is-same
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'


# https://developers.google.com/classroom/guides/auth
# If modifying these scopes, delete the file token.pickle.
#SCOPES = [ 'https://www.googleapis.com/auth/classroom.student-submissions.students.readonly']
#SCOPES = ['https://www.googleapis.com/auth/classroom.courses.readonly',
          #'https://www.googleapis.com/auth/classroom.coursework.students.readonly']
#SCOPES = ['https://www.googleapis.com/auth/classroom.courses.readonly']
#'''
#SCOPES = [ 'https://www.googleapis.com/auth/classroom.student-submissions.students.readonly https://www.googleapis.com/auth/classroom.courses.readonly https://www.googleapis.com/auth/classroom.coursework.students.readonly']
SCOPES =  ['https://www.googleapis.com/auth/classroom.profile.photos  \
           https://www.googleapis.com/auth/classroom.profile.emails  \
           https://www.googleapis.com/auth/classroom.rosters.readonly  \
           https://www.googleapis.com/auth/classroom.rosters \
           https://www.googleapis.com/auth/classroom.student-submissions.students.readonly  \
           https://www.googleapis.com/auth/classroom.courses.readonly  \
           https://www.googleapis.com/auth/classroom.coursework.students.readonly']
#SCOPES =  'https://www.googleapis.com/auth/classroom.courses.readonly'
#'''
#print(SCOPES)
#print()

# For PHYS 400: Nuclear and Particle Physics
#course_id = '48793509327'
# For PHYS 250: Intro to Computational Physics
course_id = '48793509316'

def main():
    """Shows basic usage of the Classroom API.
    Prints the names of the first 10 courses the user has access to.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('classroom', 'v1', credentials=creds)

    # Get the information for the classroom
    #'''
    course = None
    try:
        course = service.courses().get(id=course_id).execute()
        print(u'Course "{0}" found.'.format(course.get('name')))
    except errors.HttpError as e:
        error = simplejson.loads(e.content).get('error')
        if(error.get('code') == 404):
            print(u'Course with ID "{0}" not found.'.format(course_id))
        else:
            raise

    # Call the Classroom API
    #results = service.courses().list(pageSize=10).execute()
    print("Printing course ---------------\n")
    print(course)
    #'''
    students = service.courses().students().list(courseId=course_id).execute()
    work = service.courses().courseWork().list(courseId=course_id).execute()

    ##########################################################################
    # Make dataframe of students
    ##########################################################################
    print("Printing students ------------------")
    print(students)
    print()
    students = students['students']
    student_dict = {}
    student_dict['userId'] = []
    student_dict['fullName'] = []
    student_dict['emailAddress'] = []
    for student in students:
        print("Student -------------")
        print(student)
        student_dict['userId'].append(student['userId'])
        student_dict['fullName'].append(student['profile']['name']['fullName'])
        student_dict['emailAddress'].append(student['profile']['emailAddress'])

    dfst = pd.DataFrame.from_dict(student_dict)

    ##########################################################################
    # Make dataframe of coursework
    ##########################################################################
    print("Printing coursework ------------------")
    print(work)
    print()
    work = work['courseWork']
    work_dict = {}
    work_dict['id'] = []
    work_dict['title'] = []
    work_dict['dueDate'] = []
    work_dict['maxPoints'] = []
    for w in work:
        print("Work -------------")
        print(w)
        work_dict['id'].append(w['id'])
        work_dict['title'].append(w['title'])
        work_dict['dueDate'].append(w['dueDate'])
        if 'maxPoints' in w:
            work_dict['maxPoints'].append(w['maxPoints'])
        else:
            work_dict['maxPoints'].append(-1)

    dfwo = pd.DataFrame.from_dict(work_dict)


    #submission = service.courses().courseWork().studentSubmissions().list( courseId=course_id, courseWorkId='-', userId=<user ID>).execute()
    submission = service.courses().courseWork().studentSubmissions().list( courseId=course_id, courseWorkId='-').execute()
    submission2 = service.courses().courseWork().studentSubmissions().list( courseId=course_id, courseWorkId='-',pageToken=submission['nextPageToken']).execute()
    ######## MAYBEWORKS NOW
    # NOT GETTING ALL SUBMISSIONS FOR STUDENTS!!!!!!!!!!!!
    # IS THIS A LIMIT????

    #submission = service.courses().courseWork().studentSubmissions().list( courseId=course_id ).execute()
    print("Print submission !!!!!!!!!!!!!!")
    print(submission)

    submission_dict = {}

    submission_dict['courseWorkId'] = []
    submission_dict['userId'] =[]
    submission_dict['grade'] = []

    for sub in [submission,submission2]:
        for s in sub['studentSubmissions']:
            print("Print studentsSubmissions s in loop ---------=====")
            print(s)
            print()

            if "assignedGrade" in list(s.keys()):
                submission_dict['grade'].append(float(s['assignedGrade']))
                submission_dict['courseWorkId'].append(s['courseWorkId'])
                submission_dict['userId'].append(s['userId'])

    df = pd.DataFrame.from_dict(submission_dict)

    print()

    if not course:
        print('No courses found.')
    else:
        print('Course:')
        print(course['name'])

    return submission, df, dfst, dfwo,course['name'],service

################################################################################
if __name__ == '__main__':
    s,df,dfst,dfwo,coursename,service = main()
    #df[2] = df[2].astype(int)
    #df.boxplot(2,by=[0])
    #sns.boxplot(data=df,x=0,y='percentage')
    #plt.show()
    filename = coursename.replace(' ','').replace('.','') + '.h5'
    df.to_hdf(filename, key='df', mode='w')
    dfst.to_hdf(filename, key='dfst', mode='a')
    dfwo.to_hdf(filename, key='dfwo', mode='a')
