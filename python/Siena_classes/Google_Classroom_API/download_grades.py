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
SCOPES =  ['https://www.googleapis.com/auth/classroom.student-submissions.students.readonly https://www.googleapis.com/auth/classroom.courses.readonly https://www.googleapis.com/auth/classroom.coursework.students.readonly']
#SCOPES =  'https://www.googleapis.com/auth/classroom.courses.readonly'
#'''
#print(SCOPES)
#print()

# For PHYS 400: Nuclear and Particle Physics
course_id = '48793509327'

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
    print(course)
    #'''
    #work = service.courses().students().list(courseId=course_id).execute()
    work = service.courses().courseWork().list(courseId=course_id).execute()
    #work = service.courses().courseWork().studentSubmissions().list( courseId=course_id, courseWorkId='-',).execute()
    #userId=<user ID>).execute()
    print()
    print(work)
    print(work.keys())
    workids = []
    for w in work['courseWork']:
        print(w)
        print()
        workids.append(w['id'])
    print()
    print(workids)
    print()

    #submission = service.courses().courseWork().studentSubmissions().list( courseId=course_id, courseWorkId='-', userId=<user ID>).execute()
    submission = service.courses().courseWork().studentSubmissions().list( courseId=course_id, courseWorkId='-').execute()
    print(submission)

    courseWorkId = []
    userId =[]
    grade = []
    maxpoints = []
    date = []

    for s in submission['studentSubmissions']:
        print(s)
        print()

        if "assignedGrade" in list(s.keys()):
            grade.append(s['assignedGrade'])
            courseWorkId.append(s['courseWorkId'])
            date.append(s['creationTime'])
            userId.append(s['userId'])

    df = pd.DataFrame(np.array([courseWorkId, date, grade, userId]).transpose())

    
    print()

    if not course:
        print('No courses found.')
    else:
        print('Course:')
        print(course['name'])

    return submission, df

################################################################################
if __name__ == '__main__':
    s,df = main()
